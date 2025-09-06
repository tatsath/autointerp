import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from safetensors.numpy import load_file
from sparsify import Sae
from torch import nn
from tqdm import tqdm


class NeighbourCalculator:
    """
    Class to compute the neighbours of selected latents using different methods:
    - similarity: uses autoencoder weights
    - correlation: uses pre-activation records and autoencoder
    - co-occurrence: uses latent dataset statistics
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        autoencoder: Optional[nn.Module] = None,
        number_of_neighbours: int = 10,
        neighbour_cache: Optional[dict[int, list[tuple[int, float]]]] = None,
    ):
        """
        Initialize a NeighbourCalculator.

        Args:
            latent_dataset (Optional[LatentDataset]): Dataset
                containing latent activations
            autoencoder (Optional[Autoencoder]): The trained autoencoder model
            residual_stream_record (Optional[ResidualStreamRecord]): Record of
                residual stream values
        """
        self.cache_dir = cache_dir
        self.autoencoder = autoencoder
        # self.residual_stream_record = residual_stream_record
        self.number_of_neighbours = number_of_neighbours
        # load the neighbour cache from the path
        if neighbour_cache is not None:
            self.neighbour_cache = neighbour_cache
        else:
            # dictionary to cache computed neighbour lists
            self.neighbour_cache: dict[int, list[tuple[int, float]]] = {}

    def _compute_neighbour_list(
        self,
        method: Literal["encoder_similarity", "decoder_similarity", "co-occurrence"],
    ) -> None:
        """
        Compute complete neighbour lists using specified method.

        Args:
            method (str): One of 'similarity', 'correlation', or 'co-occurrence'
        """
        if method == "encoder_similarity":
            self.method = "encoder_similarity"
            self.neighbour_cache = self._compute_similarity_neighbours("encoder")
        elif method == "decoder_similarity":
            self.method = "decoder_similarity"
            self.neighbour_cache = self._compute_similarity_neighbours("decoder")
        elif method == "co-occurrence":
            self.method = "co-occurrence"
            self.neighbour_cache = self._compute_cooccurrence_neighbours()

        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'encoder similarity',"
                "'decoder similarity', or 'co-occurrence'"
            )

    def _compute_similarity_neighbours(
        self, method: Literal["encoder", "decoder"]
    ) -> dict[int, list[tuple[int, float]]]:
        """
        Compute neighbour lists based on weight similarity in the autoencoder.
        """
        assert (
            self.autoencoder is not None
        ), "Autoencoder is required for similarity-based neighbours"
        print("Computing similarity neighbours")
        # We use the encoder vectors to compute the similarity between latents
        if method == "encoder":
            encoder = self.autoencoder.encoder.weight.data.cuda()  # type: ignore
            weight_matrix_normalized = encoder / encoder.norm(dim=1, keepdim=True)  # type: ignore

        elif method == "decoder":
            # TODO: we would probably go around this by
            # having a autoencoder wrapper
            assert isinstance(
                self.autoencoder, Sae
            ), "Autoencoder must be a sparsify.Sae for decoder similarity"
            decoder = self.autoencoder.W_dec.data.cuda()  # type:ignore
            weight_matrix_normalized = decoder / decoder.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'encoder' or 'decoder'")

        wT = weight_matrix_normalized.T
        # Compute the similarity between latents
        done = False
        batch_size = weight_matrix_normalized.shape[0]
        number_latents = batch_size

        neighbour_lists = {}
        while not done:
            try:
                for start in tqdm(range(0, number_latents, batch_size)):
                    rows = wT[start : start + batch_size]
                    similarity_matrix = weight_matrix_normalized @ rows
                    # remove nan values
                    similarity_matrix = torch.nan_to_num(similarity_matrix, 0)
                    indices, values = torch.topk(
                        similarity_matrix, self.number_of_neighbours + 1, dim=1
                    )
                    neighbour_lists.update(
                        {
                            i
                            + start: list(
                                zip(indices[i].tolist()[1:], values[i].tolist()[1:])
                            )
                            for i in range(len(indices))
                        }
                    )
                    del similarity_matrix
                    torch.cuda.empty_cache()
                done = True
            except RuntimeError:  # Out of memory
                batch_size = batch_size // 2
                if batch_size < 2:
                    raise ValueError(
                        "Batch size is too small to compute similarity matrix. "
                        "You don't have enough memory."
                    )

        return neighbour_lists

    def _compute_cooccurrence_neighbours(self) -> dict[int, list[tuple[int, float]]]:
        """
        Compute neighbour lists based on feature co-occurrence in the dataset.
        If you run out of memory try reducing the token_batch_size
        Code adapted from https://github.com/taha-yassine/SAE-features/blob/main/cooccurrences/compute.py
        """

        print("Computing co-occurrence neighbours")
        assert (
            self.cache_dir is not None
        ), "Cache directory is required for co-occurrence-based neighbours"
        paths = os.listdir(self.cache_dir)

        all_locations = []
        for path in paths:
            if path.endswith(".safetensors"):
                split_data = load_file(self.cache_dir / path)
                first_feature = int(path.split("/")[-1].split("_")[0])
                locations = torch.tensor(split_data["locations"].astype(np.int64))
                locations[:, 2] = locations[:, 2] + first_feature

                all_locations.append(locations)

        # concatenate the locations and activations
        locations = torch.cat(all_locations)

        batch_index = locations[:, 0]
        ctx_index = locations[:, 1]
        latent_index = locations[:, 2]

        n_latents = int(latent_index.max().item()) + 1
        ctx_len = locations[:, 1].max().item() + 1
        n_batches = locations[:, 0].max().item() + 1
        # Convert from (batch_id, ctx_pos) to a unique 1D index

        idx_cantor = (
            (batch_index + ctx_index) * (batch_index + ctx_index + 1)
        ) // 2 + ctx_index

        # Sort the indices, because they are not sorted after concatenation
        idx_cantor, idx_cantor_sorted_idx = idx_cantor.sort(dim=0, stable=True)
        latent_index = latent_index[idx_cantor_sorted_idx]

        token_batch_size = 100_000
        co_occurrence_matrix = None
        done = False
        while not done:
            try:
                # Find indices where idx_cantor crosses each batch boundary
                batch_boundaries = torch.arange(
                    0, n_batches, token_batch_size // ctx_len
                )
                cantor_batch_boundaries = (
                    (batch_boundaries + ctx_len) * (batch_boundaries + ctx_len + 1)
                ) // 2 + ctx_len

                batch_boundaries_tensor = torch.searchsorted(
                    idx_cantor, cantor_batch_boundaries
                )
                batch_boundaries = [0] + batch_boundaries_tensor.tolist()

                if batch_boundaries[-1] != len(idx_cantor):
                    batch_boundaries.append(len(idx_cantor))
                co_occurrence_matrix = torch.zeros(
                    (n_latents, n_latents), dtype=torch.int64
                )

                for start, end in tqdm(
                    zip(batch_boundaries[:-1], batch_boundaries[1:])
                ):
                    # get all ind_cantor values between start and start
                    selected_idx_cantor = idx_cantor[start:end]
                    # shift the indices to start from 0
                    selected_idx_cantor = selected_idx_cantor - selected_idx_cantor[0]
                    selected_latent_index = latent_index[start:end]
                    number_positions = int(selected_idx_cantor.max().item()) + 1

                    # create a sparse matrix of the selected indices
                    sparse_matrix_indices = torch.stack(
                        [selected_latent_index, selected_idx_cantor], dim=0
                    )
                    sparse_matrix = torch.sparse_coo_tensor(
                        sparse_matrix_indices,
                        torch.ones(len(selected_latent_index)),
                        (n_latents, number_positions),
                        check_invariants=True,
                    )
                    sparse_matrix = sparse_matrix.cuda()
                    partial_cooc = (sparse_matrix @ sparse_matrix.T).cpu()
                    co_occurrence_matrix += partial_cooc.int()

                    del sparse_matrix, partial_cooc
                    torch.cuda.empty_cache()
                done = True

            except RuntimeError:  # Out of memory
                token_batch_size = token_batch_size // 2
                if token_batch_size < 2:
                    raise ValueError(
                        "Batch size is too small to compute similarity matrix. "
                        "You don't have enough memory."
                    )

        # Compute Jaccard similarity
        def compute_jaccard(cooc_matrix):
            self_occurrence = cooc_matrix.diagonal()
            jaccard_matrix = cooc_matrix / (
                self_occurrence[:, None] + self_occurrence - cooc_matrix
            )
            # remove the diagonal and keep the upper triangle
            return jaccard_matrix

        # Compute Jaccard similarity matrix
        assert co_occurrence_matrix is not None, "Co-occurrence matrix is not computed"
        jaccard_matrix = compute_jaccard(co_occurrence_matrix)

        # get the indices of the top k neighbours for each feature
        top_k_indices, values = torch.topk(
            jaccard_matrix, self.number_of_neighbours + 1, dim=1
        )
        del jaccard_matrix, co_occurrence_matrix
        torch.cuda.empty_cache()

        # return the neighbour lists
        return {
            i: list(zip(top_k_indices[i].tolist()[1:], values[i].tolist()[1:]))
            for i in range(len(top_k_indices))
        }

    def populate_neighbour_cache(
        self,
        method: Literal["encoder_similarity", "decoder_similarity", "co-occurrence"],
    ) -> None:
        """
        Populate the neighbour cache with the computed neighbour lists
        """
        self._compute_neighbour_list(method)

    def save_neighbour_cache(self, path: str) -> None:
        """
        Save the neighbour cache to the path as a json file
        """
        with open(path + f"-{self.method}.json", "w") as f:
            json.dump(self.neighbour_cache, f)

    def load_neighbour_cache(self, path: str) -> dict[str, dict[int, list[int]]]:
        """
        Load the neighbour cache from the path as a json file
        """
        with open(path, "r") as f:
            return json.load(f)
