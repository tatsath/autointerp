import os
import json
import fire
import importlib.util
from dataclasses import dataclass
from typing import List, Tuple, Iterable
from jaxtyping import Int, Float, Bool

from collections import defaultdict
import re
import math
import torch
from torch import Tensor
from datasets import load_dataset

from tqdm import tqdm

# Patch transformer_lens for Nemotron support (before importing HookedTransformer)
import transformer_lens.loading_from_pretrained as loading_from_pretrained
_original_get_official_model_name = loading_from_pretrained.get_official_model_name

def patched_get_official_model_name(model_name: str):
    """Patched version that supports Nemotron by mapping to Llama."""
    if "nemotron" in model_name.lower() or ("nvidia" in model_name.lower() and "nemotron" in model_name.lower()):
        # Nemotron is Llama-based, use Llama-3.1-8B-Instruct as base
        return "meta-llama/Llama-3.1-8B-Instruct"
    return _original_get_official_model_name(model_name)

loading_from_pretrained.get_official_model_name = patched_get_official_model_name

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from transformer_lens.hook_points import HookPoint
from sae_lens import SAE
from sae_lens.config import DTYPE_MAP as DTYPES
# TopK import - matches ReasonScore paper code structure
try:
    from sae_lens.sae import TopK
except ImportError:
    # If sae_lens.sae module not available, check activation function by name instead
    TopK = None
from sae_dashboard.feature_data_generator import FeatureMaskingContext


@dataclass
class SaeSelectionConfig:
    hook_point: str
    features: Iterable[int]
    minibatch_size_features: int = 256
    minibatch_size_tokens: int = 64
    device: str = "cpu"
    dtype: str = "float32"


def split_data(data, num_parts):
    """
    Split `data` into `num_parts` batches
    """
    k, m = divmod(len(data), num_parts)
    batches = [data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num_parts)]
    return batches


class RollingMean:
    def __init__(
        self, 
        tokens_of_interest: List[List[Tensor]], 
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None
    ):
        self.tokens_of_interest = tokens_of_interest
        self.ignore_tokens = ignore_tokens if ignore_tokens is not None else []
        self.expand_range = expand_range if expand_range is not None else (0, 0)

        # single-level statistics
        self._means = None
        self._counts = [0] * len(self.tokens_of_interest)
        
        # whole positive statistics
        self._mean_pos = None
        self._var_pos = None
        self._count_pos = 0

        # whole negative statistics
        self._mean_neg = None
        self._var_neg = None
        self._count_neg = 0
        
        # Track total positions for reporting
        self._total_positions = 0
        self._positive_positions = 0
        self._negative_positions = 0

    def _compute_single_mask(self, tokens: Int[Tensor, "batch seq"], ids_of_interest: Tensor):
        """Compute mask for a single token sequence with expansion."""
        seq_len = tokens.size(1)
        ids_len = len(ids_of_interest)
        
        mask = torch.zeros_like(tokens, dtype=torch.bool, device=tokens.device)
        if ids_len > seq_len:
            return mask

        ids_of_interest = ids_of_interest.view(1, 1, -1)
        windows = tokens.unfold(1, ids_len, 1)
        matches = (windows == ids_of_interest).all(dim=2)
        batch_indices, window_indices = torch.nonzero(matches, as_tuple=True)
        if len(batch_indices) == 0:
            return mask

        offsets = torch.arange(ids_len, device=tokens.device)
        spans = window_indices.unsqueeze(1) + offsets.unsqueeze(0)
        batch_expanded = batch_indices.unsqueeze(1).expand(-1, ids_len).reshape(-1)
        spans_flat = spans.reshape(-1)
        mask[batch_expanded, spans_flat] = True

        # Apply expand_range
        left, right = self.expand_range
        if left != 0 or right != 0:
            batch_indices, pos_indices = torch.nonzero(mask, as_tuple=True)
            if len(pos_indices) > 0:
                starts = torch.clamp(pos_indices - left, min=0)
                ends = torch.clamp(pos_indices + right, max=tokens.size(1) - 1)
                delta = torch.zeros(tokens.size(0), tokens.size(1) + 1, dtype=torch.int32, device=tokens.device)
                delta[batch_indices, starts] += 1
                delta[batch_indices, ends + 1] -= 1
                coverage = delta.cumsum(dim=1)
                coverage = coverage[:, :-1]  # Trim last column
                mask = coverage > 0

        return mask

    def _compute_update(
        self, tokens: Int[Tensor, "batch seq"], feature_acts: Float[Tensor, "batch seq n"],
        mask: Bool[Tensor, "batch seq"], acc_mean: Float[Tensor, "n"], acc_var: Float[Tensor, "n"], acc_count: float
    ):
        tokens = tokens[mask]
        feature_acts = feature_acts[mask]

        if tokens.numel() == 0:
            return acc_mean, acc_var, acc_count

        mean = feature_acts.mean(dim=0)
        count = feature_acts.size(0)

        upd_count = acc_count + count
        upd_mean = acc_mean + (count / upd_count) * (mean - acc_mean)
        
        # Online variance update using parallel algorithm
        if acc_count > 0:
            # Combined variance formula for merging two samples
            delta = mean - acc_mean
            upd_var = (acc_var * acc_count + feature_acts.var(dim=0, unbiased=False) * count + 
                      delta * delta * acc_count * count / upd_count) / upd_count
        else:
            upd_var = feature_acts.var(dim=0, unbiased=False)

        return upd_mean, upd_var, upd_count
    
    def update(self, tokens: Int[Tensor, "batch seq"], feature_acts: Float[Tensor, "batch seq n"]):
        assert tokens.ndim == 2 and feature_acts.ndim == 3, "tokens should be 2D, feature acts - 3D"
        assert tokens.size() == feature_acts.size()[:-1], "Batch and sequence dimensions must match"
        
        n = feature_acts.size(-1)

        if self._means is None:
            device = feature_acts.device
            dtype = feature_acts.dtype

            self._means = [
                torch.zeros(n, dtype=dtype, device=device)
                for _ in self.tokens_of_interest
            ]
            self._mean_pos = torch.zeros(n, dtype=dtype, device=device)
            self._var_pos = torch.zeros(n, dtype=dtype, device=device)
            self._mean_neg = torch.zeros(n, dtype=dtype, device=device)
            self._var_neg = torch.zeros(n, dtype=dtype, device=device)

            self.tokens_of_interest = [
                [seq.to(device) for seq in seq_group]
                for seq_group in self.tokens_of_interest
            ]
        
        if len(self.ignore_tokens) > 0:
            ignore_tensor = torch.tensor(self.ignore_tokens, dtype=torch.long, device=tokens.device)
            ignore_mask = torch.isin(tokens, ignore_tensor)
        else:
            ignore_mask = torch.zeros_like(tokens, dtype=torch.bool)
        
        mask_combined = torch.zeros_like(tokens, dtype=torch.bool)
        for i, seq_group in enumerate(self.tokens_of_interest):
            # Compute mask for one token
            group_mask = torch.zeros_like(tokens, dtype=torch.bool)
            for seq in seq_group:
                seq_mask = self._compute_single_mask(tokens, seq)
                group_mask |= seq_mask
            # Exclude ignored tokens
            group_mask = group_mask & (~ignore_mask)
            # Update rolling mean and count for the token (variance not needed for single tokens)
            if self._counts[i] == 0:
                # Initialize variance for single tokens (not used but needed for signature)
                dummy_var = torch.zeros(n, dtype=feature_acts.dtype, device=feature_acts.device)
                self._means[i], _, self._counts[i] = self._compute_update(
                    tokens, feature_acts, group_mask, self._means[i], dummy_var, self._counts[i]
                )
            else:
                dummy_var = torch.zeros(n, dtype=feature_acts.dtype, device=feature_acts.device)
                self._means[i], _, self._counts[i] = self._compute_update(
                    tokens, feature_acts, group_mask, self._means[i], dummy_var, self._counts[i]
                )
            # update mask
            mask_combined |= group_mask

        # Update 'positive' stats
        mask_pos = mask_combined & (~ignore_mask)
        pos_count = mask_pos.sum().item()
        self._mean_pos, self._var_pos, self._count_pos = self._compute_update(
            tokens, feature_acts, mask_pos, self._mean_pos, self._var_pos, self._count_pos
        )
        
        # Update 'negative' stats
        mask_neg = (~mask_combined) & (~ignore_mask)
        neg_count = mask_neg.sum().item()
        self._mean_neg, self._var_neg, self._count_neg = self._compute_update(
            tokens, feature_acts, mask_neg, self._mean_neg, self._var_neg, self._count_neg
        )
        
        # Track position counts
        total_batch_positions = (~ignore_mask).sum().item()
        self._total_positions += total_batch_positions
        self._positive_positions += pos_count
        self._negative_positions += neg_count

    def stats(self):
        """Returns tensors: means [n_features, 2], vars [n_features, 2], single_means [n_features, n_domain_tokens]."""
        single_means = torch.stack(self._means, dim=1) if len(self._means) > 0 else torch.tensor([])
        
        means = torch.stack([self._mean_pos, self._mean_neg], dim=1)
        vars = torch.stack([self._var_pos, self._var_neg], dim=1)

        return means, vars, single_means
    
    def position_stats(self):
        """Returns statistics about positive vs negative positions."""
        return {
            "total_positions": self._total_positions,
            "positive_positions": self._positive_positions,
            "negative_positions": self._negative_positions,
            "positive_percentage": (self._positive_positions / self._total_positions * 100) if self._total_positions > 0 else 0.0,
            "negative_percentage": (self._negative_positions / self._total_positions * 100) if self._total_positions > 0 else 0.0
        }


class FeatureStatisticsGenerator:
    """Generator used to accumulate domain-specific statistics for a batch of features. 
    
    Highly inspired by `sae_dashboard.FeatureDataGenerator`.
    """
    def __init__(
        self,
        cfg: SaeSelectionConfig,
        model: HookedTransformer,
        encoder: SAE,
        tokens: Int[Tensor, "batch seq"],
        domain_tokens: List[List[Tensor]],
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None
    ):
        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.token_minibatches = self.batch_tokens(tokens)
        self.domain_tokens = domain_tokens
        self.ignore_tokens = ignore_tokens
        self.expand_range = expand_range
        self.hook_layer = self.get_layer(self.cfg.hook_point)

    @torch.inference_mode()
    def batch_tokens(
        self, tokens: Int[Tensor, "batch seq"]
    ) -> list[Int[Tensor, "batch seq"]]:
        # Get tokens into minibatches, for the fwd pass
        token_minibatches = (
            (tokens,)
            if self.cfg.minibatch_size_tokens is None
            else tokens.split(self.cfg.minibatch_size_tokens)
        )
        token_minibatches = [tok.to(self.cfg.device) for tok in token_minibatches]

        return token_minibatches

    def get_layer(self, hook_point: str):
        """Get the layer (so we can do the early stopping in our forward pass)"""
        # Support both GPT-style (blocks.{layer}.{...}) and BERT-style (encoder.layer.{layer}.{...})
        layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
        if layer_match:
            return int(layer_match.group(1))
        # Try BERT/FinBERT format: encoder.layer.{layer}.{...}
        bert_match = re.search(r"encoder\.layer\.(\d+)\.", hook_point)
        if bert_match:
            return int(bert_match.group(1))
        assert False, f"Error: expecting hook_point to be 'blocks.{{layer}}.{{...}}' or 'encoder.layer.{{layer}}.{{...}}', but got {hook_point!r}"

    @torch.inference_mode()
    def get_feature_data(
        self,
        feature_indices: list[int],
    ):
        # Create objects to store the data for computing rolling stats
        feature_means = RollingMean(self.domain_tokens, self.ignore_tokens, self.expand_range)

        for i, minibatch in tqdm(
            enumerate(self.token_minibatches), desc="Statistics aggregation", 
            total=len(self.token_minibatches), leave=False
        ):
            minibatch.to(self.cfg.device)
            model_acts = self.get_model_acts(minibatch).to(self.encoder.device)

            # For TopK, compute all activations first, then select features
            # Matches ReasonScore paper: isinstance check for TopK activation function
            is_topk = False
            # Check multiple ways to detect TopK SAE
            if TopK is not None:
                is_topk = isinstance(self.encoder.activation_fn, TopK)
            
            # Fallback: check by activation function string name (matches paper logic)
            if not is_topk and hasattr(self.encoder, 'cfg'):
                if hasattr(self.encoder.cfg, 'activation_fn_str'):
                    is_topk = 'topk' in str(self.encoder.cfg.activation_fn_str).lower()
                # Also check class name
                if not is_topk:
                    encoder_class_name = self.encoder.__class__.__name__.lower()
                    is_topk = 'topk' in encoder_class_name
            
            if is_topk:
                # Get all features' activations
                all_features_acts = self.encoder.encode(model_acts)
                # Then select only the features we're interested in
                feature_acts = all_features_acts[:, :, feature_indices].to(
                    DTYPES[self.cfg.dtype]
                )
            else:
                # For other activation functions, use the masking context
                with FeatureMaskingContext(self.encoder, feature_indices):
                    feature_acts = self.encoder.encode(model_acts).to(
                        DTYPES[self.cfg.dtype]
                    )

            feature_means.update(minibatch, feature_acts)
            
        agg_means, agg_vars, agg_single_means = feature_means.stats()
        position_stats = feature_means.position_stats()

        return agg_means, agg_vars, agg_single_means, position_stats

    @torch.inference_mode()
    def get_model_acts(
        self, tokens: Int[Tensor, "batch seq"]
    ):
        # Special handling for Nemotron: need to extract from backbone.layers[layer].mixer (4480 dim)
        # transformer_lens doesn't correctly handle Nemotron's architecture
        if hasattr(self.encoder, 'cfg') and hasattr(self.encoder.cfg, 'd_in'):
            sae_d_in = self.encoder.cfg.d_in
            # Check if SAE expects 4480 dim (Nemotron mixer dimension)
            # Also check SAE config's model_name since transformer_lens maps Nemotron to Llama
            is_nemotron = False
            if hasattr(self.encoder.cfg, 'model_name'):
                is_nemotron = "nemotron" in str(self.encoder.cfg.model_name).lower()
            # Fallback: check if d_in is 4480 (Nemotron-specific dimension)
            if not is_nemotron and sae_d_in == 4480:
                is_nemotron = True
            
            if sae_d_in == 4480 and is_nemotron:
                print(f">>> Using raw transformers model for Nemotron mixer activations (dim {sae_d_in})...")
                from transformers import AutoModelForCausalLM
                import torch
                
                # Load raw transformers model if not already cached
                if not hasattr(self, '_nemotron_raw_model') or self._nemotron_raw_model is None:
                    print(f">>> Loading raw Nemotron model for layer {self.hook_layer}...")
                    # Use model_name from SAE config (original model name) or fallback
                    nemotron_model_name = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"  # Default
                    if hasattr(self.encoder.cfg, 'model_name'):
                        nemotron_model_name = self.encoder.cfg.model_name
                    print(f">>> Using model: {nemotron_model_name}")
                    
                    self._nemotron_raw_model = AutoModelForCausalLM.from_pretrained(
                        nemotron_model_name,
                        trust_remote_code=True,
                        device_map={"": self.cfg.device},
                        torch_dtype=torch.bfloat16
                    )
                    self._nemotron_raw_model.eval()
                
                # Extract mixer activations using forward hook (same approach as FinanceScore_Nemotron.py)
                mixer_activations = []
                def hook_fn(module, input, output):
                    # For Nemotron mixer, output might be a tuple, extract the first element
                    if isinstance(output, tuple):
                        mixer_activations.append(output[0].detach())
                    else:
                        mixer_activations.append(output.detach())
                
                handle = self._nemotron_raw_model.backbone.layers[self.hook_layer].mixer.register_forward_hook(hook_fn)
                
                with torch.no_grad():
                    _ = self._nemotron_raw_model(input_ids=tokens)
                
                handle.remove()
                
                if not mixer_activations:
                    raise RuntimeError("Could not extract mixer activations from Nemotron model")
                
                activation = mixer_activations[0]  # [batch, seq_len, 4480]
                print(f">>> Extracted Nemotron mixer activations: shape {activation.shape}")
                return activation
        
        # Standard transformer_lens path for other models
        def hook_fn_store_act(activation: Tensor, hook: HookPoint):
            hook.ctx["activation"] = activation

        hooks = [(self.cfg.hook_point, hook_fn_store_act)]

        self.model.run_with_hooks(
            tokens, stop_at_layer=self.hook_layer + 1, 
            fwd_hooks=hooks, return_type=None
        )

        activation = self.model.hook_dict[self.cfg.hook_point].ctx.pop(
            "activation"
        )

        return activation


class SaeSelectionRunner:
    """Runner used to collect domain-specific statistics.
    
    Highly inspired by `sae_dashboard.SaeVisRunner`.
    """
    def __init__(self, cfg: SaeSelectionConfig):
        self.cfg = cfg
        self.position_stats = None

    @torch.inference_mode()
    def run(
        self, 
        encoder: SAE,
        model: HookedTransformer, 
        tokens: Int[Tensor, "batch seq"],
        domain_tokens: List[List[Tensor]],
        ignore_tokens: List[int] = None,
        expand_range: Tuple[int, int] = None,
        alpha: float = 1.0,
        epsilon: float = 1e-12,
        score_type: str = "domain"
    ):
        # Try to fold W_dec_norm, but skip for TopK SAEs where it's not safe
        try:
            encoder.fold_W_dec_norm()
        except NotImplementedError as e:
            if "TopKSAE" in str(e) or "topk" in str(e).lower():
                print(f">>> Skipping fold_W_dec_norm() for TopK SAE (not safe for TopK)")
            else:
                raise

        features_list = self.handle_features(self.cfg.features, encoder)
        feature_batches = self.get_feature_batches(features_list)

        feature_statistics_generator = FeatureStatisticsGenerator(
            self.cfg, model, encoder, tokens, domain_tokens, ignore_tokens, expand_range
        )

        all_feature_means, all_feature_vars, all_feature_single_means = [], [], []
        position_stats_aggregated = None
        for features in tqdm(feature_batches, total=len(feature_batches), desc="Feature Selection"):
            feature_means, feature_vars, feature_single_means, position_stats = feature_statistics_generator.get_feature_data(features)

            all_feature_means.append(feature_means)
            all_feature_vars.append(feature_vars)
            all_feature_single_means.append(feature_single_means)
            
            # Aggregate position stats (they should be the same across all feature batches)
            if position_stats_aggregated is None:
                position_stats_aggregated = position_stats

        all_feature_means = torch.concat(all_feature_means, dim=0)  # [d_sae, 2]
        all_feature_vars = torch.concat(all_feature_vars, dim=0)  # [d_sae, 2]
        all_feature_single_means = torch.concat(all_feature_single_means, dim=0)  # [d_sae, |domain_tokens|]

        # Print position statistics
        if position_stats_aggregated:
            print("\n" + "=" * 80)
            print("Position Statistics: Dataset vs Provided Tokens")
            print("=" * 80)
            print(f"Total token positions processed: {position_stats_aggregated['total_positions']:,}")
            
            if position_stats_aggregated['positive_positions'] == 0:
                print(f"Positions matching provided tokens (positive): 0 (0.00%)")
                print(f"Positions NOT matching provided tokens (negative): {position_stats_aggregated['negative_positions']:,} (100.00%)")
                print("\nInterpretation:")
                print("  • 0.00% of positions match provided tokens (no tokens provided or no matches found)")
                print("  • 100.00% of positions are negative (all positions from dataset)")
                print("  • This is a dataset-only search (similar to ReasonScore when no reasoning tokens match)")
            else:
                print(f"Positions matching provided tokens (positive): {position_stats_aggregated['positive_positions']:,} ({position_stats_aggregated['positive_percentage']:.2f}%)")
                print(f"Positions NOT matching provided tokens (negative): {position_stats_aggregated['negative_positions']:,} ({position_stats_aggregated['negative_percentage']:.2f}%)")
                print("\nInterpretation (ReasonScore-style):")
                print(f"  • {position_stats_aggregated['positive_percentage']:.2f}% of positions match your provided tokens (positive)")
                print(f"  • {position_stats_aggregated['negative_percentage']:.2f}% of positions don't match (negative)")
                print("  • Both positive and negative positions come from the same dataset")
                print("  • Positive = positions where domain tokens appear (e.g., reasoning words)")
                print("  • Negative = all other positions in the dataset")
            
            print("=" * 80 + "\n")

        # Compute scores based on score_type
        if score_type == "simple":
            # Simple Reasoning Score: |μ⁺ - μ⁻|
            scores = torch.abs(all_feature_means[:, 0] - all_feature_means[:, 1])
        elif score_type == "fisher":
            # Fisher-style Score: (μ⁺ - μ⁻)² / (σ⁺ + σ⁻ + ε)
            mean_diff = all_feature_means[:, 0] - all_feature_means[:, 1]
            std_pos = torch.sqrt(all_feature_vars[:, 0] + epsilon)
            std_neg = torch.sqrt(all_feature_vars[:, 1] + epsilon)
            scores = (mean_diff ** 2) / (std_pos + std_neg + epsilon)
        else:  # "domain" - original domain-specific scoring
            # compute entropy
            probs = all_feature_single_means / (all_feature_single_means.sum(dim=1, keepdim=True) + epsilon)
            log_probs = torch.where(probs > 0, torch.log(probs), 0)
            h = -(probs * log_probs).sum(dim=1)
            # normalize
            if probs.size(1) > 1:
                h_norm = h / math.log(probs.size(1))
            else: # do not compute entropy
                h_norm = torch.ones_like(h)
            
            # compute DomainScore
            sum_pos = all_feature_means[:, 0].sum(dim=0, keepdim=True) + epsilon
            sum_neg = all_feature_means[:, 1].sum(dim=0, keepdim=True) + epsilon
            
            scores = (
                (all_feature_means[:, 0] / sum_pos) * h_norm**alpha
                - (all_feature_means[:, 1] / sum_neg)
            )

        # Store position stats as an attribute for later retrieval if needed
        self.position_stats = position_stats_aggregated

        return scores

    def handle_features(
        self, features: Iterable[int] | None, encoder_wrapper: SAE
    ) -> list[int]:
        if features is None:
            return list(range(encoder_wrapper.cfg.d_sae))
        else:
            return list(features)

    def get_feature_batches(self, features_list: list[int]) -> list[list[int]]:
        # Break up the features into batches
        feature_batches = [
            x.tolist()
            for x in torch.tensor(features_list).split(self.cfg.minibatch_size_features)
        ]
        return feature_batches


def compute_score(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    output_dir: str,
    tokens_str_path: str = None,
    sae_id: str = None,
    expand_range: Tuple[int, int] = None,
    ignore_tokens: List[int] = None,
    n_samples: int = 4096,
    alpha: float = 1.0,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    num_chunks: int = 1,
    chunk_num: int = 0,
    score_type: str = "domain"
):
    """Compute domain-specific feature scores."""
    # Check if CUDA is available and find a free GPU
    device = "cpu"
    if torch.cuda.is_available():
        import subprocess
        # Check GPU availability
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            gpu_info = result.stdout.strip().split('\n')
            free_gpu = None
            for line in gpu_info:
                idx, mem_used, mem_total, util = line.split(', ')
                mem_used = int(mem_used)
                mem_total = int(mem_total)
                util = int(util)
                # Consider GPU free if < 1GB used and < 10% utilization
                if mem_used < 1024 and util < 10:
                    free_gpu = int(idx)
                    break
            
            if free_gpu is not None:
                device = f"cuda:{free_gpu}"
                print(f">>> Using GPU {free_gpu} (free: {mem_total - mem_used}MB available)")
            else:
                # Fallback: try default cuda
                try:
                    test_tensor = torch.zeros(1, device="cuda:0")
                    del test_tensor
                    torch.cuda.empty_cache()
                    device = "cuda:0"
                    print(">>> Using default GPU 0")
                except RuntimeError:
                    device = "cpu"
                    print(">>> Warning: No free GPUs found, using CPU")
        except Exception as e:
            print(f">>> Warning: Could not check GPU status ({e}), trying default GPU")
            try:
                test_tensor = torch.zeros(1, device="cuda:0")
                del test_tensor
                torch.cuda.empty_cache()
                device = "cuda:0"
            except RuntimeError:
                device = "cpu"
                print(">>> Warning: CUDA unavailable, using CPU")

    print(">>> Loading SAE and LLM")
    # Check if sae_path is a local directory (not a HuggingFace repo)
    import os
    is_local_path = os.path.exists(sae_path) and os.path.isdir(sae_path)
    layer_match = None  # Fix for FinBERT: initialize to avoid UnboundLocalError
    
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    elif is_local_path:
        # For local paths, try load_from_pretrained first (matches ReasonScore paper approach)
        # If that fails, fall back to manual loading
        layer_match = re.match(r"blocks\.(\d+)\.", sae_id)
        if layer_match:
            layer_num = int(layer_match.group(1))
            layer_path = os.path.join(sae_path, f"layers.{layer_num}")
            
            # Check if SAE is in trainer_0 format (ae.pt directly in sae_path)
            ae_pt_path = os.path.join(sae_path, "ae.pt")
            if not os.path.exists(layer_path) and os.path.exists(ae_pt_path):
                # Handle trainer_0 format: load from ae.pt directly
                print(f">>> Loading SAE from trainer format: {sae_path}")
                try:
                    # Try loading as HuggingFace format with sae_id
                    sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)
                    print(f">>> Successfully loaded SAE using from_pretrained")
                except Exception as e:
                    # Fallback: try loading ae.pt directly
                    print(f">>> Attempting to load from ae.pt...")
                    try:
                        checkpoint = torch.load(ae_pt_path, map_location=device)
                        # The checkpoint might have different key structures
                        # Try to construct SAE from checkpoint
                        from sae_lens import SAEConfig, TopKSAE, StandardSAE
                        
                        # Load config
                        cfg_file = os.path.join(sae_path, "config.json")
                        if os.path.exists(cfg_file):
                            with open(cfg_file, 'r') as f:
                                cfg_dict = json.load(f)
                            
                            # Extract trainer config
                            trainer_cfg = cfg_dict.get('trainer', {})
                            dict_class = trainer_cfg.get('dict_class', 'BatchTopKSAE')
                            # Determine architecture based on dict_class
                            if 'TopK' in dict_class:
                                architecture = 'topk'
                                activation_fn_str = 'topk'
                            else:
                                architecture = 'standard'
                                activation_fn_str = 'relu'
                            
                            sae_cfg_dict = {
                                'architecture': architecture,
                                'd_sae': trainer_cfg.get('dict_size', 35840),
                                'd_in': trainer_cfg.get('activation_dim', 4480),
                                'hook_name': sae_id,
                                'hook_layer': trainer_cfg.get('layer', layer_num),
                                'activation_fn_str': activation_fn_str,
                                'k': trainer_cfg.get('k', 64) if architecture == 'topk' else None,
                                'device': device
                            }
                            
                            sae_cfg = SAEConfig.from_dict(sae_cfg_dict)
                            if architecture == 'topk':
                                sae = TopKSAE(sae_cfg).to(device)
                            else:
                                sae = StandardSAE(sae_cfg).to(device)
                            
                            # Load weights from checkpoint - convert Nemotron format to sae_lens format
                            if 'state_dict' in checkpoint:
                                state_dict = checkpoint['state_dict']
                            elif 'model' in checkpoint:
                                state_dict = checkpoint['model']
                            else:
                                state_dict = checkpoint
                            
                            # Convert Nemotron keys to sae_lens format
                            converted_state_dict = {}
                            for k, v in state_dict.items():
                                if "encoder.weight" in k:
                                    # encoder.weight is [d_sae, d_in], transpose to [d_in, d_sae] for W_enc
                                    converted_state_dict["W_enc"] = v.T.contiguous()
                                elif "encoder.bias" in k:
                                    converted_state_dict["b_enc"] = v.contiguous()
                                elif "decoder.weight" in k:
                                    # decoder.weight is [d_in, d_sae], but sae_lens expects [d_sae, d_in]
                                    # So we need to transpose it
                                    converted_state_dict["W_dec"] = v.T.contiguous()
                                elif "decoder.bias" in k or "b_dec" in k:
                                    converted_state_dict["b_dec"] = v.contiguous()
                                else:
                                    # Keep other keys as-is (e.g., 'k', 'threshold')
                                    converted_state_dict[k] = v
                            
                            sae.load_state_dict(converted_state_dict, strict=False)
                            
                            print(f">>> Successfully loaded SAE from ae.pt")
                        else:
                            raise RuntimeError(f"Config file not found: {cfg_file}")
                    except Exception as e2:
                        raise RuntimeError(f"Failed to load SAE from {sae_path}: {e2}")
            elif os.path.exists(layer_path):
                print(f">>> Loading SAE from local path: {layer_path}")
                # Try the ReasonScore paper approach first
                # Create temporary symlink if sae.safetensors exists but sae_weights.safetensors doesn't
                sae_file = os.path.join(layer_path, "sae.safetensors")
                sae_weights_file = os.path.join(layer_path, "sae_weights.safetensors")
                symlink_created = False
                if os.path.exists(sae_file) and not os.path.exists(sae_weights_file):
                    try:
                        os.symlink("sae.safetensors", sae_weights_file)
                        symlink_created = True
                    except OSError:
                        pass  # Symlink might already exist or permission issue
                
                try:
                    sae = SAE.load_from_pretrained(layer_path, device=device)
                    print(f">>> Successfully loaded SAE using load_from_pretrained (matches ReasonScore paper)")
                except Exception as e:
                    # Clean up symlink if created
                    if symlink_created and os.path.exists(sae_weights_file):
                        try:
                            os.remove(sae_weights_file)
                        except OSError:
                            pass
                    
                    # If key mismatch, try converting keys and reloading (minimal deviation from paper)
                    if "encoder.weight" in str(e) or "encoder.bias" in str(e):
                        print(f">>> Key format mismatch detected, converting keys to match sae_lens format...")
                        from safetensors import safe_open
                        import shutil
                        
                        # Create temporary converted file
                        temp_file = os.path.join(layer_path, "sae_weights_temp.safetensors")
                        with safe_open(sae_file, framework="pt", device="cpu") as src:
                            from safetensors.torch import save_file
                            converted_dict = {}
                            if "encoder.weight" in src.keys():
                                # encoder.weight is [d_sae, d_in], transpose to [d_in, d_sae] for W_enc
                                converted_dict["W_enc"] = src.get_tensor("encoder.weight").T.contiguous()
                            if "encoder.bias" in src.keys():
                                converted_dict["b_enc"] = src.get_tensor("encoder.bias").contiguous()
                            if "W_dec" in src.keys():
                                converted_dict["W_dec"] = src.get_tensor("W_dec").contiguous()
                            if "b_dec" in src.keys():
                                converted_dict["b_dec"] = src.get_tensor("b_dec").contiguous()
                            
                            save_file(converted_dict, temp_file)
                        
                        # Replace sae_weights.safetensors with converted version
                        if os.path.exists(sae_weights_file):
                            os.remove(sae_weights_file)
                        os.rename(temp_file, sae_weights_file)
                        
                        # Try loading again
                        try:
                            sae = SAE.load_from_pretrained(layer_path, device=device)
                            print(f">>> Successfully loaded SAE after key conversion (matches ReasonScore paper)")
                        except Exception as e2:
                            raise RuntimeError(f"Failed to load SAE from {layer_path} even after key conversion: {e2}")
                    else:
                        raise RuntimeError(f"Failed to load SAE from {layer_path} using ReasonScore approach: {e}")
            else:
                raise FileNotFoundError(f"Layer path not found: {layer_path}")
        else:
            raise ValueError(f"Could not parse sae_id to extract layer number: {sae_id}")
    else:
        # HuggingFace repo path - matches ReasonScore paper approach
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    # Load model - handle models that require trust_remote_code
    # Common models that need this: Nemotron, some custom models
    models_requiring_trust_remote_code = ["nemotron", "nvidia"]
    needs_trust_remote_code = any(
        keyword in model_path.lower() for keyword in models_requiring_trust_remote_code
    )
    
    try:
        if needs_trust_remote_code:
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device=device,
                trust_remote_code=True
            )
        else:
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device=device,
            )
    except Exception as e:
        # If loading fails, try with trust_remote_code=True as fallback
        if "trust_remote_code" not in str(e).lower() and not needs_trust_remote_code:
            print(f">>> Warning: Model loading failed, retrying with trust_remote_code=True...")
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device=device,
                trust_remote_code=True
            )
        else:
            raise
    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    print("\n" + "=" * 80)
    print("📊 Step 1: Loading Dataset")
    print("=" * 80)
    print(f">>> Loading dataset: {dataset_path}")
    # Check if it's a saved dataset (from save_to_disk)
    if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
    else:
        dataset = load_dataset(dataset_path, streaming=False, split="train")
    print(f">>> Dataset loaded: {len(dataset):,} samples")
    if column_name == "tokens":
        token_dataset = dataset.shuffle(seed=42)
        print(f">>> Using pre-tokenized dataset: {len(token_dataset):,} samples")
    else:
        print(f">>> Tokenizing dataset (column: {column_name})...")
        # Get context_size - matches ReasonScore paper approach
        context_size = getattr(sae.cfg, 'context_size', None)
        if context_size is None:
            # Try to get from metadata
            if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
                context_size = sae.cfg.metadata.get('context_size', None)
        if context_size is None:
            # Fallback: try to get from config file or use default
            context_size = 1024  # Default context size
            if is_local_path:
                # Try SAE path directly first (for FinBERT)
                cfg_file = os.path.join(sae_path, "cfg.json")
                if os.path.exists(cfg_file):
                    with open(cfg_file, 'r') as f:
                        cfg_dict = json.load(f)
                        context_size = cfg_dict.get('context_size', context_size)
                # Also try layer subdirectory (for other models)
                elif layer_match:
                    layer_path = os.path.join(sae_path, f"layers.{int(layer_match.group(1))}")
                    cfg_file = os.path.join(layer_path, "cfg.json")
                    if os.path.exists(cfg_file):
                        with open(cfg_file, 'r') as f:
                            cfg_dict = json.load(f)
                            context_size = cfg_dict.get('context_size', context_size)
        
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=model.tokenizer,
            streaming=False,
            max_length=context_size,
            column_name=column_name,
            add_bos_token=getattr(sae.cfg, 'prepend_bos', False),
            num_proc=4
        ).shuffle(seed=42)
    
    # Handle optional tokens
    if tokens_str_path is None or tokens_str_path == "":
        print(">>> No tokens file provided - search will be 100% dataset-driven")
        domain_tokens = []
        tokens_provided = False
    else:
        if not os.path.exists(tokens_str_path):
            print(f">>> Warning: tokens file not found at {tokens_str_path}, using dataset-only search")
            domain_tokens = []
            tokens_provided = False
        else:
            with open(tokens_str_path, 'r') as file:
                tokens_str = json.load(file)
            
            if not tokens_str or len(tokens_str) == 0:
                print(">>> Warning: tokens file is empty, using dataset-only search")
                domain_tokens = []
                tokens_provided = False
            else:
                print(">>> tokens str: {}".format(tokens_str))
                grouped_tokens = defaultdict(list)
                for str_token in tokens_str:
                    # we treat " A", "A", " a", "a" as the same token
                    normalized_str = str_token.lstrip().lower()
                    token_ids = model.tokenizer.encode(str_token, add_special_tokens=False)
                    grouped_tokens[normalized_str].append(torch.tensor(token_ids, dtype=torch.long))
                domain_tokens = list(grouped_tokens.values())
                print(">>> tokens ids: {}".format(domain_tokens))
                tokens_provided = True
    
    if not tokens_provided:
        print(">>> Note: Without provided tokens, all positions are considered 'negative' (dataset-driven)")
        print(">>>       Scoring methods that require positive/negative comparison may not work optimally")
    
    if expand_range is not None:
        print(">>> Using expansion: {}".format(expand_range))
    if ignore_tokens is not None:
        print(">>> Using ignore tokens: {}".format(ignore_tokens))
    
    features = list(range(sae.cfg.d_sae))
    if num_chunks > 1:
        features = split_data(features, num_chunks)[chunk_num]
        print(f">>> Processing features in chunks. Current chunk: {chunk_num}, size: {len(features)}")

    # Get hook_name - matches ReasonScore paper approach
    # For FinBERT (encoder-only), use hook_name from config metadata
    hook_name = None
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        hook_name = sae.cfg.metadata.get('hook_name', None)
    if not hook_name:
        hook_name = getattr(sae.cfg, 'hook_name', None)
    if not hook_name and is_local_path and os.path.exists(sae_path):
        # Fallback: read from config file directly
        cfg_file = os.path.join(sae_path, "cfg.json")
        if os.path.exists(cfg_file):
            with open(cfg_file, 'r') as f:
                cfg_dict = json.load(f)
            hook_name = cfg_dict.get('hook_name', None)
    if not hook_name:
        hook_name = sae_id if sae_id else sae_path
    # If hook_name is still a path, extract the hook name from it
    if os.path.exists(hook_name) or '/' in hook_name:
        # Extract hook name from path like: .../encoder.layer.10.output -> encoder.layer.10.output
        if 'encoder.layer.' in hook_name:
            match = re.search(r'(encoder\.layer\.\d+\.\w+)', hook_name)
            if match:
                hook_name = match.group(1)
    # Convert FinBERT hook format to transformer_lens format
    # encoder.layer.10.output -> blocks.10.hook_resid_post
    if 'encoder.layer.' in hook_name:
        match = re.search(r'encoder\.layer\.(\d+)\.', hook_name)
        if match:
            layer_num = match.group(1)
            hook_name = f"blocks.{layer_num}.hook_resid_post"
    print(f">>> Using hook_point: {hook_name}")
    
    sae_selection_cfg = SaeSelectionConfig(
        hook_point=hook_name,
        features=features,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        device=device,
        dtype="float32"
    )

    print("\n" + "=" * 80)
    print("📊 Step 2: Computing Feature Scores")
    print("=" * 80)
    print(f">>> Processing {n_samples:,} samples")
    print(f">>> Scoring {len(features):,} features using '{score_type}' method")
    if tokens_provided:
        print(f">>> Using {len(domain_tokens)} domain token groups")
    print(">>> This may take several minutes...")
    print()
    
    runner = SaeSelectionRunner(sae_selection_cfg)
    feature_scores = runner.run(
        encoder=sae,
        model=model,
        tokens=token_dataset["tokens"][:n_samples],
        domain_tokens=domain_tokens,
        ignore_tokens=ignore_tokens,
        expand_range=expand_range,
        alpha=alpha,
        score_type=score_type
    )
    
    print("\n>>> Feature scoring complete!")

    # save feature scores
    print("\n" + "=" * 80)
    print("📊 Step 3: Saving Results")
    print("=" * 80)
    os.makedirs(output_dir, exist_ok=True)
    output_name = "feature_scores.pt" if num_chunks == 1 else "feature_scores_{}.pt".format(chunk_num)
    output_path = os.path.join(output_dir, output_name)
    torch.save(feature_scores.cpu(), output_path)
    print(f">>> Feature scores saved to: {output_path}")
    
    # Save position statistics to JSON file
    if hasattr(runner, 'position_stats') and runner.position_stats:
        position_stats_path = os.path.join(output_dir, "position_statistics.json")
        with open(position_stats_path, 'w') as f:
            json.dump(runner.position_stats, f, indent=2)
        print(f">>> Position statistics saved to: {position_stats_path}")

if __name__ == "__main__":
    fire.Fire(compute_score)

