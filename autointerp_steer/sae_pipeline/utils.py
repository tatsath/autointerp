from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sae_pipeline.sae import JumpReLUSAE
import h5py
from tqdm import tqdm
import numpy as np
from sae_pipeline.sae import get_SAE_features


def load_model(model_name: str, model_token: str, device: str):
    """ Load model and tokenizer based on model name """
    print(f"Loading model: {model_name}...")
    if model_name == 'gemma-2-2b':
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", device_map=device, token = model_token
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", token = model_token)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model, tokenizer



def generate_and_save(file_path: str, tokenized_dataset: Dataset, saes: list, model: AutoModelForCausalLM, layers: list, device: str, compression_rate: int = 4):
    """
    Generates SAE features and mean-pooled activations for given layers and saves them to an HDF5 file.

    Args:
        file_path (str): Path to the output HDF5 file.
        tokenized_dataset (Iterable): Tokenized inputs to process.
        saes (dict): Dictionary of SAE models by layer.
        model: Pretrained transformer model.
        layers (list[int]): Layers to extract features from.
        device (str): Device for computation ('cpu' or 'cuda').
        compression_rate (int): Gzip compression level (default: 4).
    """
    with h5py.File(file_path, 'w') as f:
        sae_datasets = {}
        mean_act_datasets = {}

        for layer in layers:
            d = saes[layer].d_sae
            sae_datasets[layer] = f.create_dataset(
                f'sae_features_layer{layer}', shape=(0, d), maxshape=(None, d),
                dtype=np.float32, compression='gzip', compression_opts=compression_rate
            )
            mean_act_datasets[layer] = f.create_dataset(
                f'activations_layer{layer}', shape=(0, model.config.hidden_size),
                maxshape=(None, model.config.hidden_size),
                dtype=np.float32, compression='gzip', compression_opts=compression_rate
            )

        for i, input_ids in tqdm(enumerate(tokenized_dataset), total=len(tokenized_dataset)):
            sae_features, mean_pooled_activations = get_SAE_features(
                input_ids, model, saes, layers, device
            )

            for layer in layers:
                sae_datasets[layer].resize(sae_datasets[layer].shape[0] + 1, axis=0)
                sae_datasets[layer][-1] = sae_features[layer]
                mean_act_datasets[layer].resize(mean_act_datasets[layer].shape[0] + 1, axis=0)
                mean_act_datasets[layer][-1] = mean_pooled_activations[layer]

    print(f"Features saved to {file_path}")
