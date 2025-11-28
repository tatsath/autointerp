import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import json
from huggingface_hub import hf_hub_download
from typing import List
from pathlib import Path


class JumpReLUSAE(nn.Module):
    
    def __init__(self, d_model, d_sae):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.d_sae = d_sae
        
    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    
    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec
    
    def forward(self, acts):
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


def load_params(path: str, device: str):
    """ Load parameters from a given path """
    print(f"Loading parameters from {path}...")
    params = np.load(path)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
    return params, pt_params


def load_model_config(config_file: str, place: str, layer: int, width: str):
    """ Load the model configuration from a JSON file """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Fetch repo_id and filename for the given layer and width
        return config[place + '-' + str(layer) + '-' + width]
    except KeyError:
        raise ValueError(f"Configuration for layer {layer} and width {width} not found in {config_file}.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_file} not found.")
    except json.JSONDecodeError:
        raise ValueError(f"Error decoding the JSON config file {config_file}.")
        
        
def load_saes(layers: List[int], place: str, width: str, device: str):

    saes = {}
    for layer_idx in layers:
        # Load configuration for the specific layer
        config_path = Path(__file__).resolve().parent / 'sae_config.json'
        config = load_model_config(config_path, place, layer_idx, width)
        repo_id, filename = config['repo_id'], config['filename']

        # Download model parameters
        path_to_params = hf_hub_download(repo_id=repo_id, filename=filename, force_download=False)
        params, pt_params = load_params(path_to_params, device)

        # Initialize and load the SAE model
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        saes[layer_idx] = sae

    return saes




def gather_residual_activations(model, target_layers, input_ids):
    """
    Gather activations from specific layers of the model.
    
    Args:
        model: The model from which activations are to be gathered.
        target_layers: A list of layer numbers (indices) from which to gather activations.
        input_ids: The input tensor to the model.
    
    Returns:
        activations: A dictionary where keys are layer indices and values are the activations from those layers.
    """
    activations = {}

    # Define the hook to gather the activations
    def gather_target_act_hook(layer_idx):
        def hook(mod, inputs, outputs):
            activations[layer_idx] = outputs[0].detach()  # Store the activation
            return outputs
        return hook

    # Register forward hooks to the target layers
    handles = []
    for layer_idx in target_layers:
        hook = gather_target_act_hook(layer_idx)
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)

    # Run the model's forward pass
    _ = model(input_ids)  # Assuming inputs is a dictionary with 'input_ids' and 'attention_mask'

    # Remove the hooks after activations are gathered
    for handle in handles:
        handle.remove()

    return activations


def get_SAE_features(input_ids, model, saes, target_layers, device):

    # Gather activations from the specified target layers
    activations = gather_residual_activations(model, target_layers, input_ids.to(device))
    
    # Clear CUDA cache to avoid memory issues
    torch.cuda.empty_cache()

    # Dictionaries to store SAE features and mean-pooled activations for each layer
    sae_features = {}
    mean_pooled_activations = {}

    # Process activations for each target layer
    for layer_idx, target_act in activations.items():
        # Save the mean-pooled activations
        mean_pooled_activations[layer_idx] = target_act.mean(dim=1).detach().cpu().numpy()

        # Encode activations using the SAE model
        sae_acts = saes[layer_idx].encode(target_act.to(torch.float32))

        act = sae_acts[0, 1:].sum(0).float().detach().cpu().numpy()

        sae_features[layer_idx] = act

    return sae_features, mean_pooled_activations


