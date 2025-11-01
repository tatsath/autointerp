from functools import partial
from tqdm import tqdm
import torch
import json
import os
import pickle
from sae_lens import SAE, ActivationsStore
import numpy as np
from pathlib import Path

def find_max_activation(model, sae, activation_store, feature_idx, layer=None, num_batches=1):
    # Ensure model is on the correct device
    import torch
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'device'):
        device = torch.device(model.cfg.device)
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    max_activation = 0.0
    # Get hook_name first
    hook_name = getattr(sae.cfg, 'hook_name', None)
    if hook_name is None and layer is not None:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif hook_name is None:
        raise ValueError("Cannot determine hook_name from SAE config")
    
    # Get hook_layer from config or use provided layer
    hook_layer = getattr(sae.cfg, 'hook_layer', None)
    if hook_layer is None and layer is not None:
        hook_layer = layer
    elif hook_layer is None:
        # Try to infer from hook_name if it contains layer info
        if 'blocks.' in hook_name:
            try:
                hook_layer = int(hook_name.split('blocks.')[1].split('.')[0])
            except (ValueError, IndexError):
                raise ValueError("Cannot determine hook_layer from SAE config")
        else:
            raise ValueError("Cannot determine hook_layer from SAE config")
    
    import torch
    # Ensure SAE is on the same device as model
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'device'):
        model_device = torch.device(model.cfg.device)
    else:
        model_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Move SAE to model device if needed
    if hasattr(sae, 'cfg') and hasattr(sae.cfg, 'device'):
        if torch.device(sae.cfg.device) != model_device:
            sae = sae.to(model_device)
    
    for _ in tqdm(range(num_batches), desc="Finding max activation"):
        tokens = activation_store.get_batch_tokens()
        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=hook_layer + 1,
            names_filter=[hook_name],
        )
        sae_in = cache[hook_name]
        # Ensure sae_in is on the correct device
        if sae_in.device != model_device:
            sae_in = sae_in.to(model_device)
        feature_acts = sae.encode(sae_in)
        # Handle different tensor shapes
        if feature_acts.dim() > 2:
            feature_acts = feature_acts.squeeze()
        if feature_acts.dim() == 2:
            # Shape: [batch*seq, n_features] or [batch, n_features]
            batch_max = feature_acts[:, feature_idx].max().item()
        elif feature_acts.dim() == 1:
            # Shape: [n_features] - single activation
            batch_max = feature_acts[feature_idx].item()
        else:
            # Flatten to 2D: [batch*seq, n_features]
            feature_acts = feature_acts.flatten(0, -2)
            batch_max = feature_acts[:, feature_idx].max().item()
        max_activation = max(max_activation, batch_max)
    return max_activation

def steering_hook_fn(activations, hook, steering_strength, steering_vector, max_act):
    return activations + max_act * steering_strength * steering_vector

def ablate_feature_hook_fn(feature_activations, hook, feature_ids, position=None):
    if position is None:
        feature_activations[:, :, feature_ids] = 0
    else:
        feature_activations[:, position, feature_ids] = 0
    return feature_activations

def generate_with_steering(model, sae, prompt, feature_idx, max_act, strength=1.0, crop=False, layer=None):
    prepend_bos = getattr(sae.cfg, 'prepend_bos', True)
    hook_name = getattr(sae.cfg, 'hook_name', None)
    if hook_name is None and layer is not None:
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif hook_name is None:
        raise ValueError("Cannot determine hook_name from SAE config")
    
    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    steering_vector = sae.W_dec[feature_idx].to(model.cfg.device)

    if strength != 0.0:
        hook = partial(steering_hook_fn, steering_strength=strength, steering_vector=steering_vector, max_act=max_act)
    else:
        hook = partial(ablate_feature_hook_fn, feature_ids=feature_idx)

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        output = model.generate(
            input_ids,
            max_new_tokens=50,  # Reduced from 95 to 50 for faster generation
            temperature=0.7,
            top_p=0.9,
            stop_at_eos=False,
            prepend_bos=prepend_bos,
        )
    if crop:
        return model.tokenizer.decode(output[0][input_ids.shape[1]:])
    return model.tokenizer.decode(output[0])

def run_steering_experiment(model, prompts, top_features_per_layer, layers, output_folder, device, sae_path=None, dataset=None):
    """
    Run steering experiment on features.
    
    Args:
        model: Transformer model (HookedTransformer or similar)
        prompts: List of prompt strings
        top_features_per_layer: Dict of {layer: [(feature_id, score), ...]}
        layers: List of layer numbers
        output_folder: Where to save steering outputs
        device: Device string
        sae_path: Optional local SAE path (if None, uses HuggingFace)
        dataset: Optional dataset string for ActivationsStore (if None, uses SAE config or default)
    """
    results = {}
    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        results[layer] = {}
        
        # Load SAE - either local or from HuggingFace
        if sae_path:
            from .steering_utils import load_local_sae
            sae = load_local_sae(sae_path, layer, device)
        else:
            sae, _, _ = SAE.from_pretrained(
                release="gemma-scope-2b-pt-res-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical"
            )
            sae = sae.to(device)
        # Use provided dataset, or SAE config, or default
        activation_dataset = dataset
        if activation_dataset is None:
            activation_dataset = getattr(sae.cfg, 'dataset_path', None)
        if activation_dataset is None:
            # No hardcoded fallback - dataset must be provided via script
            raise ValueError("dataset argument is required. Please provide dataset via script arguments.")
        
        # If dataset is a tuple (repo, config), load it first since ActivationsStore expects string or loaded Dataset
        from datasets import load_dataset
        if isinstance(activation_dataset, tuple):
            repo, config = activation_dataset
            activation_dataset = load_dataset(repo, config, split="train", streaming=True)
        
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            dataset=activation_dataset,
            streaming=True,
            store_batch_size_prompts=1,  # Reduced for memory efficiency
            train_batch_size_tokens=512,  # Reduced from 4096 to 512
            n_batches_in_buffer=4,  # Reduced from 32 to 4
            device=device,
        )

        top_features = top_features_per_layer[layer]
        total_features = len(top_features)
        total_prompts = len(prompts)
        
        for j, (feature_id, _) in enumerate(top_features):
            results[layer][feature_id] = {}
            print(f"\n📊 Feature {feature_id} ({j+1}/{total_features})")
            
            for l, prompt in enumerate(prompts):
                remaining = (total_features - j - 1) * total_prompts + (total_prompts - l - 1)
                print(f"  Prompt {l+1}/{total_prompts} | Remaining: {remaining}", end="\r", flush=True)
                
                results[layer][feature_id][prompt] = {}

                max_act = find_max_activation(model, sae, activation_store, feature_id, layer=layer)
                prepend_bos = getattr(sae.cfg, 'prepend_bos', True)
                normal_text = model.generate(prompt, max_new_tokens=50, stop_at_eos=False, prepend_bos=prepend_bos)  # Reduced from 95 to 50
                results[layer][feature_id][prompt]['original'] = normal_text

                # Use 4 steering levels for speed: negative, weak negative, weak positive, positive
                steering_strengths = [-2.0, -1.0, 1.0, 2.0]
                for strength in steering_strengths:
                    steered_text = generate_with_steering(model, sae, prompt, feature_id, max_act, strength, crop=(l > 29), layer=layer)
                    results[layer][feature_id][prompt][strength] = steered_text

                out_path = os.path.join(output_folder, f"generated_texts_layer_{layer}_feature_{feature_id}_{j}_prompt_{l}.json")
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=4)
            
            print(f"  ✅ Feature {feature_id} completed ({j+1}/{total_features})")
    return results


def get_top_features_from_xgboost(model_name, place, width, layers, feature_type, top_n=10):
    top_features_per_layer = {}
    for layer in layers:
        path = Path(__file__).resolve().parent.parent / f"models/{model_name}-{place}-{width}_layer{layer}_{feature_type}_xgboost.pkl"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            clf = pickle.load(f)

        booster = clf.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        # Extract feature importances like "f0", "f123", etc.
        sorted_feats = sorted(
            ((int(feat[1:]), score) for feat, score in gain_dict.items()),
            key=lambda x: x[1],
            reverse=True
        )
        top_features_per_layer[layer] = sorted_feats[:top_n]
    return top_features_per_layer