from functools import partial
from tqdm import tqdm
import torch
import json
import os
import pickle
from sae_lens import SAE, ActivationsStore
import numpy as np
from pathlib import Path

def find_max_activation(model, sae, activation_store, feature_idx, hook_name, hook_layer, num_batches=1):
    """
    Find the maximum activation for a given feature index.
    
    Args:
        model: Transformer model
        sae: SAE model (already on correct device)
        activation_store: ActivationsStore instance
        feature_idx: Feature index to find max activation for
        hook_name: Pre-computed hook name (e.g., "blocks.19.hook_resid_post")
        hook_layer: Pre-computed hook layer number
        num_batches: Number of batches to process (default: 1 for speed)
    
    Returns:
        Maximum activation value
    """
    max_activation = 0.0
    device = sae.W_dec.device
    
    for _ in range(num_batches):
        tokens = activation_store.get_batch_tokens()
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                stop_at_layer=hook_layer + 1,
                names_filter=[hook_name],
            )
            sae_in = cache[hook_name].to(device)
            feature_acts = sae.encode(sae_in).flatten(0, 1)
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

def generate_with_steering(model, sae, prompt, feature_idx, max_act, hook_name, prepend_bos, strength=1.0, crop=False, max_new_tokens=32):
    """
    Generate text with feature steering.
    
    Args:
        model: Transformer model
        sae: SAE model (already on correct device)
        prompt: Input prompt string
        feature_idx: Feature index to steer
        max_act: Maximum activation value
        hook_name: Pre-computed hook name
        prepend_bos: Whether to prepend BOS token
        strength: Steering strength
        crop: Whether to crop output to only new tokens
        max_new_tokens: Maximum tokens to generate (default: 32 for speed)
    """
    input_ids = model.to_tokens(prompt, prepend_bos=prepend_bos)
    steering_vector = sae.W_dec[feature_idx].to(model.cfg.device)

    if strength != 0.0:
        hook = partial(steering_hook_fn, steering_strength=strength, steering_vector=steering_vector, max_act=max_act)
    else:
        hook = partial(ablate_feature_hook_fn, feature_ids=feature_idx)

    with model.hooks(fwd_hooks=[(hook_name, hook)]):
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                stop_at_eos=False,
                prepend_bos=prepend_bos,
            )
    if crop:
        return model.tokenizer.decode(output[0][input_ids.shape[1]:])
    return model.tokenizer.decode(output[0])

def run_steering_experiment(model, prompts, top_features_per_layer, layers, output_folder, device, sae_path=None, dataset=None, num_batches=1, max_new_tokens=32):
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
        num_batches: Number of batches for max activation search (default: 1 for speed)
        max_new_tokens: Maximum tokens to generate (default: 32 for speed)
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
        
        # Pre-compute static values (moved out of inner loops)
        # Use hook name from SAE config if available, otherwise default
        hook_name = getattr(sae.cfg, 'hook_name', f"blocks.{layer}.hook_resid_post")
        hook_layer = getattr(sae.cfg, 'hook_layer', layer)
        prepend_bos = getattr(sae.cfg, 'prepend_bos', True)
        
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
            # For financial-news and similar datasets, we need to handle column mapping
            # Load a small sample first to check column names
            sample = load_dataset(repo, config if config else None, split="train[:1]", streaming=False)
            if hasattr(sample, 'column_names'):
                # Check if we need to map columns
                if 'headline' in sample.column_names and 'text' not in sample.column_names:
                    # Use map to add 'text' column from 'headline' for streaming dataset
                    def add_text_column(example):
                        if 'headline' in example:
                            example['text'] = example['headline']
                        elif 'article' in example:
                            example['text'] = example['article']
                        return example
                    activation_dataset = load_dataset(repo, config if config else None, split="train", streaming=True)
                    activation_dataset = activation_dataset.map(add_text_column)
                else:
                    activation_dataset = load_dataset(repo, config if config else None, split="train", streaming=True)
            else:
                activation_dataset = load_dataset(repo, config if config else None, split="train", streaming=True)
        elif isinstance(activation_dataset, str):
            # String path - load and check columns
            sample = load_dataset(activation_dataset, split="train[:1]", streaming=False)
            if hasattr(sample, 'column_names'):
                if 'headline' in sample.column_names and 'text' not in sample.column_names:
                    def add_text_column(example):
                        if 'headline' in example:
                            example['text'] = example['headline']
                        elif 'article' in example:
                            example['text'] = example['article']
                        return example
                    activation_dataset = load_dataset(activation_dataset, split="train", streaming=True)
                    activation_dataset = activation_dataset.map(add_text_column)
                else:
                    activation_dataset = load_dataset(activation_dataset, split="train", streaming=True)
            else:
                activation_dataset = load_dataset(activation_dataset, split="train", streaming=True)
        
        # Handle datasets with non-standard column names (e.g., financial-news has 'headline' instead of 'text')
        # For non-streaming datasets loaded directly
        if hasattr(activation_dataset, 'column_names') and activation_dataset.column_names is not None:
            # For non-streaming datasets, we can check and rename columns
            if 'headline' in activation_dataset.column_names and 'text' not in activation_dataset.column_names:
                # Rename 'headline' to 'text' for ActivationsStore compatibility
                activation_dataset = activation_dataset.rename_column('headline', 'text')
            elif 'article' in activation_dataset.column_names and 'text' not in activation_dataset.column_names:
                # Rename 'article' to 'text' for ActivationsStore compatibility
                activation_dataset = activation_dataset.rename_column('article', 'text')
        elif hasattr(activation_dataset, 'features') and activation_dataset.features is not None:
            # For streaming datasets, check features and add text column if needed
            if 'headline' in activation_dataset.features and 'text' not in activation_dataset.features:
                def add_text_column(example):
                    if 'headline' in example:
                        example['text'] = example['headline']
                    elif 'article' in example:
                        example['text'] = example['article']
                    return example
                activation_dataset = activation_dataset.map(add_text_column)
            elif 'article' in activation_dataset.features and 'text' not in activation_dataset.features:
                def add_text_column(example):
                    if 'article' in example:
                        example['text'] = example['article']
                    return example
                activation_dataset = activation_dataset.map(add_text_column)
        
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
            
            # Find max activation once per feature (not per prompt)
            max_act = find_max_activation(model, sae, activation_store, feature_id, hook_name, hook_layer, num_batches=num_batches)
            
            for l, prompt in enumerate(prompts):
                remaining = (total_features - j - 1) * total_prompts + (total_prompts - l - 1)
                print(f"  Prompt {l+1}/{total_prompts} | Remaining: {remaining}", end="\r", flush=True)
                
                # Check if this result already exists
                out_path = os.path.join(output_folder, f"generated_texts_layer_{layer}_feature_{feature_id}_{j}_prompt_{l}.json")
                if os.path.exists(out_path):
                    # Load existing results
                    try:
                        with open(out_path, "r") as f:
                            existing_results = json.load(f)
                        if str(layer) in existing_results and str(feature_id) in existing_results[str(layer)] and prompt in existing_results[str(layer)][str(feature_id)]:
                            print(f"  Prompt {l+1}/{total_prompts} | Skipping (already exists)", end="\r", flush=True)
                            continue
                    except:
                        pass
                
                results[layer][feature_id][prompt] = {}

                # Generate original text with error handling
                try:
                    with torch.no_grad():
                        normal_text = model.generate(prompt, max_new_tokens=max_new_tokens, stop_at_eos=False, prepend_bos=prepend_bos)
                    results[layer][feature_id][prompt]['original'] = normal_text
                except Exception as e:
                    error_msg = str(e).lower()
                    if "tensor size" in error_msg or "rotary" in error_msg or "sizes of tensors" in error_msg:
                        print(f"\n  ⚠️  Warning: Generation failed for feature {feature_id}, prompt {l+1} due to rotary embedding issue. Skipping this prompt...")
                        # Skip this prompt and continue to next one
                        continue
                    else:
                        print(f"\n  ❌ Unexpected error for feature {feature_id}, prompt {l+1}: {e}")
                        raise

                # Use 4 steering levels: negative, weak negative, weak positive, positive
                steering_strengths = [-4.0, -2.0, 2.0, 4.0]
                for strength in steering_strengths:
                    try:
                        steered_text = generate_with_steering(
                            model, sae, prompt, feature_id, max_act, 
                            hook_name, prepend_bos, strength, 
                            crop=(l > 29), max_new_tokens=max_new_tokens
                        )
                        results[layer][feature_id][prompt][strength] = steered_text
                    except RuntimeError as e:
                        if "tensor size" in str(e).lower() or "rotary" in str(e).lower():
                            print(f"\n  ⚠️  Warning: Steering failed for feature {feature_id}, prompt {l+1}, strength {strength}. Using placeholder.")
                            results[layer][feature_id][prompt][strength] = "[Generation failed due to rotary embedding issue]"
                        else:
                            raise

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