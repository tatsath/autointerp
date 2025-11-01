"""
Utility functions for loading local SAEs and adapting steering to different model/SAE configurations.
"""
from pathlib import Path
from sae_lens import SAE
import json
import pandas as pd


def load_local_sae(sae_path: str, layer: int, device: str = "cuda:0"):
    """
    Load SAE from local safetensors file.
    Simple wrapper that fixes the config file and uses SAELens' built-in loader.
    
    Args:
        sae_path: Path to SAE directory
        layer: Layer number
        device: Device to load on
    
    Returns:
        SAE object
    """
    sae_path = Path(sae_path)
    layer_dir = sae_path / f"layers.{layer}"
    
    if not layer_dir.exists():
        raise FileNotFoundError(f"Layer {layer} directory not found: {layer_dir}")
    
    cfg_file = layer_dir / "cfg.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")
    
    # Read and fix config if needed (add missing required fields)
    with open(cfg_file, 'r') as f:
        cfg_dict = json.load(f)
    
    # Infer d_sae from safetensors if missing
    safetensors_file = layer_dir / "sae.safetensors"
    if "d_sae" not in cfg_dict and safetensors_file.exists():
        from safetensors import safe_open
        with safe_open(str(safetensors_file), framework="pt") as f:
            if "W_dec" in f.keys():
                d_sae = f.get_tensor("W_dec").shape[0]
                cfg_dict["d_sae"] = d_sae
            elif "encoder.weight" in f.keys():
                # Infer from encoder weight shape
                enc_weight = f.get_tensor("encoder.weight")
                d_sae = enc_weight.shape[0] if enc_weight.shape[0] > enc_weight.shape[1] else enc_weight.shape[1]
                cfg_dict["d_sae"] = d_sae
    
    config_modified = False
    if "dtype" not in cfg_dict:
        cfg_dict["dtype"] = "float32"
        config_modified = True
    if "hook_name" not in cfg_dict:
        cfg_dict["hook_name"] = f"blocks.{layer}.hook_resid_post"
        config_modified = True
    if "hook_layer" not in cfg_dict:
        cfg_dict["hook_layer"] = layer
        config_modified = True
    if "prepend_bos" not in cfg_dict:
        cfg_dict["prepend_bos"] = True
        config_modified = True
    if "context_size" not in cfg_dict:
        cfg_dict["context_size"] = 1024
        config_modified = True
    
    # Save fixed config if modified (non-destructive, we'll restore it)
    original_config_backup = None
    if config_modified:
        with open(cfg_file, 'r') as f:
            original_config_backup = json.load(f)
        with open(cfg_file, 'w') as f:
            json.dump(cfg_dict, f, indent=2)
    
    # Custom converter function to handle key mapping
    def custom_sae_loader(path: str, device: str, cfg_overrides=None):
        from safetensors import safe_open
        from sae_lens.loading.pretrained_sae_loaders import read_sae_components_from_disk
        
        # Load config
        cfg_file = Path(path) / "cfg.json"
        with open(cfg_file, 'r') as f:
            cfg_dict = json.load(f)
        
        # Apply overrides
        if cfg_overrides:
            cfg_dict.update(cfg_overrides)
        
        # Load and convert weights
        safetensors_file = Path(path) / "sae.safetensors"
        if not safetensors_file.exists():
            safetensors_file = Path(path) / "sae_weights.safetensors"
        
        state_dict = {}
        with safe_open(str(safetensors_file), framework="pt", device=device) as f:
            keys = list(f.keys())
            
            # Map keys to SAELens format
            if "encoder.weight" in keys:
                enc_weight = f.get_tensor("encoder.weight")
                # encoder.weight is [d_sae, d_model], transpose to [d_model, d_sae] for W_enc
                # SAELens expects W_enc: [d_model, d_sae]
                state_dict["W_enc"] = enc_weight.T
            elif "W_enc" in keys:
                state_dict["W_enc"] = f.get_tensor("W_enc")
            
            if "encoder.bias" in keys:
                state_dict["b_enc"] = f.get_tensor("encoder.bias")
            elif "b_enc" in keys:
                state_dict["b_enc"] = f.get_tensor("b_enc")
            else:
                # Create zero bias
                import torch
                state_dict["b_enc"] = torch.zeros(state_dict["W_enc"].shape[1], device=device)
            
            if "W_dec" in keys:
                state_dict["W_dec"] = f.get_tensor("W_dec")
            
            if "b_dec" in keys:
                state_dict["b_dec"] = f.get_tensor("b_dec")
            else:
                # Create zero bias
                import torch
                state_dict["b_dec"] = torch.zeros(state_dict["W_enc"].shape[0], device=device)
        
        return cfg_dict, state_dict
    
    try:
        # Use custom loader
        sae = SAE.load_from_disk(str(layer_dir), device=device, converter=custom_sae_loader)
        return sae
    finally:
        # Restore original config if we modified it
        if config_modified and original_config_backup:
            with open(cfg_file, 'w') as f:
                json.dump(original_config_backup, f, indent=2)


def get_features_from_csv(csv_path: str, feature_column: str = "large_feature", top_n: int = None):
    """
    Extract feature list from CSV file (e.g., similarity map).
    
    Args:
        csv_path: Path to CSV file
        feature_column: Column name containing feature IDs
        top_n: Optional limit on number of features
    
    Returns:
        List of feature IDs
    """
    df = pd.read_csv(csv_path)
    
    if feature_column not in df.columns:
        raise ValueError(f"Column '{feature_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    features = sorted(df[feature_column].unique().tolist())
    
    if top_n:
        features = features[:top_n]
    
    return features


def get_features_per_layer_from_csv(csv_path: str, feature_column: str = "large_feature", layer: int = 19, top_n: int = 10):
    """
    Get features for a specific layer, formatted for steering experiment.
    
    Returns:
        Dictionary: {layer: [(feature_id, score), ...]}
    """
    features = get_features_from_csv(csv_path, feature_column, top_n)
    
    # Format as (feature_id, score) tuples with dummy scores
    # The steering code expects this format from XGBoost
    feature_list = [(int(f), 1.0) for f in features]
    
    return {layer: feature_list}


def load_prompts_from_dataset(dataset_repo: str, dataset_name: str = "default", dataset_split: str = "train[:10]", num_prompts: int = 30):
    """
    Load prompts from HuggingFace dataset for steering experiments.
    For OPENTHOUGHTS-114K with metadata config, performs stratified sampling across domains
    (matching methodology: "stratified across different domains to ensure a balanced representation").
    
    Args:
        dataset_repo: HuggingFace dataset repository
        dataset_name: Dataset name/config
        dataset_split: Dataset split (e.g., "train[:10]")
        num_prompts: Number of prompts to extract
    
    Returns:
        List of prompt strings
    """
    from datasets import load_dataset
    import random
    
    try:
        dataset = load_dataset(dataset_repo, dataset_name, split=dataset_split, streaming=False)
        
        # Extract text column
        text_column = "text" if "text" in dataset.column_names else dataset.column_names[0]
        
        # Check if dataset has domain information for stratification (OPENTHOUGHTS-114K)
        if "domain" in dataset.column_names and num_prompts > 10:
            # Stratified sampling: evenly sample across domains
            domains = set(dataset["domain"]) if "domain" in dataset.column_names else None
            if domains and len(domains) > 1:
                prompts_per_domain = max(1, num_prompts // len(domains))
                prompts = []
                selected_indices = set()
                
                # Shuffle dataset first for randomness
                dataset = dataset.shuffle(seed=42)
                
                # Sample from each domain
                for domain in domains:
                    domain_indices = [i for i, d in enumerate(dataset["domain"]) if d == domain]
                    available_indices = [i for i in domain_indices if i not in selected_indices]
                    
                    if available_indices:
                        sample_size = min(prompts_per_domain, len(available_indices))
                        sampled_indices = random.sample(available_indices, sample_size)
                        selected_indices.update(sampled_indices)
                        
                        for idx in sampled_indices:
                            prompts.append(dataset[idx][text_column])
                
                # If we need more, fill randomly from remaining
                if len(prompts) < num_prompts:
                    all_indices = set(range(len(dataset)))
                    remaining_indices = list(all_indices - selected_indices)
                    additional_needed = num_prompts - len(prompts)
                    
                    if remaining_indices:
                        additional_size = min(additional_needed, len(remaining_indices))
                        additional_indices = random.sample(remaining_indices, additional_size)
                        for idx in additional_indices:
                            prompts.append(dataset[idx][text_column])
                
                # Shuffle final list
                random.shuffle(prompts)
                prompts = prompts[:num_prompts]
            else:
                # Fallback to simple random sampling
                dataset = dataset.shuffle(seed=42)
                sampled_indices = random.sample(range(len(dataset)), min(num_prompts, len(dataset)))
                prompts = [dataset[idx][text_column] for idx in sampled_indices]
        else:
            # Simple sampling: shuffle and take first N
            dataset = dataset.shuffle(seed=42)
            sampled_indices = random.sample(range(len(dataset)), min(num_prompts, len(dataset)))
            prompts = [dataset[idx][text_column] for idx in sampled_indices]
        
        return prompts
    except Exception as e:
        raise RuntimeError(f"Failed to load prompts from dataset {dataset_repo}: {e}")

