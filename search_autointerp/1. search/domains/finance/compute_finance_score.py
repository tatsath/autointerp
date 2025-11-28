import os
import sys
import json
import fire
import torch

# Add parent directory to path to import core functions
# File is now in: 1. search/domains/finance/compute_finance_score.py
# Need to reference: 1. search/main/compute_score.py
script_dir = os.path.dirname(os.path.abspath(__file__))
search_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up to "1. search/"
if search_dir not in sys.path:
    sys.path.insert(0, search_dir)

# Import compute_score function from main directory
import importlib.util
compute_score_path = os.path.join(search_dir, 'main', 'compute_score.py')
spec = importlib.util.spec_from_file_location("compute_score", compute_score_path)
compute_score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_score_module)
compute_score = compute_score_module.compute_score


def compute_finance_score(
    model_path: str,
    sae_path: str,
    sae_id: str = None,
    config_path: str = "domains/finance/config.json",
    minibatch_size_features: int = 128,
    minibatch_size_tokens: int = 32,
    num_chunks: int = 1,
    chunk_num: int = 0,
    score_type: str = "domain",
    output_dir: str = None,
    run_name: str = None
):
    """Compute FinanceScore using domain-specific configuration.
    
    This wrapper uses the core compute_score function with finance-specific
    parameters from config.json and applies quantile filtering to select top features.
    """
    # Load finance-specific configuration (resolve relative to script location)
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), os.path.basename(config_path))
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Resolve relative paths in config relative to config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if not os.path.isabs(config['tokens_str_path']):
        config['tokens_str_path'] = os.path.join(config_dir, config['tokens_str_path'])
    base_output_dir = config['output_dir']
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(config_dir, base_output_dir)
    if not os.path.isabs(config.get('dashboard_output_dir', '')):
        config['dashboard_output_dir'] = os.path.join(config_dir, config.get('dashboard_output_dir', ''))
    
    # Determine output directory: use provided output_dir, or create model-based folder
    if output_dir is None:
        # Extract model name from model_path (e.g., "meta-llama/Llama-3.1-8B-Instruct" -> "llama3.1_8b")
        model_name = model_path.split('/')[-1].lower().replace('-', '_').replace('.', '_')
        # Extract layer from sae_id (e.g., "blocks.28.hook_resid_post" -> "l28")
        if sae_id:
            layer_match = sae_id.split('.')
            layer = layer_match[1] if len(layer_match) > 1 else "unknown"
            layer_suffix = f"l{layer}"
        else:
            layer_suffix = "unknown"
        
        # Create run name: use provided run_name or generate from timestamp
        if run_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = timestamp
        
        # Create output folder: {model_name}_{layer}_{run_name}
        output_dir = os.path.join(base_output_dir, f"{model_name}_{layer_suffix}_{run_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f">>> Results will be saved to: {output_dir}")
    
    # Update config to use the output directory
    config['output_dir'] = output_dir
    
    # Convert expand_range tuple format for compute_score
    expand_range = tuple(config['expand_range'])
    
    # Calculate n_samples if target_tokens is specified (for 10M tokens like ReasonScore)
    n_samples = config.get('n_samples', 10000)
    # Temporarily disable target_tokens calculation for quick testing
    # if 'target_tokens' in config and config['target_tokens']:
    #     context_size = config.get('context_size', 1024)
    #     calculated_samples = max(n_samples, config['target_tokens'] // context_size)
    #     print(f">>> Target tokens: {config['target_tokens']:,}")
    #     print(f">>> Context size: {context_size}")
    #     print(f">>> Calculated n_samples for target: {calculated_samples:,}")
    #     print(f">>> Expected total tokens: ~{calculated_samples * context_size:,}")
    #     n_samples = calculated_samples
    print(f">>> Using n_samples from config: {n_samples:,} (for quick testing)")
    
    # Compute scores using core function
    print(">>> Computing FinanceScore using core compute_score function...")
    compute_score(
        model_path=model_path,
        sae_path=sae_path,
        dataset_path=config['dataset_path'],
        tokens_str_path=config['tokens_str_path'],
        output_dir=config['output_dir'],
        sae_id=sae_id,
        expand_range=expand_range,
        ignore_tokens=config.get('ignore_tokens'),
        n_samples=n_samples,
        alpha=config['alpha'],
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        num_chunks=num_chunks,
        chunk_num=chunk_num,
        score_type=score_type
    )
    
    # Apply quantile filtering to get top features
    scores_path = os.path.join(config['output_dir'], "feature_scores.pt")
    if num_chunks > 1:
        scores_path = os.path.join(config['output_dir'], f"feature_scores_{chunk_num}.pt")
    
    if os.path.exists(scores_path):
        feature_scores = torch.load(scores_path, weights_only=True, map_location="cpu")
        quantile_threshold = config.get('quantile_threshold', 0.95)
        # Get selection method from config: "quantile" (like ReasonScore) or "topk"
        selection_method = config.get('selection_method', 'quantile')
        # Get the number of top features to select from config (used for topk method)
        num_top_features = config.get('num_top_features', config.get('expected_top_features', 200))
        if 'expected_top_features' in config and 'num_top_features' not in config:
            # Use expected_top_features if num_top_features not specified
            num_top_features = config['expected_top_features']
        
        if selection_method == 'quantile':
            # ReasonScore-style: Use quantile threshold (percentile-based selection)
            quantile_value = torch.quantile(feature_scores, quantile_threshold)
            top_features = (feature_scores >= quantile_value).nonzero(as_tuple=True)[0]
            quantile_value_used = float(quantile_value)
            print(f">>> Using quantile threshold ({quantile_threshold}): {quantile_value_used:.6f}")
            print(f">>> Top features selected: {len(top_features)} (percentile-based, like ReasonScore)")
        else:
            # Top-k selection: Filter out zero scores first, then take top N
            non_zero_mask = feature_scores > 0
            non_zero_scores = feature_scores[non_zero_mask]
            non_zero_indices = torch.where(non_zero_mask)[0]
            
            if len(non_zero_scores) > 0:
                # Take top N features by score (from non-zero scores) - number comes from config
                top_k = min(num_top_features, len(non_zero_scores))
                top_k_values, top_k_indices = torch.topk(non_zero_scores, k=top_k, dim=0)
                top_features = non_zero_indices[top_k_indices]
                quantile_value_used = top_k_values[-1].item()  # Minimum score of selected features
                print(f">>> Using top-{top_k} selection from config (num_top_features={num_top_features})")
            else:
                # All scores are zero - take top N anyway (from config)
                top_k = min(num_top_features, len(feature_scores))
                top_k_values, top_k_indices = torch.topk(feature_scores, k=top_k, dim=0)
                top_features = top_k_indices
                quantile_value_used = top_k_values[-1].item()
                print(f">>> All scores are zero, taking top-{top_k} features from config anyway")
            
            print(f">>> Top features selected: {len(top_features)} (requested from config: {num_top_features})")
        
        # Save top feature indices
        top_features_path = os.path.join(config['output_dir'], "top_features.pt")
        torch.save(top_features, top_features_path)
        
        # Save scores with feature indices for reference
        top_scores_dict = {
            'feature_indices': top_features.tolist(),
            'scores': feature_scores[top_features].tolist(),
            'quantile_threshold': quantile_threshold,
            'quantile_value': quantile_value_used,
            'selection_method': selection_method,
            'num_top_features_requested': num_top_features if selection_method == 'topk' else None
        }
        top_scores_path = os.path.join(config['output_dir'], "top_features_scores.json")
        with open(top_scores_path, 'w') as f:
            json.dump(top_scores_dict, f, indent=2)
        
        print(f">>> Top features saved to: {top_features_path}")
        print(f">>> Top features scores saved to: {top_scores_path}")


if __name__ == "__main__":
    fire.Fire(compute_finance_score)

