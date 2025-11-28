import os
import sys
import json
import fire
import torch

# Add parent directory to path to import core functions
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import compute_score function from domain_common directory
import importlib.util
compute_score_path = os.path.join(parent_dir, 'domain_common', 'compute_score.py')
spec = importlib.util.spec_from_file_location("compute_score", compute_score_path)
compute_score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_score_module)
compute_score = compute_score_module.compute_score


def compute_finance_score(
    model_path: str,
    sae_path: str,
    sae_id: str = None,
    config_path: str = None,
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    num_chunks: int = 1,
    chunk_num: int = 0,
    score_type: str = "domain"
):
    """Compute FinanceScore using domain-specific configuration.
    
    This wrapper uses the core compute_score function with finance-specific
    parameters from config.json and applies quantile filtering to select top features.
    """
    # Load finance-specific configuration (resolve relative to script location)
    if config_path is None:
        # Try config.yaml first, then config.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(script_dir, "config.yaml")
        json_path = os.path.join(script_dir, "config.json")
        if os.path.exists(yaml_path):
            config_path = yaml_path
        elif os.path.exists(json_path):
            config_path = json_path
        else:
            config_path = json_path  # Default to JSON
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), os.path.basename(config_path))
    
    # Load config (support both JSON and YAML)
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    
    # Resolve relative paths in config relative to config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Handle tokens_str_path - support both .json and .txt
    tokens_path = config.get('tokens_str_path', 'finance_tokens.json')
    if not os.path.isabs(tokens_path):
        tokens_path = os.path.join(config_dir, tokens_path)
    
    # If tokens.txt, convert to JSON; if finance_tokens.json, use it directly
    if tokens_path.endswith('.txt'):
        with open(tokens_path, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        tokens_json_path = tokens_path.replace('.txt', '.json')
        with open(tokens_json_path, 'w') as f:
            json.dump(tokens, f, indent=2)
        config['tokens_str_path'] = tokens_json_path
    elif not os.path.isabs(config['tokens_str_path']):
        config['tokens_str_path'] = os.path.join(config_dir, config['tokens_str_path'])
    
    if not os.path.isabs(config['output_dir']):
        config['output_dir'] = os.path.join(config_dir, config['output_dir'])
        os.makedirs(config['output_dir'], exist_ok=True)
    if not os.path.isabs(config.get('dashboard_output_dir', '')):
        config['dashboard_output_dir'] = os.path.join(config_dir, config.get('dashboard_output_dir', ''))
        os.makedirs(config['dashboard_output_dir'], exist_ok=True)
    
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
        quantile_threshold = config['quantile_threshold']
        
        # Get quantile value
        quantile_value = torch.quantile(feature_scores, quantile_threshold)
        top_features = (feature_scores >= quantile_value).nonzero(as_tuple=True)[0]
        
        print(f">>> FinanceScore quantile threshold ({quantile_threshold}): {quantile_value:.6f}")
        print(f">>> Top features selected: {len(top_features)} (expected: ~{config['expected_top_features']})")
        
        # Save top feature indices
        top_features_path = os.path.join(config['output_dir'], "top_features.pt")
        torch.save(top_features, top_features_path)
        
        # Save scores with feature indices for reference
        top_scores_dict = {
            'feature_indices': top_features.tolist(),
            'scores': feature_scores[top_features].tolist(),
            'quantile_threshold': quantile_threshold,
            'quantile_value': float(quantile_value)
        }
        top_scores_path = os.path.join(config['output_dir'], "top_features_scores.json")
        with open(top_scores_path, 'w') as f:
            json.dump(top_scores_dict, f, indent=2)
        
        print(f">>> Top features saved to: {top_features_path}")
        print(f">>> Top features scores saved to: {top_scores_path}")


if __name__ == "__main__":
    fire.Fire(compute_finance_score)
