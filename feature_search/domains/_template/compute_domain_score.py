import os
import sys
import json
import fire
import torch

# Add parent directory to path to import core functions
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import compute_score function from main directory
import importlib.util
compute_score_path = os.path.join(parent_dir, 'main', 'compute_score.py')
spec = importlib.util.spec_from_file_location("compute_score", compute_score_path)
compute_score_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute_score_module)
compute_score = compute_score_module.compute_score


def compute_domain_score(
    model_path: str,
    sae_path: str,
    sae_id: str = None,
    config_path: str = "domains/your_domain/config.json",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    num_chunks: int = 1,
    chunk_num: int = 0,
    score_type: str = "domain"
):
    """Compute DomainScore using domain-specific configuration.
    
    This wrapper uses the core compute_score function with domain-specific
    parameters from config.json and applies quantile filtering to select top features.
    """
    # Load domain-specific configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Convert expand_range tuple format for compute_score
    expand_range = tuple(config['expand_range'])
    
    # Compute scores using core function
    print(f">>> Computing DomainScore using core compute_score function...")
    compute_score(
        model_path=model_path,
        sae_path=sae_path,
        dataset_path=config['dataset_path'],
        tokens_str_path=config['tokens_str_path'],
        output_dir=config['output_dir'],
        sae_id=sae_id,
        expand_range=expand_range,
        ignore_tokens=config.get('ignore_tokens'),
        n_samples=config['n_samples'],
        alpha=config.get('alpha', 1.0),
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
        
        print(f">>> DomainScore quantile threshold ({quantile_threshold}): {quantile_value:.6f}")
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
    fire.Fire(compute_domain_score)

