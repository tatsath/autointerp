#!/usr/bin/env python3
"""
Unified command-line interface for feature search.
Takes all necessary inputs and generates a list of relevant features.
"""

import os
import sys
import json
import fire
import torch
from typing import List, Tuple, Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from compute_score import compute_score
from compute_dashboard import compute_dashboard


def run_feature_search(
    # Required parameters
    model_path: str,
    sae_path: str,
    dataset_path: str,
    output_dir: str,
    
    # Optional token configuration
    tokens_str_path: Optional[str] = None,
    
    # SAE configuration
    sae_id: Optional[str] = None,
    
    # Token matching configuration
    expand_range: Optional[Tuple[int, int]] = None,
    ignore_tokens: Optional[List[int]] = None,
    
    # Scoring configuration
    score_type: str = "domain",  # "domain", "simple", or "fisher"
    alpha: float = 1.0,  # Only used with score_type="domain"
    
    # Data configuration
    n_samples: int = 4096,
    column_name: str = "text",
    
    # Feature selection
    num_features: int = 100,  # Number of top features to return
    selection_method: str = "topk",  # "topk" or "quantile"
    quantile_threshold: float = 0.95,  # Only used with selection_method="quantile"
    
    # Processing configuration
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    num_chunks: int = 1,
    chunk_num: int = 0,
    
    # Dashboard generation (optional)
    generate_dashboard: bool = False,
    dashboard_n_samples: int = 5000,
    separate_files: bool = False,
):
    """
    Run complete feature search pipeline.
    
    Args:
        model_path: Path to the model (HuggingFace repo or local)
        sae_path: Path to SAE (HuggingFace repo or local directory)
        dataset_path: Path to dataset (HuggingFace repo or local)
        output_dir: Directory to save results
        tokens_str_path: (Optional) Path to JSON file with domain-specific token strings.
                         If not provided, search will be 100% dataset-driven.
        sae_id: SAE identifier (e.g., "blocks.19.hook_resid_post")
        expand_range: Tuple (left, right) to expand context around matched tokens
        ignore_tokens: List of token IDs to ignore
        score_type: Scoring method - "domain", "simple", or "fisher"
        alpha: Weight for entropy term (only for score_type="domain")
        n_samples: Number of samples to process
        column_name: Dataset column name containing text
        num_features: Number of top features to return
        selection_method: "topk" (select top N) or "quantile" (select by quantile)
        quantile_threshold: Quantile threshold (0-1) for selection_method="quantile"
        minibatch_size_features: Batch size for feature processing
        minibatch_size_tokens: Batch size for token processing
        num_chunks: Number of chunks to split features into
        chunk_num: Current chunk number (0-indexed)
        generate_dashboard: Whether to generate HTML dashboard
        dashboard_n_samples: Number of samples for dashboard generation
        separate_files: Whether to save dashboard as separate files per feature
    
    Returns:
        Dictionary with feature indices, scores, and paths to output files
    """
    print("=" * 80)
    print("Feature Search Pipeline")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"SAE: {sae_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Score Type: {score_type}")
    print(f"Number of Features: {num_features}")
    print(f"Selection Method: {selection_method}")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Compute feature scores
    print("Step 1: Computing feature scores...")
    print("-" * 80)
    
    compute_score(
        model_path=model_path,
        sae_path=sae_path,
        dataset_path=dataset_path,
        tokens_str_path=tokens_str_path,
        output_dir=output_dir,
        sae_id=sae_id,
        expand_range=expand_range,
        ignore_tokens=ignore_tokens,
        n_samples=n_samples,
        alpha=alpha,
        column_name=column_name,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        num_chunks=num_chunks,
        chunk_num=chunk_num,
        score_type=score_type
    )
    
    print()
    print("✅ Feature scores computed!")
    print()
    
    # Step 2: Select top features
    print("Step 2: Selecting top features...")
    print("-" * 80)
    
    # Load feature scores
    scores_path = os.path.join(output_dir, "feature_scores.pt")
    if num_chunks > 1:
        # Merge chunks if needed
        if not os.path.exists(scores_path):
            filenames = [n for n in os.listdir(output_dir) if "feature_scores" in n and n.endswith(".pt")]
            filenames.sort()
            if filenames:
                print(f">>> Merging {len(filenames)} chunks...")
                scores = torch.concat([torch.load(os.path.join(output_dir, n), weights_only=True) for n in filenames])
                torch.save(scores, scores_path)
            else:
                scores_path = os.path.join(output_dir, f"feature_scores_{chunk_num}.pt")
    
    feature_scores = torch.load(scores_path, weights_only=True, map_location="cpu")
    
    # Select features based on method
    quantile_value = None
    if selection_method == "quantile":
        quantile_value = torch.quantile(feature_scores, quantile_threshold)
        top_indices = (feature_scores >= quantile_value).nonzero(as_tuple=True)[0]
        print(f">>> Quantile threshold ({quantile_threshold}): {quantile_value:.6f}")
        print(f">>> Selected {len(top_indices)} features above quantile")
    else:  # topk
        top_indices = feature_scores.topk(k=min(num_features, len(feature_scores))).indices
        print(f">>> Selected top {len(top_indices)} features")
    
    top_indices = top_indices.tolist()
    top_scores = feature_scores[top_indices].tolist()
    
    # Save top features
    top_features_path = os.path.join(output_dir, "top_features.pt")
    torch.save(torch.tensor(top_indices), top_features_path)
    
    # Save feature list with scores
    feature_list = {
        "feature_indices": top_indices,
        "scores": top_scores,
        "num_features": len(top_indices),
        "selection_method": selection_method,
        "score_type": score_type,
    }
    if selection_method == "quantile":
        feature_list["quantile_threshold"] = quantile_threshold
        feature_list["quantile_value"] = float(quantile_value)
    
    feature_list_path = os.path.join(output_dir, "feature_list.json")
    with open(feature_list_path, 'w') as f:
        json.dump(feature_list, f, indent=2)
    
    print(f">>> Top features saved to: {top_features_path}")
    print(f">>> Feature list saved to: {feature_list_path}")
    print()
    
    # Step 3: Generate dashboard (optional)
    if generate_dashboard:
        print("Step 3: Generating dashboard...")
        print("-" * 80)
        
        try:
            compute_dashboard(
                model_path=model_path,
                sae_path=sae_path,
                dataset_path=dataset_path,
                scores_dir=output_dir,
                output_dir=os.path.join(output_dir, "dashboards"),
                sae_id=sae_id,
                num_features=len(top_indices),
                column_name=column_name,
                minibatch_size_features=minibatch_size_features,
                minibatch_size_tokens=minibatch_size_tokens,
                n_samples=dashboard_n_samples,
                separate_files=separate_files
            )
            print("✅ Dashboard generated!")
        except Exception as e:
            print(f"⚠️  Dashboard generation failed: {e}")
            print("   (Feature scores and list are still available)")
        print()
    
    # Print summary
    print("=" * 80)
    print("Feature Search Complete!")
    print("=" * 80)
    print(f"Top {len(top_indices)} features selected:")
    print(f"  Indices: {top_indices[:10]}{'...' if len(top_indices) > 10 else ''}")
    print(f"  Score range: [{min(top_scores):.6f}, {max(top_scores):.6f}]")
    print()
    print("Output files:")
    print(f"  • Feature scores: {scores_path}")
    print(f"  • Top features (tensor): {top_features_path}")
    print(f"  • Feature list (JSON): {feature_list_path}")
    if generate_dashboard:
        print(f"  • Dashboard: {os.path.join(output_dir, 'dashboards', f'features-{len(top_indices)}.html')}")
    print("=" * 80)
    
    return feature_list


if __name__ == "__main__":
    fire.Fire(run_feature_search)

