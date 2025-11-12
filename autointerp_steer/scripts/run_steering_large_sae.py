#!/usr/bin/env python3
"""
Run steering experiments on large SAE using local safetensors.
Adapts the existing steering infrastructure for Llama-3.1-8B-Instruct + large SAE.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae_lens import HookedSAETransformer
from sae_pipeline.steering import run_steering_experiment
from sae_pipeline.steering_utils import (
    get_features_per_layer_from_csv,
    load_prompts_from_dataset
)


def main():
    parser = argparse.ArgumentParser(
        description="Run steering experiments on large SAE features.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Where to save generated texts')
    parser.add_argument('--sae_path', type=str, required=True,
                       help='Path to local SAE directory or safetensors file')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                       help='Base model name')
    parser.add_argument('--layer', type=int, default=19,
                       help='Layer number to analyze')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Torch device')
    
    # Feature selection
    parser.add_argument('--features_csv', type=str, default=None,
                       help='CSV file with features (e.g., similarity map)')
    parser.add_argument('--feature_column', type=str, default='large_feature',
                       help='Column name in CSV containing feature IDs')
    parser.add_argument('--top_n_features', type=int, default=10,
                       help='Number of top features to analyze')
    parser.add_argument('--features_list', type=int, nargs='+', default=None,
                       help='Explicit list of feature IDs (overrides CSV)')
    
    # Prompt arguments
    parser.add_argument('--prompts_file', type=str, default=None,
                       help='JSON file with prompts (overrides dataset)')
    parser.add_argument('--dataset_repo', type=str, required=True,
                       help='HuggingFace dataset repository for prompts')
    parser.add_argument('--dataset_name', type=str, default='default',
                       help='Dataset name/config')
    parser.add_argument('--dataset_split', type=str, default='train[:50]',
                       help='Dataset split')
    parser.add_argument('--num_prompts', type=int, default=10,
                       help='Number of prompts to use (default: 10 for speed)')
    parser.add_argument('--num_batches', type=int, default=1,
                       help='Number of batches for max activation search (default: 1 for speed)')
    parser.add_argument('--max_new_tokens', type=int, default=32,
                       help='Maximum tokens to generate (default: 32 for speed)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok=True)
    
    print("="*60)
    print("🚀 Large SAE Steering Experiment")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"SAE Path: {args.sae_path}")
    print(f"Layer: {args.layer}")
    print(f"Output: {args.output_folder}")
    print()
    
    # Load model
    print("Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_name, device=args.device)
    print("✓ Model loaded")
    
    # Load prompts
    print("\nLoading prompts...")
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
        print(f"✓ Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = load_prompts_from_dataset(
            args.dataset_repo,
            args.dataset_name,
            args.dataset_split,
            args.num_prompts
        )
        print(f"✓ Loaded {len(prompts)} prompts from dataset")
    
    # Get features
    print("\nSelecting features...")
    if args.features_list:
        # Use explicit feature list
        feature_list = [(int(f), 1.0) for f in args.features_list]
        top_features_per_layer = {args.layer: feature_list}
        print(f"✓ Using {len(args.features_list)} explicitly provided features")
    elif args.features_csv and os.path.exists(args.features_csv):
        # Load from CSV
        top_features_per_layer = get_features_per_layer_from_csv(
            args.features_csv,
            args.feature_column,
            args.layer,
            args.top_n_features
        )
        num_features = len(top_features_per_layer[args.layer])
        print(f"✓ Loaded {num_features} features from {args.features_csv}")
    else:
        raise ValueError(
            "Must provide either --features_list or --features_csv. "
            f"CSV path was: {args.features_csv}"
        )
    
    # Run steering experiment
    print(f"\n{'='*60}")
    print("Starting steering experiments...")
    print(f"{'='*60}\n")
    
    # Pass dataset as tuple (repo, config) for ActivationsStore
    dataset_identifier = (args.dataset_repo, args.dataset_name) if args.dataset_name != "default" else args.dataset_repo
    
    run_steering_experiment(
        model=model,
        prompts=prompts,
        top_features_per_layer=top_features_per_layer,
        layers=[args.layer],
        output_folder=args.output_folder,
        device=args.device,
        sae_path=args.sae_path,
        dataset=dataset_identifier,
        num_batches=args.num_batches,
        max_new_tokens=args.max_new_tokens
    )
    
    print(f"\n{'='*60}")
    print("✅ Steering experiments completed!")
    print(f"Results saved to: {args.output_folder}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

