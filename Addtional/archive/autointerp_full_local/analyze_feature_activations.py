#!/usr/bin/env python3
"""
Analyze feature activations from autointerp_full results.

This script:
1. Loads cached activations from latents directory
2. Computes per-feature statistics (max activation, coverage, importance)
3. Shows top tokens where each feature fires most
4. Displays positive (activating) and negative (FAISS hard negatives) examples
"""

import json
import torch
import argparse
import re
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoTokenizer
from collections import defaultdict
import sys

# Import autointerp_full_local components
from autointerp_full_local.latents.loader import LatentDataset
from autointerp_full_local.config import SamplerConfig, ConstructorConfig


def safe_mean_activation(mean_act_tensor: torch.Tensor, feat_id: int) -> float:
    """Safely get mean activation, handling inf/nan."""
    val = mean_act_tensor[feat_id].item()
    if torch.isinf(torch.tensor(val)) or torch.isnan(torch.tensor(val)):
        return 0.0
    return float(val)


def find_hookpoint_in_latents(latents_dir: Path) -> str:
    """Auto-detect hookpoint from latents directory structure."""
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    # Look for subdirectories in latents (each hookpoint has its own directory)
    hookpoint_dirs = [d for d in latents_dir.iterdir() if d.is_dir() and (d / "config.json").exists()]
    
    if len(hookpoint_dirs) == 0:
        raise FileNotFoundError(f"No hookpoint directories found in {latents_dir}")
    elif len(hookpoint_dirs) == 1:
        return hookpoint_dirs[0].name
    else:
        # Multiple hookpoints - return the first one and warn
        print(f"Warning: Multiple hookpoints found: {[d.name for d in hookpoint_dirs]}")
        print(f"Using: {hookpoint_dirs[0].name}")
        return hookpoint_dirs[0].name


def get_model_from_config(latents_dir: Path, hookpoint: str) -> str:
    """Try to get model name from config.json."""
    config_path = latents_dir / hookpoint / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            if "model_name" in config:
                return config["model_name"]
    return None


def load_cached_latents(results_dir: Path, hookpoint: str):
    """Load cached latent activations from safetensors files."""
    latents_dir = results_dir / "latents" / hookpoint
    
    if not latents_dir.exists():
        raise FileNotFoundError(f"Latents directory not found: {latents_dir}")
    
    # Load config
    config_path = latents_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Find all safetensors files (they're named like "0_79.safetensors")
    safetensor_files = sorted(latents_dir.glob("*.safetensors"))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {latents_dir}")
    
    print(f"Loading {len(safetensor_files)} safetensors files...")
    
    # Load all shards
    all_activations = []
    all_tokens = []
    all_locations = []
    
    for sf_file in safetensor_files:
        data = load_file(str(sf_file))
        
        # autointerp_full saves sparse format: locations [n_nonzero, 3], activations [n_nonzero]
        # locations columns: [batch_idx, seq_idx, feature_idx]
        # safetensors can return either numpy or torch tensors
        if "activations" in data:
            act = data["activations"]
            if isinstance(act, torch.Tensor):
                all_activations.append(act)
            else:
                all_activations.append(torch.from_numpy(act))
        if "tokens" in data:
            tok = data["tokens"]
            if isinstance(tok, torch.Tensor):
                all_tokens.append(tok)
            else:
                all_tokens.append(torch.from_numpy(tok))
        if "locations" in data:
            loc = data["locations"]
            if isinstance(loc, torch.Tensor):
                all_locations.append(loc)
            else:
                all_locations.append(torch.from_numpy(loc))
    
    # Concatenate all shards
    if all_activations:
        activations = torch.cat(all_activations, dim=0)
        print(f"  Activations shape: {activations.shape} (sparse, {len(activations)} non-zero)")
    else:
        raise ValueError("No activations found in safetensors files")
    
    if all_tokens:
        tokens = torch.cat(all_tokens, dim=0)
        print(f"  Tokens shape: {tokens.shape}")
    else:
        tokens = None
        print("  Warning: No tokens found")
    
    if all_locations:
        locations = torch.cat(all_locations, dim=0)
        print(f"  Locations shape: {locations.shape} (columns: [batch_idx, seq_idx, feature_idx])")
    else:
        raise ValueError("No locations found in safetensors files")
    
    return activations, tokens, locations, config


def compute_feature_statistics(
    activations: torch.Tensor, 
    locations: torch.Tensor, 
    top_k_features: int = None,
    specific_features: list[int] = None
):
    """Compute per-feature statistics from sparse activations - OPTIMIZED."""
    # locations: [n_nonzero, 3] where columns are [batch_idx, seq_idx, feature_idx]
    # activations: [n_nonzero]
    
    n_nonzero = len(activations)
    feature_indices = locations[:, 2].long()  # Extract feature indices
    
    # Get number of features (max feature index + 1)
    num_features = int(feature_indices.max().item()) + 1
    
    print(f"\nComputing statistics for {num_features} features from {n_nonzero} non-zero activations...")
    print("  Using vectorized operations for speed...")
    
    # Use scatter operations for much faster aggregation
    # Max activation per feature
    max_act = torch.zeros(num_features, dtype=activations.dtype, device=activations.device)
    max_act.scatter_reduce_(0, feature_indices, activations, reduce="amax", include_self=False)
    
    # Sum and count per feature
    sum_act = torch.zeros(num_features, dtype=activations.dtype, device=activations.device)
    count_act = torch.zeros(num_features, dtype=torch.long, device=activations.device)
    sum_act.scatter_add_(0, feature_indices, activations)
    count_act.scatter_add_(0, feature_indices, torch.ones_like(feature_indices, dtype=torch.long))
    
    # Mean activation (handle division by zero properly)
    # Use float32 for calculation to avoid inf issues
    mean_act = torch.zeros(num_features, dtype=torch.float32, device=activations.device)
    nonzero_mask = count_act > 0
    if nonzero_mask.any():
        mean_act[nonzero_mask] = sum_act[nonzero_mask].float() / count_act[nonzero_mask].float()
    # Set mean to 0 for features with no activations (instead of inf/nan)
    mean_act[~nonzero_mask] = 0.0
    
    # Coverage: fraction of non-zero activations
    total_possible = n_nonzero
    coverage = count_act.float() / total_possible
    
    # Importance = max_act * coverage (heuristic)
    importance = max_act * coverage
    
    # Determine which features to analyze
    if specific_features is not None:
        # Use specific features provided
        valid_features = [fid for fid in specific_features if fid < num_features]
        if len(valid_features) < len(specific_features):
            invalid = set(specific_features) - set(valid_features)
            print(f"  Warning: {len(invalid)} features out of range: {sorted(invalid)[:10]}...")
        feature_list = [(fid, importance[fid].item()) for fid in valid_features]
        # Sort by importance
        feature_list.sort(key=lambda x: x[1], reverse=True)
    elif top_k_features is not None:
        # Use top K features
        k = min(top_k_features, num_features)
        top_vals, top_idx = torch.topk(importance, k=k)
        feature_list = list(zip(top_idx.tolist(), top_vals.tolist()))
    else:
        # Default: top 10
        k = min(10, num_features)
        top_vals, top_idx = torch.topk(importance, k=k)
        feature_list = list(zip(top_idx.tolist(), top_vals.tolist()))
    
    stats = {
        "max_activation": max_act,
        "coverage": coverage,
        "mean_activation": mean_act,
        "count": count_act,
        "importance": importance,
        "top_features": feature_list,
        "num_features": num_features,
    }
    
    return stats


def show_top_tokens_for_feature(
    feature_id: int,
    activations: torch.Tensor,
    locations: torch.Tensor,
    tokens: torch.Tensor,
    tokenizer: AutoTokenizer,
    top_k: int = 30,
    sample_size: int = 50000,  # Sample top activations for speed
):
    """Show top K tokens by activation value (not count) for a feature.
    
    Args:
        top_k: Number of top tokens to return
        sample_size: Number of top activations to sample (for speed)
    Returns:
        List of tuples: (token_text, count, token_id, max_act, avg_act)
    """
    # Convert locations to long for indexing compatibility
    locations = locations.long()
    # Filter for this feature
    feature_mask = locations[:, 2] == feature_id
    feature_acts = activations[feature_mask]
    feature_locations = locations[feature_mask]
    
    if len(feature_acts) == 0:
        return []
    
    if tokens is None:
        return []
    
    # Sample top activations for speed (process strongest activations)
    n_sample = min(sample_size, len(feature_acts))
    if n_sample < len(feature_acts):
        top_indices = torch.topk(feature_acts, k=n_sample).indices
        sampled_acts = feature_acts[top_indices]
        sampled_locations = feature_locations[top_indices]
    else:
        sampled_acts = feature_acts
        sampled_locations = feature_locations
    
    # locations: [batch_idx, seq_idx, feature_idx]
    # tokens: [batch, seq]
    token_data = defaultdict(lambda: {"count": 0, "max_act": float("-inf"), "sum_act": 0.0})
    
    # Process sampled activations
    for i, loc in enumerate(sampled_locations):
        batch_idx = int(loc[0].item())
        seq_idx = int(loc[1].item())
        act_value = sampled_acts[i].item()
        
        if batch_idx < tokens.shape[0] and seq_idx < tokens.shape[1]:
            tok_id = tokens[batch_idx, seq_idx].item()
            token_data[tok_id]["count"] += 1
            token_data[tok_id]["sum_act"] += act_value
            token_data[tok_id]["max_act"] = max(token_data[tok_id]["max_act"], act_value)
    
    # Calculate average activation and filter meaningful tokens
    meaningful_tokens = []
    for tid, data in token_data.items():
        if data["count"] > 0:
            avg_act = data["sum_act"] / data["count"]
            text = tokenizer.decode([tid]).strip()
            
            # Filter out pure punctuation, numbers, and whitespace
            # Keep tokens that have letters or are meaningful
            if text and (any(c.isalpha() for c in text) or len(text) > 2):
                meaningful_tokens.append((
                    text,
                    data["count"],
                    tid,
                    data["max_act"],
                    avg_act
                ))
    
    # Sort by max_act (strongest activations first), then by count
    meaningful_tokens.sort(key=lambda x: (x[3], x[1]), reverse=True)
    
    return meaningful_tokens[:top_k]


def load_examples_from_dataset(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
):
    """Load activating and non-activating examples using LatentDataset."""
    latents_dir = results_dir / "latents"
    
    # Load run config to get constructor config
    run_config_path = results_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create configs from run_config
    sampler_cfg = SamplerConfig(**run_config["sampler_cfg"])
    constructor_cfg = ConstructorConfig(**run_config["constructor_cfg"])
    
    # Create dataset for this specific feature
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    
    print(f"\nLoading examples for feature {feature_id}...")
    try:
        dataset = LatentDataset(
            raw_dir=str(latents_dir),
            modules=[hookpoint],
            sampler_cfg=sampler_cfg,
            constructor_cfg=constructor_cfg,
            latents=latents_dict,
            tokenizer=tokenizer,
        )
        
        # Find the record for this feature
        record = None
        for r in dataset:
            if int(r.latent) == feature_id:
                record = r
                break
        
        if record is None:
            print(f"  Warning: Feature {feature_id} not found in dataset")
            return None, None
        
        # Get examples
        pos_examples = record.examples if hasattr(record, "examples") else []
        neg_examples = record.not_active if hasattr(record, "not_active") else []
        
        print(f"  Found {len(pos_examples)} activating examples")
        print(f"  Found {len(neg_examples)} non-activating examples")
        
        return pos_examples, neg_examples
        
    except Exception as e:
        print(f"  Error loading examples: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def extract_features_from_explanations(results_dir: Path, hookpoint: str) -> list[int]:
    """Extract feature IDs from explanation files."""
    explanations_dir = results_dir / "explanations"
    if not explanations_dir.exists():
        raise FileNotFoundError(f"Explanations directory not found: {explanations_dir}")
    
    # Pattern: hookpoint_latent{feature_id}.txt
    pattern = f"{hookpoint}_latent"
    feature_ids = []
    
    for file in explanations_dir.glob(f"{pattern}*.txt"):
        # Extract feature ID from filename like "backbone.layers.28_latent2330.txt"
        filename = file.name
        # Remove prefix and .txt suffix
        feature_str = filename.replace(f"{pattern}", "").replace(".txt", "")
        try:
            feature_id = int(feature_str)
            feature_ids.append(feature_id)
        except ValueError:
            print(f"Warning: Could not parse feature ID from {filename}")
    
    return sorted(set(feature_ids))  # Remove duplicates and sort


def extract_features_from_summary_file(summary_path: Path, max_features: int = None) -> list[int]:
    """Extract feature IDs from summary file (same format as run_nemotron.sh uses).
    
    Expected format: lines like "1. Feature 1234: description"
    """
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
    feature_ids = []
    
    with open(summary_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                feature_id = int(match.group(1))
                feature_ids.append(feature_id)
                if max_features and len(feature_ids) >= max_features:
                    break
    
    return feature_ids


def extract_features_from_list_file(list_path: Path) -> list[int]:
    """Extract feature IDs from a space-separated list file (same as nemotron_finance_news_features_list.txt).
    
    Expected format: "1234 5678 9012 ..." (space-separated integers)
    """
    if not list_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {list_path}")
    
    with open(list_path, 'r') as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"Feature list file is empty: {list_path}")
        
        # Split by whitespace and convert to integers
        feature_ids = [int(x) for x in content.split()]
    
    return feature_ids


def load_explanation(results_dir: Path, hookpoint: str, feature_id: int) -> str:
    """Load explanation text for a feature if it exists."""
    explanations_dir = results_dir / "explanations"
    explanation_file = explanations_dir / f"{hookpoint}_latent{feature_id}.txt"
    
    if explanation_file.exists():
        try:
            with open(explanation_file, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"  Warning: Could not read explanation for feature {feature_id}: {e}")
            return None
    return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze feature activations from autointerp_full_local results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to results directory containing latents/ subdirectory"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (e.g., 'meta-llama/Llama-3.1-8B-Instruct'). "
             "If not provided, will try to auto-detect from config.json"
    )
    
    parser.add_argument(
        "--hookpoint",
        type=str,
        default=None,
        help="Hookpoint name (e.g., 'layers.19' or 'backbone.layers.28'). "
             "If not provided, will auto-detect from latents directory"
    )
    
    parser.add_argument(
        "--top-k-features",
        type=int,
        default=None,
        help="Number of top features to analyze in detail (ignored if --features-from-explanations is used)"
    )
    
    parser.add_argument(
        "--features-from-explanations",
        action="store_true",
        help="Analyze all features that have explanations in the explanations/ directory"
    )
    
    parser.add_argument(
        "--features-from-summary",
        type=str,
        default=None,
        help="Path to features summary file (e.g., top_finance_features_summary.txt). "
             "Extracts features using same pattern as run_nemotron.sh"
    )
    
    parser.add_argument(
        "--max-features-from-summary",
        type=int,
        default=None,
        help="Maximum number of features to extract from summary file (default: all)"
    )
    
    parser.add_argument(
        "--features-from-list",
        type=str,
        default=None,
        help="Path to features list file (e.g., nemotron_finance_news_features_list.txt). "
             "Reads space-separated feature IDs, same format as generated by run_nemotron.sh"
    )
    
    parser.add_argument(
        "--top-tokens",
        type=int,
        default=30,
        help="Number of top tokens to show per feature (sorted by activation strength)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of top activations to sample per feature (for speed, default: 50000)"
    )
    
    parser.add_argument(
        "--num-pos-examples",
        type=int,
        default=10,
        help="Number of positive examples to show"
    )
    
    parser.add_argument(
        "--num-neg-examples",
        type=int,
        default=10,
        help="Number of negative examples to show"
    )
    
    parser.add_argument(
        "--skip-faiss-loading",
        action="store_true",
        help="Skip slow FAISS example loading for faster execution"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis results (default: results_dir/feature_analysis)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert results_dir to Path
    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    latents_dir = results_dir / "latents"
    
    # Auto-detect hookpoint if not provided
    if args.hookpoint is None:
        print("Auto-detecting hookpoint from latents directory...")
        hookpoint = find_hookpoint_in_latents(latents_dir)
        print(f"  Detected hookpoint: {hookpoint}")
    else:
        hookpoint = args.hookpoint
    
    # Auto-detect model if not provided
    if args.base_model is None:
        print("Auto-detecting model from config.json...")
        base_model = get_model_from_config(latents_dir, hookpoint)
        if base_model is None:
            raise ValueError(
                "Could not auto-detect model name. Please provide --base-model argument. "
                f"Expected config.json at: {latents_dir / hookpoint / 'config.json'}"
            )
        print(f"  Detected model: {base_model}")
    else:
        base_model = args.base_model
    
    # Set output directory
    if args.output_dir is None:
        output_dir = results_dir / "feature_analysis"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("Feature Activation Analysis")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Base model: {base_model}")
    print(f"Hookpoint: {hookpoint}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Determine which features to analyze
    specific_features = None
    if args.features_from_list:
        print("\nReading features from list file...")
        list_path = Path(args.features_from_list).expanduser().resolve()
        specific_features = extract_features_from_list_file(list_path)
        print(f"  Found {len(specific_features)} features in list file")
        if len(specific_features) == 0:
            raise ValueError(f"No features found in list file: {list_path}")
    elif args.features_from_summary:
        print("\nExtracting features from summary file...")
        summary_path = Path(args.features_from_summary).expanduser().resolve()
        specific_features = extract_features_from_summary_file(
            summary_path, 
            max_features=args.max_features_from_summary
        )
        print(f"  Found {len(specific_features)} features in summary file")
        if len(specific_features) == 0:
            raise ValueError(f"No features found in summary file: {summary_path}")
    elif args.features_from_explanations:
        print("\nExtracting features from explanations directory...")
        specific_features = extract_features_from_explanations(results_dir, hookpoint)
        print(f"  Found {len(specific_features)} features with explanations")
        if len(specific_features) == 0:
            raise ValueError(f"No features found in explanations directory: {results_dir / 'explanations'}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load cached latents
    print("\n" + "=" * 80)
    print("Loading cached activations...")
    print("=" * 80)
    activations, tokens, locations, config = load_cached_latents(results_dir, hookpoint)
    
    # Compute feature statistics
    print("\n" + "=" * 80)
    print("Computing feature statistics...")
    print("=" * 80)
    stats = compute_feature_statistics(
        activations, 
        locations, 
        top_k_features=args.top_k_features if not args.features_from_explanations else None,
        specific_features=specific_features
    )
    
    # Print features to analyze
    num_features_to_analyze = len(stats["top_features"])
    if args.features_from_explanations:
        print(f"\nAnalyzing {num_features_to_analyze} features from explanations:")
    else:
        print(f"\nTop {num_features_to_analyze} features by importance:")
    print("-" * 80)
    for idx, (feat_id, importance) in enumerate(stats["top_features"]):
        max_act = stats["max_activation"][feat_id].item()
        coverage = stats["coverage"][feat_id].item()
        mean_act = safe_mean_activation(stats["mean_activation"], feat_id)
        print(
            f"  {idx+1:2d}. Feature {feat_id:6d}  "
            f"importance={importance:.4f}  "
            f"max={max_act:.3f}  "
            f"coverage={coverage:.4f}  "
            f"mean={mean_act:.4f}"
        )
    
    # Save statistics
    stats_file = output_dir / "feature_statistics.json"
    stats_dict = {
        "top_features": [
            {
                "feature_id": feat_id,
                "importance": float(importance),
                "max_activation": float(stats["max_activation"][feat_id].item()),
                "coverage": float(stats["coverage"][feat_id].item()),
                "mean_activation": float(stats["mean_activation"][feat_id].item()),
            }
            for feat_id, importance in stats["top_features"]
        ]
    }
    with open(stats_file, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n✓ Saved statistics to {stats_file}")
    
    # Analyze each feature in detail
    print("\n" + "=" * 80)
    print("Detailed Feature Analysis")
    print("=" * 80)
    print(f"Analyzing {len(stats['top_features'])} features...")
    
    for idx, (feat_id, importance) in enumerate(stats["top_features"]):
        print(f"\n{'=' * 80}")
        print(f"Feature {feat_id} (Rank {idx+1}, Importance={importance:.4f})")
        print(f"{'=' * 80}")
        
        # Load explanation if available
        explanation = load_explanation(results_dir, hookpoint, feat_id)
        if explanation:
            print(f"\nExplanation: {explanation}")
        
        # Top tokens
        top_tokens = []
        if tokens is not None:
            top_tokens = show_top_tokens_for_feature(
                feat_id, activations, locations, tokens, tokenizer, 
                top_k=args.top_tokens,
                sample_size=args.sample_size
            )
            if top_tokens:
                print(f"\nTop {len(top_tokens)} activating words (by activation strength):")
                print("-" * 80)
                for i, (text, count, tid, max_act, avg_act) in enumerate(top_tokens):
                    print(
                        f"  {i+1:2d}. {repr(text):35s}  "
                        f"max_act={max_act:7.2f}  "
                        f"avg_act={avg_act:7.2f}  "
                        f"count={count:5d}"
                    )
        
        # Load examples (skip if --skip-faiss-loading is set)
        if args.skip_faiss_loading:
            pos_examples, neg_examples = None, None
            print("\n(Skipping FAISS example loading for speed)")
        else:
            pos_examples, neg_examples = load_examples_from_dataset(
                results_dir, hookpoint, feat_id, base_model
            )
        
        # Show positive examples
        if pos_examples:
            print(f"\n{'=' * 80}")
            print(f"POSITIVE (Activating) Examples ({min(args.num_pos_examples, len(pos_examples))} shown):")
            print(f"{'=' * 80}")
            for i, ex in enumerate(pos_examples[:args.num_pos_examples]):
                text = ex.text.replace("\n", " ").strip()
                activation = ex.activation if hasattr(ex, "activation") else "N/A"
                print(f"\n[{i+1}] Activation: {activation}")
                print(f"    {text}")
                print("-" * 80)
        
        # Show negative examples
        if neg_examples:
            print(f"\n{'=' * 80}")
            print(f"NEGATIVE (FAISS Hard Negatives) Examples ({min(args.num_neg_examples, len(neg_examples))} shown):")
            print(f"{'=' * 80}")
            for i, ex in enumerate(neg_examples[:args.num_neg_examples]):
                text = ex.text.replace("\n", " ").strip()
                activation = ex.activation if hasattr(ex, "activation") else "≈0"
                print(f"\n[{i+1}] Activation: {activation}")
                print(f"    {text}")
                print("-" * 80)
        
        # Save detailed analysis for this feature
        feature_file = output_dir / f"feature_{feat_id}_analysis.txt"
        with open(feature_file, "w") as f:
            f.write(f"Feature {feat_id} Analysis\n")
            f.write("=" * 80 + "\n\n")
            if explanation:
                f.write(f"Explanation: {explanation}\n\n")
            f.write(f"Importance: {importance:.4f}\n")
            f.write(f"Max Activation: {stats['max_activation'][feat_id].item():.4f}\n")
            f.write(f"Coverage: {stats['coverage'][feat_id].item():.4f}\n")
            f.write(f"Mean Activation: {stats['mean_activation'][feat_id].item():.4f}\n\n")
            
            if top_tokens:
                f.write(f"Top {len(top_tokens)} activating words (by activation strength):\n")
                f.write("-" * 80 + "\n")
                for i, (text, count, tid, max_act, avg_act) in enumerate(top_tokens):
                    f.write(
                        f"{i+1:2d}. {repr(text):35s}  "
                        f"max_act={max_act:7.2f}  "
                        f"avg_act={avg_act:7.2f}  "
                        f"count={count:5d}\n"
                    )
                f.write("\n")
            
            if pos_examples:
                f.write(f"\nPOSITIVE Examples:\n")
                f.write("=" * 80 + "\n")
                for i, ex in enumerate(pos_examples[:args.num_pos_examples]):
                    text = ex.text.replace("\n", " ").strip()
                    activation = ex.activation if hasattr(ex, "activation") else "N/A"
                    f.write(f"\n[{i+1}] Activation: {activation}\n")
                    f.write(f"{text}\n")
                    f.write("-" * 80 + "\n")
            
            if neg_examples:
                f.write(f"\nNEGATIVE Examples:\n")
                f.write("=" * 80 + "\n")
                for i, ex in enumerate(neg_examples[:args.num_neg_examples]):
                    text = ex.text.replace("\n", " ").strip()
                    activation = ex.activation if hasattr(ex, "activation") else "≈0"
                    f.write(f"\n[{i+1}] Activation: {activation}\n")
                    f.write(f"{text}\n")
                    f.write("-" * 80 + "\n")
        
        print(f"\n✓ Saved detailed analysis to {feature_file}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - feature_statistics.json: Overall statistics")
    print(f"  - feature_*_analysis.txt: Detailed analysis per feature")


if __name__ == "__main__":
    main()
