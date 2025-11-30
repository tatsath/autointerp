#!/usr/bin/env python3
"""
Show top activating tokens for a feature.

This script loads examples from LatentDataset and shows the top activating tokens
with their activation values and frequencies.
"""

import json
import argparse
import torch
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig


def show_top_tokens_for_feature(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    top_k: int = 30,
    output_file: Path = None,
):
    """
    Show top activating tokens for a feature.
    
    Args:
        results_dir: Directory containing AutoInterp results
        hookpoint: Hookpoint name (e.g., "layers.19")
        feature_id: Feature ID to analyze
        base_model: Base model name for tokenizer
        top_k: Number of top tokens to show
        output_file: Optional file to save output
    """
    latents_dir = results_dir / "latents"
    
    # Load run config
    run_config_path = results_dir / "run_config.json"
    if not run_config_path.exists():
        print(f"Error: run_config.json not found in {results_dir}")
        return
    
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create configs
    sampler_cfg = SamplerConfig(**run_config["sampler_cfg"])
    constructor_cfg = ConstructorConfig(**run_config["constructor_cfg"])
    
    # Create dataset for this feature
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    
    print(f"\nAnalyzing top tokens for feature {feature_id}...")
    
    try:
        dataset = LatentDataset(
            raw_dir=str(latents_dir),
            modules=[hookpoint],
            sampler_cfg=sampler_cfg,
            constructor_cfg=constructor_cfg,
            latents=latents_dict,
            tokenizer=tokenizer,
        )
        
        # Find the record
        record = None
        for r in dataset:
            latent_val = r.latent
            if hasattr(latent_val, 'latent_index'):
                latent_val = latent_val.latent_index
            elif hasattr(latent_val, '__int__'):
                latent_val = int(latent_val)
            else:
                try:
                    latent_val = int(latent_val)
                except:
                    continue
            
            if latent_val == feature_id:
                record = r
                break
        
        if record is None:
            print(f"  Warning: Feature {feature_id} not found in dataset")
            return
        
        # Get examples
        examples = record.examples if hasattr(record, "examples") and record.examples else []
        
        if not examples:
            print(f"  No examples found for feature {feature_id}")
            return
        
        print(f"  Found {len(examples)} activating examples")
        
        # Collect all tokens with their activations
        token_activations = []  # List of (token_str, activation_value)
        
        for example in examples:
            tokens = example.tokens
            tokens_list = tokens.tolist()
            activations = example.activations.tolist()
            
            # Decode tokens
            str_tokens = [
                tokenizer.decode([t], skip_special_tokens=False) for t in tokens_list
            ]
            
            # Add to collection
            for token_str, activation in zip(str_tokens, activations):
                token_clean = token_str.strip()
                if token_clean:  # Skip empty tokens
                    token_activations.append((token_clean, float(activation)))
        
        # Aggregate by token (sum activations, count occurrences)
        token_stats = {}
        for token, activation in token_activations:
            if token not in token_stats:
                token_stats[token] = {
                    'total_activation': 0.0,
                    'max_activation': 0.0,
                    'count': 0,
                    'avg_activation': 0.0
                }
            token_stats[token]['total_activation'] += activation
            token_stats[token]['max_activation'] = max(token_stats[token]['max_activation'], activation)
            token_stats[token]['count'] += 1
        
        # Calculate averages
        for token in token_stats:
            token_stats[token]['avg_activation'] = (
                token_stats[token]['total_activation'] / token_stats[token]['count']
            )
        
        # Sort by max activation (or total activation)
        sorted_tokens = sorted(
            token_stats.items(),
            key=lambda x: x[1]['max_activation'],
            reverse=True
        )[:top_k]
        
        # Prepare output
        output_lines = []
        output_lines.append("=" * 80)
        output_lines.append(f"TOP {len(sorted_tokens)} ACTIVATING TOKENS FOR FEATURE {feature_id}")
        output_lines.append("=" * 80)
        output_lines.append("")
        output_lines.append(f"Total examples analyzed: {len(examples)}")
        output_lines.append(f"Total token activations: {len(token_activations)}")
        output_lines.append("")
        output_lines.append(f"{'Rank':<6} {'Token':<30} {'Max Act':<12} {'Avg Act':<12} {'Count':<8}")
        output_lines.append("-" * 80)
        
        for rank, (token, stats) in enumerate(sorted_tokens, 1):
            token_display = token[:28] + ".." if len(token) > 30 else token
            output_lines.append(
                f"{rank:<6} {token_display:<30} {stats['max_activation']:<12.4f} "
                f"{stats['avg_activation']:<12.4f} {stats['count']:<8}"
            )
        
        output_text = "\n".join(output_lines)
        print(output_text)
        
        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(output_text)
            print(f"\n  Saved to: {output_file}")
        
        # Also save as JSON
        json_data = {
            "feature_id": feature_id,
            "total_examples": len(examples),
            "total_token_activations": len(token_activations),
            "top_tokens": [
                {
                    "rank": rank,
                    "token": token,
                    "max_activation": stats['max_activation'],
                    "avg_activation": stats['avg_activation'],
                    "count": stats['count'],
                    "total_activation": stats['total_activation'],
                }
                for rank, (token, stats) in enumerate(sorted_tokens, 1)
            ]
        }
        
        if output_file:
            json_file = output_file.with_suffix('.json')
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"  Saved JSON to: {json_file}")
        
    except Exception as e:
        print(f"  Error analyzing tokens: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Show top activating tokens for a feature"
    )
    parser.add_argument(
        "results_dir",
        type=str,
        help="Path to AutoInterp results directory",
    )
    parser.add_argument(
        "--hookpoint",
        type=str,
        required=True,
        help="Hookpoint name (e.g., 'layers.19')",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name for tokenizer",
    )
    parser.add_argument(
        "--feature_id",
        type=int,
        required=True,
        help="Feature ID to analyze",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help="Number of top tokens to show (default: 30)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional file to save output (default: Analysis/feature_analysis/TOP_TOKENS_{feature_id}.txt)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.output_file is None:
        output_file = results_dir / "Analysis" / "feature_analysis" / f"TOP_TOKENS_{args.feature_id}.txt"
    else:
        output_file = Path(args.output_file)
    
    show_top_tokens_for_feature(
        results_dir=results_dir,
        hookpoint=args.hookpoint,
        feature_id=args.feature_id,
        base_model=args.base_model,
        top_k=args.top_k,
        output_file=output_file,
    )


if __name__ == "__main__":
    main()

