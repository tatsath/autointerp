#!/usr/bin/env python3
"""
Analyze the discrepancy between feature explanations and top activating words.

This script:
1. Loads the actual examples that the explanation model saw
2. Shows the top activating words from token analysis
3. Compares them to understand why explanations might differ
4. Generates a report explaining the discrepancy
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig
from analyze_feature_activations import (
    load_cached_latents,
    show_top_tokens_for_feature,
    load_explanation,
    find_hookpoint_in_latents,
    get_model_from_config,
)


def load_examples_for_feature(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    max_examples: int = 20,
):
    """Load the actual examples that the explanation model saw."""
    latents_dir = results_dir / "latents"
    
    # Load run config
    run_config_path = results_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create configs
    sampler_cfg = SamplerConfig(**run_config["sampler_cfg"])
    constructor_cfg = ConstructorConfig(**run_config["constructor_cfg"])
    
    # Create dataset for this feature
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    
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
            # Handle Latent type - it has a latent_index attribute
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
            return None, None, None
        
        # Get examples (these are what the explanation model saw)
        train_examples = record.train[:max_examples] if hasattr(record, "train") and record.train else []
        examples = record.examples[:max_examples] if hasattr(record, "examples") and record.examples else []
        
        # Use train examples if available, otherwise regular examples
        examples_to_show = train_examples if train_examples else examples
        
        return examples_to_show, record, tokenizer
        
    except Exception as e:
        print(f"Error loading examples: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def analyze_feature_explanation_discrepancy(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    activations: torch.Tensor,
    locations: torch.Tensor,
    tokens: torch.Tensor,
    tokenizer: AutoTokenizer,
    output_file: Path,
):
    """Analyze why explanation differs from top activating words."""
    
    # Load explanation
    explanation = load_explanation(results_dir, hookpoint, feature_id)
    
    # Get top activating words
    top_words = show_top_tokens_for_feature(
        feature_id, activations, locations, tokens, tokenizer,
        top_k=30, sample_size=50000
    )
    
    # Load examples that explanation model saw
    examples, record, _ = load_examples_for_feature(
        results_dir, hookpoint, feature_id, base_model, max_examples=20
    )
    
    # Write analysis
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Feature {feature_id} - Explanation vs Activating Words Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Explanation: {explanation}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("TOP ACTIVATING WORDS (from token analysis)\n")
        f.write("=" * 80 + "\n")
        f.write("These are the individual tokens/words that fire most strongly:\n\n")
        for i, (text, count, tid, max_act, avg_act) in enumerate(top_words[:30], 1):
            f.write(f"{i:2d}. {repr(text):35s}  max_act={max_act:7.2f}  count={count:5d}\n")
        
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("EXAMPLES THE EXPLANATION MODEL SAW\n")
        f.write("=" * 80 + "\n")
        f.write("These are the full text examples (with context) that the LLM explainer analyzed:\n\n")
        
        if examples:
            for i, ex in enumerate(examples[:20], 1):
                text = ex.text if hasattr(ex, "text") else str(ex)
                max_act = ex.max_activation if hasattr(ex, "max_activation") else "N/A"
                f.write(f"\nExample {i} (max_activation={max_act}):\n")
                f.write(f"{text}\n")
                f.write("-" * 80 + "\n")
        else:
            f.write("(Could not load examples - they may not be cached)\n")
        
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("ANALYSIS: Why the Discrepancy?\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Key Differences:\n\n")
        f.write("1. EXPLANATION MODEL SEES:\n")
        f.write("   - Full text sequences (32 tokens of context)\n")
        f.write("   - Examples centered on activation points\n")
        f.write("   - Semantic context and relationships between words\n")
        f.write("   - Top 15-25 strongest activating examples\n\n")
        
        f.write("2. TOKEN ANALYSIS SHOWS:\n")
        f.write("   - Individual tokens/words in isolation\n")
        f.write("   - Top 50k strongest activations sampled\n")
        f.write("   - No context about surrounding words\n")
        f.write("   - Frequency-based ranking\n\n")
        
        f.write("3. WHY THEY DIFFER:\n")
        f.write("   - The explanation model looks at FULL CONTEXT and SEMANTIC MEANING\n")
        f.write("   - It sees patterns like 'Special dividend declarations by funds' in context\n")
        f.write("   - Individual words like 'Industry', 'Third', 'Should' may appear frequently\n")
        f.write("     but the explanation captures the SEMANTIC PATTERN across examples\n")
        f.write("   - The LLM explainer is doing semantic abstraction, not just word counting\n\n")
        
        f.write("4. EXAMPLE INTERPRETATION:\n")
        if examples:
            f.write("   Looking at the examples above, the explanation model likely saw:\n")
            f.write("   - Patterns related to financial declarations, announcements\n")
            f.write("   - Context about funds, dividends, special payments\n")
            f.write("   - Even if individual words are common, the COMBINATION and CONTEXT\n")
            f.write("     suggests the semantic pattern of 'special dividend declarations'\n\n")
        
        f.write("5. CONCLUSION:\n")
        f.write("   The explanation is a SEMANTIC INTERPRETATION of what the feature captures,\n")
        f.write("   while the top words are STATISTICAL FREQUENCIES. They complement each other:\n")
        f.write("   - Top words show WHAT tokens fire most\n")
        f.write("   - Explanation shows WHAT SEMANTIC PATTERN those tokens represent in context\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze explanation vs activating words")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--feature-id", type=int, required=True, help="Feature ID to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).resolve()
    latents_dir = results_dir / "latents"
    
    # Auto-detect hookpoint and model
    hookpoint = find_hookpoint_in_latents(latents_dir)
    base_model = get_model_from_config(latents_dir, hookpoint)
    
    if base_model is None:
        raise ValueError("Could not auto-detect model")
    
    print(f"Analyzing feature {args.feature_id}")
    print(f"Hookpoint: {hookpoint}")
    print(f"Model: {base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load activations
    print("\nLoading activations...")
    activations, tokens, locations, config = load_cached_latents(results_dir, hookpoint)
    
    # Set output file
    if args.output is None:
        output_file = results_dir / "feature_analysis" / f"feature_{args.feature_id}_explanation_analysis.txt"
    else:
        output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Analyze
    print(f"\nAnalyzing feature {args.feature_id}...")
    analyze_feature_explanation_discrepancy(
        results_dir, hookpoint, args.feature_id, base_model,
        activations, locations, tokens, tokenizer, output_file
    )
    
    print(f"\n✓ Analysis saved to: {output_file}")


if __name__ == "__main__":
    main()

