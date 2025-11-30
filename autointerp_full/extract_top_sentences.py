#!/usr/bin/env python3
"""
Extract top activating and non-activating sentences for each feature.

This script loads examples from the LatentDataset and extracts:
- Top positive activating sentences (sorted by max activation)
- Top negative/non-activating sentences
- Saves to analysis directory with full decoded text and activating tokens marked
"""

import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig


def mark_activating_tokens_in_text(
    full_decoded: str,
    str_tokens: list[str],
    all_activations: list[float],
    threshold: float
) -> str:
    """
    Mark activating tokens in the decoded text with <<token>> markers.
    
    Args:
        full_decoded: The fully decoded text from tokenizer.decode()
        str_tokens: List of individual token strings
        all_activations: List of activation values for each token
        threshold: Activation threshold above which tokens are marked
        
    Returns:
        Text with activating tokens marked as <<token>>
    """
    # Identify which tokens activate
    activating_indices = set()
    for i, (token_str, activation) in enumerate(zip(str_tokens, all_activations)):
        if float(activation) > threshold:
            activating_indices.add(i)
    
    if not activating_indices:
        return full_decoded
    
    # Reconstruct text from tokens with markers
    marked_parts = []
    for i, token_str in enumerate(str_tokens):
        if i in activating_indices:
            marked_parts.append(f"<<{token_str}>>")
        else:
            marked_parts.append(token_str)
    
    # Join tokens
    marked_text = "".join(marked_parts)
    
    # Check if simple concatenation matches (after normalization)
    normalized_decoded = full_decoded.replace(" ", "")
    normalized_marked = marked_text.replace("<<", "").replace(">>", "").replace(" ", "")
    
    if normalized_decoded == normalized_marked:
        return marked_text
    
    # Complex case: align tokens in full_decoded
    result = full_decoded
    tokens_to_mark = [(i, str_tokens[i]) for i in sorted(activating_indices, reverse=True)]
    
    for i, token_str in tokens_to_mark:
        token_clean = token_str.strip()
        if not token_clean:
            continue
        
        if f"<<{token_clean}>>" not in result:
            if token_clean in result:
                parts = result.rsplit(token_clean, 1)
                if len(parts) == 2:
                    result = parts[0] + f"<<{token_clean}>>" + parts[1]
    
    return result


def format_example(example, tokenizer, threshold_factor=0.1):
    """
    Format an example with activating tokens marked.
    
    Args:
        example: ActivatingExample object
        tokenizer: Tokenizer for decoding
        threshold_factor: Fraction of max activation to use as threshold
        
    Returns:
        Dictionary with formatted text and metadata
    """
    tokens = example.tokens
    tokens_list = tokens.tolist()
    all_activations = example.activations.tolist()
    max_activation = float(example.activations.max().item())
    
    # Decode full text
    full_decoded = tokenizer.decode(tokens_list, skip_special_tokens=False)
    
    # Decode individual tokens
    str_tokens = [
        tokenizer.decode([t], skip_special_tokens=False) for t in tokens_list
    ]
    
    # Calculate threshold
    threshold = max_activation * threshold_factor
    
    # Mark activating tokens
    marked_text = mark_activating_tokens_in_text(
        full_decoded, str_tokens, all_activations, threshold
    )
    
    return {
        "text": marked_text,
        "text_clean": full_decoded,
        "max_activation": max_activation,
        "num_activating_tokens": sum(1 for act in all_activations if float(act) > threshold),
    }


def extract_top_sentences_for_feature(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    top_k_positive: int = 20,
    top_k_negative: int = 10,
    output_dir: Path = None,
):
    """
    Extract top activating and non-activating sentences for a feature.
    
    Args:
        results_dir: Directory containing AutoInterp results
        hookpoint: Hookpoint name (e.g., "layers.19")
        feature_id: Feature ID to analyze
        base_model: Base model name for tokenizer
        top_k_positive: Number of top positive examples to extract
        top_k_negative: Number of top negative examples to extract
        output_dir: Directory to save output (default: results_dir/Analysis/feature_analysis)
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
    
    print(f"\nExtracting sentences for feature {feature_id}...")
    
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
        pos_examples = record.examples if hasattr(record, "examples") and record.examples else []
        neg_examples = record.not_active if hasattr(record, "not_active") and record.not_active else []
        
        print(f"  Found {len(pos_examples)} activating examples")
        print(f"  Found {len(neg_examples)} non-activating examples")
        
        # Sort positive examples by max activation
        sorted_pos = sorted(
            pos_examples,
            key=lambda e: e.max_activation if hasattr(e, 'max_activation') else 0.0,
            reverse=True
        )[:top_k_positive]
        
        # Sort negative examples (by max activation if available, otherwise random)
        sorted_neg = sorted(
            neg_examples,
            key=lambda e: e.max_activation if hasattr(e, 'max_activation') else 0.0,
            reverse=False  # Lowest activations first
        )[:top_k_negative]
        
        # Format examples
        positive_sentences = []
        for i, example in enumerate(sorted_pos, 1):
            formatted = format_example(example, tokenizer)
            formatted["rank"] = i
            positive_sentences.append(formatted)
        
        negative_sentences = []
        for i, example in enumerate(sorted_neg, 1):
            formatted = format_example(example, tokenizer)
            formatted["rank"] = i
            negative_sentences.append(formatted)
        
        # Save to file
        if output_dir is None:
            output_dir = results_dir / "Analysis" / "feature_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"TOP_SENTENCES_{feature_id}.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TOP ACTIVATING AND NON-ACTIVATING SENTENCES FOR FEATURE {feature_id}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"TOP {len(positive_sentences)} POSITIVE (ACTIVATING) SENTENCES\n")
            f.write("-" * 80 + "\n\n")
            
            for sent in positive_sentences:
                f.write(f"Rank {sent['rank']}: Max Activation = {sent['max_activation']:.4f}\n")
                f.write(f"Activating Tokens: {sent['num_activating_tokens']}\n")
                f.write(f"Text: {sent['text']}\n")
                f.write(f"Clean Text: {sent['text_clean']}\n")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            f.write(f"TOP {len(negative_sentences)} NEGATIVE (NON-ACTIVATING) SENTENCES\n")
            f.write("-" * 80 + "\n\n")
            
            for sent in negative_sentences:
                f.write(f"Rank {sent['rank']}: Max Activation = {sent['max_activation']:.4f}\n")
                f.write(f"Activating Tokens: {sent['num_activating_tokens']}\n")
                f.write(f"Text: {sent['text']}\n")
                f.write(f"Clean Text: {sent['text_clean']}\n")
                f.write("\n")
        
        print(f"  Saved to: {output_file}")
        
        # Also save as JSON for programmatic access
        json_file = output_dir / f"TOP_SENTENCES_{feature_id}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "feature_id": feature_id,
                "positive_sentences": positive_sentences,
                "negative_sentences": negative_sentences,
            }, f, indent=2)
        
        print(f"  Saved JSON to: {json_file}")
        
    except Exception as e:
        print(f"  Error extracting sentences: {e}")
        import traceback
        traceback.print_exc()


def extract_all_features(
    results_dir: Path,
    hookpoint: str,
    base_model: str,
    top_k_positive: int = 20,
    top_k_negative: int = 10,
):
    """
    Extract top sentences for all features that have explanations.
    
    Args:
        results_dir: Directory containing AutoInterp results
        hookpoint: Hookpoint name
        base_model: Base model name for tokenizer
        top_k_positive: Number of top positive examples per feature
        top_k_negative: Number of top negative examples per feature
    """
    explanations_dir = results_dir / "explanations"
    if not explanations_dir.exists():
        print(f"Error: explanations directory not found: {explanations_dir}")
        return
    
    # Find all explanation files
    explanation_files = list(explanations_dir.glob("*.txt"))
    
    if not explanation_files:
        print(f"No explanation files found in {explanations_dir}")
        return
    
    print(f"Found {len(explanation_files)} explanation files")
    
    # Extract feature IDs from filenames
    feature_ids = []
    for file in explanation_files:
        filename = file.stem
        if "latent" in filename:
            # Format: layers.19_latent123.txt or backbone.layers.19_latent123.txt
            parts = filename.split("_")
            if len(parts) >= 2:
                feature_part = parts[-1]  # latent123
                try:
                    feature_id = int(feature_part.replace("latent", ""))
                    feature_ids.append(feature_id)
                except ValueError:
                    continue
    
    print(f"Extracting sentences for {len(feature_ids)} features...")
    
    for i, feature_id in enumerate(feature_ids, 1):
        print(f"\n[{i}/{len(feature_ids)}] Processing feature {feature_id}...")
        try:
            extract_top_sentences_for_feature(
                results_dir=results_dir,
                hookpoint=hookpoint,
                feature_id=feature_id,
                base_model=base_model,
                top_k_positive=top_k_positive,
                top_k_negative=top_k_negative,
            )
        except Exception as e:
            print(f"  Error processing feature {feature_id}: {e}")
        continue
    
    print("\n" + "=" * 80)
    print("Extraction complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Extract top activating and non-activating sentences for features"
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
        default=None,
        help="Specific feature ID to extract (if not provided, extracts all)",
    )
    parser.add_argument(
        "--top_k_positive",
        type=int,
        default=20,
        help="Number of top positive examples to extract (default: 20)",
    )
    parser.add_argument(
        "--top_k_negative",
        type=int,
        default=10,
        help="Number of top negative examples to extract (default: 10)",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.feature_id is not None:
        # Extract for single feature
        extract_top_sentences_for_feature(
            results_dir=results_dir,
            hookpoint=args.hookpoint,
            feature_id=args.feature_id,
            base_model=args.base_model,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
        )
    else:
        # Extract for all features
        extract_all_features(
            results_dir=results_dir,
            hookpoint=args.hookpoint,
            base_model=args.base_model,
            top_k_positive=args.top_k_positive,
            top_k_negative=args.top_k_negative,
        )


if __name__ == "__main__":
    main()

