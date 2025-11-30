#!/usr/bin/env python3
"""
Deep analysis of why explainer chose a specific explanation.
Analyzes all examples, patterns, and activation values to understand the logic.
"""

import json
import torch
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig
from analyze_feature_activations import (
    find_hookpoint_in_latents,
    get_model_from_config,
    load_explanation,
)


def decode_example_properly(example, tokenizer):
    """Properly decode an example to readable text."""
    if hasattr(example, 'str_tokens') and example.str_tokens:
        return " ".join(example.str_tokens)
    elif hasattr(example, 'tokens'):
        tokens = example.tokens
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return tokenizer.decode(tokens, skip_special_tokens=False)
    elif hasattr(example, 'text'):
        return example.text
    else:
        return "Could not decode"


def analyze_explanation_logic(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    output_file: Path,
):
    """Deep analysis of why explainer chose the explanation."""
    
    latents_dir = results_dir / "latents"
    
    # Load run config
    run_config_path = results_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    sampler_cfg = SamplerConfig(**run_config["sampler_cfg"])
    constructor_cfg = ConstructorConfig(**run_config["constructor_cfg"])
    
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    
    explanation = load_explanation(results_dir, hookpoint, feature_id)
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"DEEP ANALYSIS: Why Feature {feature_id} Explanation\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"EXPLANATION: {explanation}\n\n")
        
        try:
            dataset = LatentDataset(
                raw_dir=str(latents_dir),
                modules=[hookpoint],
                sampler_cfg=sampler_cfg,
                constructor_cfg=constructor_cfg,
                latents=latents_dict,
                tokenizer=tokenizer,
            )
            
            record = None
            for r in dataset:
                if hasattr(r.latent, 'latent_index') and r.latent.latent_index == feature_id:
                    record = r
                    break
            
            if not record:
                f.write("Could not load record from dataset.\n")
                return
            
            # Get ALL examples (not just top 15)
            train_examples = record.train if hasattr(record, "train") and record.train else []
            examples = record.examples if hasattr(record, "examples") and record.examples else []
            all_examples = train_examples if train_examples else examples
            
            f.write("=" * 80 + "\n")
            f.write("ALL EXAMPLES WITH ACTIVATIONS (Sorted by Max Activation)\n")
            f.write("=" * 80 + "\n\n")
            
            # Sort by max activation
            sorted_examples = sorted(
                all_examples,
                key=lambda x: x.max_activation if hasattr(x, 'max_activation') else 0.0,
                reverse=True
            )
            
            # Analyze patterns
            all_words = []
            high_activation_words = defaultdict(list)  # word -> list of (activation, example_idx)
            
            for idx, ex in enumerate(sorted_examples[:25]):  # Top 25
                max_act = ex.max_activation if hasattr(ex, 'max_activation') else 0.0
                text = decode_example_properly(ex, tokenizer)
                
                f.write(f"\nExample {idx+1} (max_activation={max_act:.4f}):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{text}\n")
                
                # Get activating tokens
                if hasattr(ex, 'activations') and hasattr(ex, 'str_tokens'):
                    activations = ex.activations
                    str_tokens = ex.str_tokens
                    
                    # Find threshold
                    if len(activations) > 0:
                        max_act_val = activations.max().item() if hasattr(activations.max(), 'item') else float(activations.max())
                        threshold = max(0.01 * max_act_val, 0.1)  # 1% of max or 0.1
                    else:
                        threshold = 0.0
                    
                    activating_tokens = []
                    for i, (token, act) in enumerate(zip(str_tokens, activations)):
                        act_val = act.item() if hasattr(act, 'item') else float(act)
                        if act_val > threshold:
                            activating_tokens.append((token, act_val))
                            # Store for pattern analysis
                            clean_token = token.strip().lower()
                            if any(c.isalpha() for c in clean_token):
                                high_activation_words[clean_token].append((act_val, idx))
                    
                    if activating_tokens:
                        f.write(f"\nActivating tokens (threshold={threshold:.4f}):\n")
                        for token, act_val in activating_tokens[:20]:  # Top 20 per example
                            f.write(f"  <<{token}>> (act={act_val:.4f})\n")
                
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PATTERN ANALYSIS: What Words Appear Across Multiple Examples?\n")
            f.write("=" * 80 + "\n\n")
            
            # Find words that appear in multiple high-activation examples
            common_words = {}
            for word, occurrences in high_activation_words.items():
                if len(occurrences) >= 3:  # Appears in at least 3 examples
                    avg_act = sum(act for act, _ in occurrences) / len(occurrences)
                    max_act = max(act for act, _ in occurrences)
                    common_words[word] = {
                        'count': len(occurrences),
                        'avg_activation': avg_act,
                        'max_activation': max_act,
                        'example_indices': [idx for _, idx in occurrences]
                    }
            
            # Sort by frequency and activation
            sorted_common = sorted(
                common_words.items(),
                key=lambda x: (x[1]['count'], x[1]['avg_activation']),
                reverse=True
            )
            
            f.write("Words appearing in 3+ examples with high activations:\n")
            f.write("-" * 80 + "\n")
            for word, data in sorted_common[:30]:
                f.write(
                    f"{word:20s}  appears_in={data['count']:2d} examples  "
                    f"avg_act={data['avg_activation']:6.2f}  "
                    f"max_act={data['max_activation']:6.2f}  "
                    f"examples={data['example_indices'][:5]}\n"
                )
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("ANALYSIS: Why This Explanation?\n")
            f.write("=" * 80 + "\n\n")
            
            # Check if explanation words appear in examples
            explanation_words = explanation.lower().split()
            f.write(f"Explanation words: {explanation_words}\n\n")
            
            f.write("Checking if explanation words appear in examples:\n")
            for word in explanation_words:
                if word in high_activation_words:
                    occurrences = high_activation_words[word]
                    f.write(f"  '{word}': Found in {len(occurrences)} examples\n")
                    for act_val, ex_idx in occurrences[:5]:
                        f.write(f"    Example {ex_idx+1}: activation={act_val:.4f}\n")
                else:
                    f.write(f"  '{word}': NOT FOUND in high-activation tokens\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("POTENTIAL ISSUES WITH EXPLANATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Check if explanation is based on a single example
            explanation_phrase_lower = explanation.lower()
            matching_examples = []
            for idx, ex in enumerate(sorted_examples[:25]):
                text = decode_example_properly(ex, tokenizer).lower()
                if explanation_phrase_lower in text or any(word in text for word in explanation_words if len(word) > 3):
                    matching_examples.append((idx, ex.max_activation if hasattr(ex, 'max_activation') else 0.0))
            
            f.write(f"Examples containing explanation words:\n")
            for ex_idx, max_act in matching_examples[:10]:
                f.write(f"  Example {ex_idx+1}: max_activation={max_act:.4f}\n")
            
            if len(matching_examples) == 1:
                f.write("\n⚠️  WARNING: Explanation appears to be based on ONLY ONE example!\n")
                f.write("This is problematic - explanations should capture patterns across multiple examples.\n")
            elif len(matching_examples) < 3:
                f.write(f"\n⚠️  WARNING: Explanation appears in only {len(matching_examples)} examples.\n")
                f.write("This might not represent the feature well.\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("WHAT THE EXPLAINER SHOULD HAVE SEEN\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Top 5 examples by activation:\n")
            for idx, ex in enumerate(sorted_examples[:5]):
                max_act = ex.max_activation if hasattr(ex, 'max_activation') else 0.0
                text = decode_example_properly(ex, tokenizer)
                f.write(f"\n{idx+1}. (max_act={max_act:.4f}): {text[:200]}...\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("""
The explainer LLM should:
1. Look at ALL examples (not just one)
2. Find COMMON PATTERNS across multiple examples
3. Abstract to a concept that covers MOST examples

If the explanation only matches 1-2 examples, it might be:
- Overly specific
- Based on a rare pattern
- Not representative of the feature

A better explanation should:
- Appear in at least 5-10 examples
- Capture the semantic pattern across examples
- Not be based on a single high-activation token

\n""")
        
        except Exception as e:
            f.write(f"\nError: {e}\n")
            import traceback
            f.write(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Deep analysis of explanation logic")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--feature-id", type=int, required=True, help="Feature ID")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).resolve()
    latents_dir = results_dir / "latents"
    
    hookpoint = find_hookpoint_in_latents(latents_dir)
    base_model = get_model_from_config(latents_dir, hookpoint)
    
    print(f"Deep analyzing feature {args.feature_id}")
    
    if args.output is None:
        output_file = results_dir / "feature_analysis" / f"feature_{args.feature_id}_deep_analysis.txt"
    else:
        output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    analyze_explanation_logic(
        results_dir, hookpoint, args.feature_id, base_model, output_file
    )
    
    print(f"\n✓ Deep analysis saved to: {output_file}")


if __name__ == "__main__":
    main()



