#!/usr/bin/env python3
"""
Show EXACTLY what the explainer LLM sees for a given feature.
This reconstructs the exact prompt format sent to the LLM.
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer

# Import the actual explainer to get the exact format
import sys
sys.path.insert(0, str(Path(__file__).parent))

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig
from autointerp_full.explainers.np_max_act_explainer import SYSTEM_CONCISE


def show_explainer_input(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    k_max_act: int = 24,
    window: int = 12,
):
    """Show exactly what the explainer LLM sees."""
    
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
    
    print("=" * 100)
    print(f"EXACT INPUT TO EXPLAINER LLM FOR FEATURE {feature_id}")
    print("=" * 100)
    print()
    
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
            if hasattr(r.latent, 'latent_index'):
                if r.latent.latent_index == feature_id:
                    record = r
                    break
            else:
                try:
                    if int(r.latent) == feature_id:
                        record = r
                        break
                except:
                    continue
        
        if not record:
            print("ERROR: Could not find record for feature", feature_id)
            return
        
        # Get examples (same as explainer uses)
        examples_to_use = record.train if hasattr(record, "train") and record.train else record.examples
        
        if not examples_to_use:
            print("ERROR: No examples found")
            return
        
        # Sort by max activation (same as explainer)
        sorted_examples = sorted(
            examples_to_use, key=lambda e: e.max_activation, reverse=True
        )[:k_max_act]
        
        print(f"Loaded {len(sorted_examples)} examples (top {k_max_act} by max activation)")
        print()
        
        # Build max_act_examples exactly as explainer does
        max_act_examples = []
        
        for i, example in enumerate(sorted_examples, 1):
            # Find the index of max activation
            max_act_idx = int(example.activations.argmax().item())
            max_activation = float(example.activations[max_act_idx].item())
            
            # Get tokens and string tokens
            tokens = example.tokens
            str_tokens = example.str_tokens or [
                tokenizer.decode([t]) for t in tokens.tolist()
            ]
            
            # Extract context window (EXACT same logic as explainer)
            start_idx = max(0, max_act_idx - window)
            end_idx = min(len(tokens), max_act_idx + window + 1)
            
            # THIS IS THE KEY: tokens are joined WITHOUT SPACES
            left_context = "".join(str_tokens[start_idx:max_act_idx])
            right_context = "".join(str_tokens[max_act_idx + 1 : end_idx])
            current_token = str_tokens[max_act_idx] if max_act_idx < len(str_tokens) else ""
            
            # Get next token if available
            next_token = None
            if max_act_idx + 1 < len(str_tokens):
                next_token = str_tokens[max_act_idx + 1]
            
            example_dict = {
                "token": current_token,
                "left_context": left_context,
                "right_context": right_context,
                "activation": max_activation,
            }
            
            if next_token:
                example_dict["next_token"] = next_token
            
            max_act_examples.append(example_dict)
            
            # Show first few examples in detail
            if i <= 5:
                print(f"\n{'='*100}")
                print(f"EXAMPLE {i} (max_activation={max_activation:.4f})")
                print(f"{'='*100}")
                print(f"Max activation at token index: {max_act_idx}")
                print(f"Window: [{start_idx}:{max_act_idx}] + [{max_act_idx}] + [{max_act_idx+1}:{end_idx}]")
                print()
                print(f"LEFT_CONTEXT ({len(str_tokens[start_idx:max_act_idx])} tokens):")
                print(f"  Raw tokens: {str_tokens[start_idx:max_act_idx]}")
                print(f"  Joined (NO SPACES): {repr(left_context)}")
                print()
                print(f"HIGHLIGHTED TOKEN:")
                print(f"  {repr(current_token)}")
                print()
                print(f"RIGHT_CONTEXT ({len(str_tokens[max_act_idx+1:end_idx])} tokens):")
                print(f"  Raw tokens: {str_tokens[max_act_idx+1:end_idx]}")
                print(f"  Joined (NO SPACES): {repr(right_context)}")
                print()
                print(f"FULL CONTEXT (for comparison - what LLM DOESN'T see):")
                full_context = " ".join(str_tokens[start_idx:end_idx])
                print(f"  {full_context}")
                print()
        
        # Build the prompt data exactly as explainer does
        prompt_data = {
            "feature_id": f"latent_{feature_id}",
            "max_act_examples": max_act_examples,
        }
        
        print("\n" + "=" * 100)
        print("SYSTEM PROMPT (sent first)")
        print("=" * 100)
        print(SYSTEM_CONCISE)
        print()
        
        print("=" * 100)
        print("USER PROMPT (JSON - sent second)")
        print("=" * 100)
        print(json.dumps(prompt_data, indent=2))
        print()
        
        print("=" * 100)
        print("KEY ISSUES IDENTIFIED")
        print("=" * 100)
        print("""
1. TOKENS JOINED WITHOUT SPACES:
   - left_context and right_context use "".join() - NO SPACES
   - Makes word boundaries unclear
   - Example: "MunicipalIncomeFund" instead of "Municipal Income Fund"

2. ONLY ONE TOKEN HIGHLIGHTED:
   - Only the max-activation token is shown in "token" field
   - Other activating tokens in context are NOT marked
   - LLM can't see which other tokens also activated

3. CONTEXT IS SPLIT:
   - left_context (12 tokens) + token + right_context (12 tokens)
   - Breaks semantic coherence of phrases/sentences
   - LLM sees fragments, not complete thoughts

4. SMALL WINDOW:
   - Only ±12 tokens around max activation
   - May miss broader semantic patterns
   - Full examples are 32 tokens, but explainer only sees ±12

5. SPECIAL TOKENS VISIBLE:
   - Special tokens like <<SPECIAL_12>>> appear in text
   - May confuse the LLM about what's actual content
        """)
        
        # Show comparison: what it SHOULD see vs what it DOES see
        print("=" * 100)
        print("COMPARISON: What LLM SHOULD See vs What It DOES See")
        print("=" * 100)
        print()
        
        if sorted_examples:
            example = sorted_examples[0]
            max_act_idx = int(example.activations.argmax().item())
            tokens = example.tokens
            str_tokens = example.str_tokens or [tokenizer.decode([t]) for t in tokens.tolist()]
            
            # What it DOES see (current format)
            start_idx = max(0, max_act_idx - window)
            end_idx = min(len(tokens), max_act_idx + window + 1)
            left_context = "".join(str_tokens[start_idx:max_act_idx])
            right_context = "".join(str_tokens[max_act_idx + 1 : end_idx])
            current_token = str_tokens[max_act_idx]
            
            print("WHAT IT DOES SEE (current format):")
            print(f"  left_context: {repr(left_context)}")
            print(f"  token: {repr(current_token)}")
            print(f"  right_context: {repr(right_context)}")
            print()
            
            # What it SHOULD see (better format)
            full_window = " ".join(str_tokens[start_idx:end_idx])
            # Mark all activating tokens
            threshold = example.activations.max().item() * 0.1  # 10% of max
            marked_tokens = []
            for j, (tok, act) in enumerate(zip(str_tokens[start_idx:end_idx], 
                                                 example.activations[start_idx:end_idx])):
                if act.item() > threshold:
                    marked_tokens.append(f"<<{tok}>>")
                else:
                    marked_tokens.append(tok)
            marked_window = " ".join(marked_tokens)
            
            print("WHAT IT SHOULD SEE (better format):")
            print(f"  Full window with spaces: {full_window}")
            print(f"  With activating tokens marked: {marked_window}")
            print()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Show exactly what explainer LLM sees")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--feature-id", type=int, required=True, help="Feature ID")
    parser.add_argument("--k-max-act", type=int, default=24, help="Number of examples (default: 24)")
    parser.add_argument("--window", type=int, default=12, help="Context window size (default: 12)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).resolve()
    latents_dir = results_dir / "latents"
    
    # Auto-detect hookpoint and model
    from analyze_feature_activations import find_hookpoint_in_latents, get_model_from_config
    
    hookpoint = find_hookpoint_in_latents(latents_dir)
    base_model = get_model_from_config(latents_dir, hookpoint)
    
    if base_model is None:
        raise ValueError("Could not auto-detect model")
    
    print(f"Analyzing feature {args.feature_id}")
    print(f"Hookpoint: {hookpoint}")
    print(f"Model: {base_model}")
    print()
    
    show_explainer_input(
        results_dir, hookpoint, args.feature_id, base_model,
        k_max_act=args.k_max_act, window=args.window
    )


if __name__ == "__main__":
    main()



