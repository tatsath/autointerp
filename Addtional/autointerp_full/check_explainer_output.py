#!/usr/bin/env python3
"""
Check what the explainer is ACTUALLY sending to the LLM for a specific feature.
"""

import sys
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig
from autointerp_full.explainers.np_max_act_explainer import NPMaxActExplainer

def check_explainer_output(results_dir: Path, feature_id: int):
    """Check what the explainer actually sends."""
    
    # Load run config
    run_config_path = results_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    base_model = run_config['model']
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    hookpoint = 'backbone.layers.28'
    
    # Create configs
    sampler_cfg = SamplerConfig(**run_config['sampler_cfg'])
    constructor_cfg = ConstructorConfig(**run_config['constructor_cfg'])
    
    # Load examples
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    latents_dir = results_dir / 'latents'
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
    
    if not record:
        print(f"ERROR: Could not find record for feature {feature_id}")
        return
    
    # Get examples (same as explainer uses)
    examples_to_use = record.train if hasattr(record, "train") and record.train else record.examples
    
    if not examples_to_use:
        print(f"ERROR: No examples found for feature {feature_id}")
        return
    
    print(f"Loaded {len(examples_to_use)} examples for feature {feature_id}")
    print()
    
    # Create explainer instance (we only need _build_prompt, so we can use a dummy client)
    class DummyClient:
        pass
    
    explainer = NPMaxActExplainer(
        client=DummyClient(),
        tokenizer=tokenizer,
        k_max_act=24,
        window=12,  # This parameter should be ignored in new code
    )
    
    # Build prompt using the actual explainer
    prompt = explainer._build_prompt(examples_to_use)
    
    print("=" * 100)
    print(f"ACTUAL PROMPT SENT BY EXPLAINER FOR FEATURE {feature_id}")
    print("=" * 100)
    print()
    
    print("SYSTEM PROMPT:")
    print("-" * 100)
    from autointerp_full.explainers.np_max_act_explainer import SYSTEM_CONCISE
    print(SYSTEM_CONCISE)
    print()
    
    print("USER PROMPT (First 3 examples):")
    print("-" * 100)
    user_prompt = {
        'feature_id': f'latent_{feature_id}',
        'max_act_examples': prompt[:3]  # First 3 examples
    }
    print(json.dumps(user_prompt, indent=2))
    print()
    
    # Show detailed breakdown of first example
    if prompt:
        first_example = prompt[0]
        print("=" * 100)
        print("DETAILED BREAKDOWN OF FIRST EXAMPLE")
        print("=" * 100)
        print()
        print(f"Keys in example dict: {list(first_example.keys())}")
        print()
        
        if 'text' in first_example:
            print("✅ NEW FORMAT DETECTED:")
            print(f"  - 'text' field: {first_example['text'][:200]}...")
            print(f"  - 'activating_tokens' count: {len(first_example.get('activating_tokens', []))}")
            if first_example.get('activating_tokens'):
                print(f"  - First 5 activating tokens:")
                for tok_info in first_example['activating_tokens'][:5]:
                    print(f"      {tok_info['token']}: {tok_info['activation']:.4f}")
            print(f"  - 'max_activation': {first_example.get('max_activation')}")
        elif 'token' in first_example:
            print("❌ OLD FORMAT DETECTED:")
            print(f"  - 'token': {first_example.get('token')}")
            print(f"  - 'left_context': {first_example.get('left_context', '')[:100]}...")
            print(f"  - 'right_context': {first_example.get('right_context', '')[:100]}...")
            print(f"  - 'activation': {first_example.get('activation')}")
        print()
        
        # Show the actual text that LLM sees
        if 'text' in first_example:
            print("TEXT THAT LLM SEES (first example):")
            print("-" * 100)
            print(first_example['text'])
            print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str)
    parser.add_argument("--feature-id", type=int, required=True)
    args = parser.parse_args()
    
    check_explainer_output(Path(args.results_dir), args.feature_id)

