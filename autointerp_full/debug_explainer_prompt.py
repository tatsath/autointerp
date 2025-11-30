#!/usr/bin/env python3
"""Debug script to see what prompt is actually being sent to LLM"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from autointerp_full.latents.latents import LatentDataset
from autointerp_full.explainers.np_max_act_explainer import NPMaxActExplainer
from autointerp_full.explainers.default.prompt_loader import set_prompt_override

# Enable prompt override
set_prompt_override(True, "prompts_finance.yaml")

# Load a feature to test
results_dir = Path("results/llama31_8b_top10_finance_features")
if not results_dir.exists():
    print(f"Results directory not found: {results_dir}")
    sys.exit(1)

# Load dataset
dataset = LatentDataset(
    results_dir,
    hookpoint="layers.19",
    feature_nums=[215],  # Test with one feature
)

if len(dataset) == 0:
    print("No features found in dataset")
    sys.exit(1)

record = dataset[0]
print(f"Testing with feature: {record.latent}")
print(f"Number of examples: {len(record.examples)}")
print(f"Number of train examples: {len(record.train) if record.train else 0}")

# Create a dummy client to test prompt building
class DummyClient:
    pass

# Create explainer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

explainer = NPMaxActExplainer(
    DummyClient(),
    tokenizer=tokenizer,
    k_max_act=50,
    window=3,
    verbose=True,
)

# Build prompt
examples_to_use = record.train if record.train else record.examples
messages = explainer._build_prompt(examples_to_use)

print("\n" + "=" * 80)
print("SYSTEM PROMPT (first 1000 chars):")
print("=" * 80)
print(messages[0]['content'][:1000])

print("\n" + "=" * 80)
print("USER PROMPT (JSON structure):")
print("=" * 80)
user_content = messages[1]['content']
try:
    prompt_data = json.loads(user_content)
    print(f"Feature ID: {prompt_data.get('feature_id', 'N/A')}")
    print(f"Number of examples: {len(prompt_data.get('max_act_examples', []))}")
    if prompt_data.get('max_act_examples'):
        print(f"\nFirst example keys: {list(prompt_data['max_act_examples'][0].keys())}")
        print(f"\nFirst example text (first 300 chars):")
        print(prompt_data['max_act_examples'][0].get('text', '')[:300])
        print(f"\nFirst example activating_tokens (first 5):")
        print(prompt_data['max_act_examples'][0].get('activating_tokens', [])[:5])
except json.JSONDecodeError:
    print("User content is not valid JSON:")
    print(user_content[:500])

print("\n" + "=" * 80)
print("FULL USER PROMPT LENGTH:")
print("=" * 80)
print(f"Total length: {len(user_content)} characters")
print(f"Estimated tokens: ~{len(user_content) // 4}")



