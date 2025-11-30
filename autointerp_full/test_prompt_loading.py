#!/usr/bin/env python3
"""Test script to verify prompt loading and see what's being sent to LLM"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from autointerp_full.explainers.default.prompt_loader import (
    set_prompt_override, 
    load_prompts_from_yaml,
    get_np_max_act_prompt
)

# Test prompt loading
prompt_file = "prompts_finance.yaml"
print(f"Testing prompt loading from: {prompt_file}")
print("=" * 80)

# Enable prompt override
set_prompt_override(True, prompt_file)

# Load config
config = load_prompts_from_yaml(prompt_file)
print(f"Config loaded: {config is not None}")
if config:
    print(f"Config keys: {list(config.keys())}")
    if 'explainers' in config:
        print(f"Explainers keys: {list(config['explainers'].keys())}")
        if 'np_max_act' in config['explainers']:
            print(f"np_max_act keys: {list(config['explainers']['np_max_act'].keys())}")

# Get the prompt
default_prompt = "You are labeling ONE hidden feature from a language model..."
loaded_prompt = get_np_max_act_prompt('system_concise', default_prompt)

print("\n" + "=" * 80)
print("PROMPT COMPARISON:")
print("=" * 80)
print(f"Default prompt length: {len(default_prompt)}")
print(f"Loaded prompt length: {len(loaded_prompt)}")
print(f"Prompts match: {loaded_prompt == default_prompt}")
print(f"Prompt was loaded from YAML: {loaded_prompt != default_prompt}")

print("\n" + "=" * 80)
print("LOADED PROMPT (first 1000 chars):")
print("=" * 80)
print(loaded_prompt[:1000])

print("\n" + "=" * 80)
print("CHECKING KEY REQUIREMENTS:")
print("=" * 80)
checks = {
    "Contains '6-10 words'": '6-10 words' in loaded_prompt or '6–10 words' in loaded_prompt,
    "Contains 'EVENT | SECTOR'": 'EVENT | SECTOR' in loaded_prompt,
    "Contains 'Do NOT include the word entity'": 'Do NOT include the word "entity"' in loaded_prompt or 'Do NOT include the word entity' in loaded_prompt,
    "Contains 'Corporate spin-offs' example": 'Corporate spin-offs' in loaded_prompt,
    "Contains 'finance-specific'": 'finance-specific' in loaded_prompt or 'domain-tight' in loaded_prompt,
}

for check, result in checks.items():
    status = "✅" if result else "❌"
    print(f"{status} {check}")



