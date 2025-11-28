#!/usr/bin/env python3
"""Quick test script to run the labeling pipeline"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.run_labeling import run_labeling

# Default values
FEATURE_LIST_PATH = "/home/nvidia/Documents/Hariom/autointerp/feature_search/test_results/feature_list.json"
SAE_PATH = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
SAE_ID = "blocks.19.hook_resid_post"
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "jyanimaulik/yahoo_finance_stockmarket_news"
OUTPUT_DIR = "results"
N_SAMPLES = int(os.environ.get("N_SAMPLES", 50))  # Small for quick test
MAX_EXAMPLES = int(os.environ.get("MAX_EXAMPLES", 5))  # Small for quick test

print("=" * 80)
print("Running Feature Labeling Pipeline (Test Mode)")
print("=" * 80)
print(f"Feature List: {FEATURE_LIST_PATH}")
print(f"SAE Path: {SAE_PATH}")
print(f"SAE ID: {SAE_ID}")
print(f"Model: {MODEL_PATH}")
print(f"Dataset: {DATASET_PATH}")
print(f"N Samples: {N_SAMPLES}")
print(f"Max Examples: {MAX_EXAMPLES}")
print("=" * 80)
print()

try:
    run_labeling(
        feature_list_path=FEATURE_LIST_PATH,
        model_path=MODEL_PATH,
        sae_path=SAE_PATH,
        dataset_path=DATASET_PATH,
        sae_id=SAE_ID,
        output_dir=OUTPUT_DIR,
        n_samples=N_SAMPLES,
        max_examples_per_feature=MAX_EXAMPLES
    )
    print("\n" + "=" * 80)
    print("✅ Pipeline completed successfully!")
    print("=" * 80)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

