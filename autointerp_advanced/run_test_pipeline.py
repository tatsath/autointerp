#!/usr/bin/env python3
"""Run the labeling pipeline with first 5 features"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main.run_labeling import run_labeling

print("=" * 80)
print("Running Feature Labeling Pipeline")
print("Features: [11, 313, 251, 165, 28]")
print("=" * 80)
print()

try:
    run_labeling(
        feature_indices=[11, 313, 251, 165, 28],
        model_path='meta-llama/Llama-3.1-8B-Instruct',
        sae_path='/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU',
        dataset_path='jyanimaulik/yahoo_finance_stockmarket_news',
        sae_id='blocks.19.hook_resid_post',
        output_dir='results',
        n_samples=10,  # Small for quick test
        max_examples_per_feature=2  # Small for quick test
    )
    print("\n✅ Pipeline completed successfully!")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

