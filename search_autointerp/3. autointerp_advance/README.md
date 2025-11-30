# 3. AutoInterp Advance - Advanced Labeling

This module provides advanced labeling functionality using SaeVisRunner to extract examples and generate labels.

## Pipeline

1. **extract_examples.py** - Extracts top activating examples using SaeVisRunner
2. **generate_labels.py** - Generates labels from extracted examples using LLM

## Quick Start

```bash
# Run complete advanced labeling pipeline
python run_labeling_advanced.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --sae_id blocks.19.hook_resid_post \
    --search_output ../results/1_search \
    --output_dir ../results/3_labeling_advance \
    --n_samples 5000 \
    --max_examples_per_feature 20
```

## Prerequisites

- Search results must exist in `../results/1_search/`
- Model and SAE paths must be accessible

## Output

Results are saved to `../results/3_labeling_advance/`:
- `feature_examples.jsonl` - Extracted examples in JSONL format
- `feature_labels.json` - Generated labels with LLM and search-based methods

## Files

- `extract_examples.py` - Extracts examples using SaeVisRunner
- `extract_examples_simple.py` - Simplified version
- `generate_labels.py` - Generates labels from examples
- `compute_dashboard.py` - HTML dashboard generation (optional)
- `run_labeling_advanced.py` - Unified entry point (chains extract + label)

## Differences from Lite

- Uses SaeVisRunner for more accurate example extraction
- Processes full sequences, not just context windows
- Can generate both LLM-based and search-based labels



