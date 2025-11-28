# 2. AutoInterp Lite - Basic Labeling

This module provides basic labeling functionality by collecting positive/negative context windows and generating labels using LLM.

## Pipeline

1. **collect_examples.py** - Collects token-level context windows that activate each feature
2. **label_features.py** - Generates human-readable labels using vLLM API

## Quick Start

```bash
# Run complete labeling pipeline
python run_labeling.py \
    --search_output ../results/1_search \
    --output_dir ../results/2_labeling_lite

# Or run steps individually
python collect_examples.py  # Collects examples
python label_features.py    # Generates labels
```

## Prerequisites

- Search results must exist in `../results/1_search/`
- vLLM server must be running (for label_features.py)
  - Default: `http://localhost:8002/v1`
  - Model: `Qwen/Qwen2.5-72B-Instruct`

## Output

Results are saved to `../results/2_labeling_lite/`:
- `activating_sentences.json` - Positive/negative context windows for each feature
- `feature_labels.json` - Generated labels with granularity and explanations

## Configuration

- `finance_vocab.txt` - Finance vocabulary for coverage calculation
- Paths are automatically resolved relative to search output

## Files

- `collect_examples.py` - Collects activating context windows
- `label_features.py` - Generates labels using vLLM API
- `run_labeling.py` - Unified entry point (chains both steps)

