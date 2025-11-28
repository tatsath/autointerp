# autointerp_advanced

A standalone package for extracting feature examples and generating human-readable labels from Sparse Autoencoder (SAE) features.

## How to Run

### Bash Script (Recommended)

```bash
# Simplest usage - runs with default feature numbers
bash run_llama_labeling.sh

# With custom feature numbers (comma-separated)
bash run_llama_labeling.sh "11,313,251,165,28"

# With feature numbers and custom SAE path
bash run_llama_labeling.sh "11,313,251" /path/to/sae blocks.19.hook_resid_post

# Or with a JSON file path (still supported)
bash run_llama_labeling.sh path/to/top_features_scores.json
```

**Optional arguments (all have defaults):**
1. `feature_numbers` (optional): Comma-separated feature indices (e.g., "11,313,251") or path to JSON file (default: "11,313,251,165,28,375,53,379,297,207,132,32,181,166,350,215,249,376,56,381")
2. `sae_path` (optional): Path to SAE model directory (default: `/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU`)
3. `sae_id` (optional): SAE hook point (default: "blocks.19.hook_resid_post")

**Optional parameters (set as environment variables or edit script):**
- `MODEL_PATH`: Base model path (default: "meta-llama/Llama-3.1-8B-Instruct")
- `DATASET_PATH`: Dataset path (default: "jyanimaulik/yahoo_finance_stockmarket_news")
- `OUTPUT_DIR`: Output directory (default: "results")
- `N_SAMPLES`: Number of samples to process (default: 1000)
- `MAX_EXAMPLES`: Max examples per feature (default: 20)

### Python Function

```python
from main.run_labeling import run_labeling

# With feature numbers directly (recommended)
run_labeling(
    feature_indices=[11, 313, 251, 165, 28],
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    sae_path="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU",
    dataset_path="jyanimaulik/yahoo_finance_stockmarket_news",
    sae_id="blocks.19.hook_resid_post"
)

# Or with JSON file path (still supported)
run_labeling(
    feature_indices="path/to/top_features_scores.json",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
    sae_path="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU",
    dataset_path="jyanimaulik/yahoo_finance_stockmarket_news",
    sae_id="blocks.19.hook_resid_post"
)
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. **Activate conda environment:**
   ```bash
   conda activate sae
   ```

2. **Install required packages:**
   ```bash
   pip install sae-lens sae-dashboard transformer-lens transformers datasets torch
   ```

3. **Verify installation:**
   ```bash
   python -c "import sae_lens; import sae_dashboard; print('Installation successful')"
   ```

## Input Format

You can provide features in two ways:

**Option 1: Direct feature numbers (recommended)**
```bash
bash run_llama_labeling.sh "11,313,251,165,28"
```

**Option 2: JSON file (optional)**
If you have a JSON file with feature indices and scores:
```json
{
  "feature_indices": [6, 7, 11, 313, 251],
  "scores": [0.335, 0.216, 0.049, 0.032, 0.028]
}
```
You can pass the file path: `bash run_llama_labeling.sh path/to/features.json`

## Parameters

### Required Parameters

- **`feature_indices`** (list or str): List of feature indices (integers) or path to JSON file with `feature_indices` and `scores`
- **`model_path`** (str): Path to the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
- **`sae_path`** (str): Path to SAE model directory
- **`dataset_path`** (str): Path to dataset (e.g., "jyanimaulik/yahoo_finance_stockmarket_news")
- **`sae_id`** (str): SAE hook point identifier (e.g., "blocks.19.hook_resid_post")

### Optional Parameters

- **`output_dir`** (str, default: "results"): Directory to save outputs
- **`n_samples`** (int, default: 1000): Number of dataset samples to process
  - *Lower values = faster but less comprehensive examples*
  - *Higher values = more examples but slower processing*
- **`max_examples_per_feature`** (int, default: 20): Max examples to extract per feature
  - *Controls how many top activating examples are saved*
- **`device`** (str, default: auto-detect): Device to use (e.g., "cuda:0", "cuda:1")
- **`column_name`** (str, default: "text"): Dataset column name containing text
- **`minibatch_size_features`** (int, default: 256): Batch size for feature processing
  - *Increase if you have more GPU memory*
- **`minibatch_size_tokens`** (int, default: 64): Batch size for token processing
  - *Decrease if you run out of memory*
- **`label_model_path`** (str, default: same as model_path): Model to use for label generation
- **`max_examples_for_labeling`** (int, default: 10): Max examples to use per feature in label prompt
  - *More examples = better labels but slower generation*

## Output Format

### feature_examples.jsonl

Each line contains a JSON object with feature examples:

```json
{
  "feature_id": 6,
  "finance_score": 0.33539581298828125,
  "examples": [
    {
      "activation": 2.7747,
      "sequence_index": 455,
      "token_position": 944,
      "text": " truckload posting an 8% increase and LTL volume growth of 29%. On the earnings call"
    }
  ]
}
```

### feature_labels.json

Contains generated labels for all features:

```json
{
  "num_features": 20,
  "features": [
    {
      "feature_index": 6,
      "score": 0.33539581298828125,
      "label_llm": "Stock Market Sentiment Analysis Feature",
      "label_search": "Market market trading share",
      "label": "Stock Market Sentiment Analysis Feature",
      "num_examples": 38
    }
  ]
}
```

## Labeling Approaches

The package generates labels using two approaches:

1. **LLM-based**: Uses a language model to analyze activating examples and generate descriptive labels (less than 10 words)
2. **Search-based**: Extracts common keywords/patterns from examples to create labels

Both labels are included in the output JSON file.

## Notes

- The package automatically handles local SAE loading with proper configuration parsing
- All outputs are saved to the `results/` directory by default
- The pipeline runs in the foreground to allow easy error checking and debugging
- Labels are limited to less than 10 words for conciseness
- To create scripts for other models, copy `run_llama_labeling.sh` and modify the default values

## Folder Structure

```
autointerp_advanced/
├── main/                        # Core scripts
│   ├── __init__.py
│   ├── extract_examples.py     # Extract examples from SaeVisData
│   ├── generate_labels.py      # Generate labels (LLM + search-based)
│   └── run_labeling.py         # Unified runner function
├── run_llama_labeling.sh       # Sample script for Llama model
├── README.md                    # This file
└── results/                     # Output directory (created automatically)
    ├── feature_examples.jsonl   # Extracted examples
    └── feature_labels.json      # Generated labels
```
