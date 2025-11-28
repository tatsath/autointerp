# Finance Feature Examples Extraction and Labeling

This directory contains scripts to extract top activating examples for finance features and generate human-readable labels, following the SAE-Reasoning paper approach but adapted for finance features.

## Pipeline Overview

1. **Extract Examples** (`compute_examples_finance.py`): Extract top activating examples using SaeVisRunner
2. **Generate Labels** (`generate_labels_from_examples.py`): Generate human-readable labels from extracted examples

## Scripts

### `compute_examples_finance.py`

Extracts top activating examples for features from a feature list JSON file. Uses `SaeVisRunner` from `sae_dashboard` to collect feature activations and examples, then outputs them as JSONL instead of HTML dashboards.

**Usage:**

```bash
python compute_examples_finance.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --sae_path "/path/to/sae" \
    --sae_id "blocks.19.hook_resid_post" \
    --dataset_path "jyanimaulik/yahoo_finance_stockmarket_news" \
    --feature_list_path "../domains/finance/scores/top_features_scores.json" \
    --output_path "../domains/finance/scores/feature_examples_finance.jsonl" \
    --n_samples 5000 \
    --max_examples_per_feature 20
```

**Parameters:**
- `model_path`: Path to model (HuggingFace repo or local)
- `sae_path`: Path to SAE (HuggingFace repo or local directory)
- `sae_id`: SAE identifier (e.g., "blocks.19.hook_resid_post")
- `dataset_path`: Path to dataset (HuggingFace repo)
- `feature_list_path`: Path to JSON file with feature indices and scores
- `output_path`: Path to output JSONL file
- `column_name`: Dataset column name containing text (default: "text")
- `minibatch_size_features`: Batch size for feature processing (default: 256)
- `minibatch_size_tokens`: Batch size for token processing (default: 64)
- `n_samples`: Number of samples to process (default: 5000)
- `max_examples_per_feature`: Maximum examples to extract per feature (default: 20)

### `compute_examples_finance.sh`

Convenience shell script wrapper with default configuration.

**Usage:**

```bash
./compute_examples_finance.sh
```

Or with environment variables:

```bash
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct" \
SAE_PATH="/path/to/sae" \
FEATURE_LIST_PATH="../domains/finance/scores/top_features_scores.json" \
./compute_examples_finance.sh
```

### `generate_labels_from_examples.py`

Generates human-readable labels for features based on their extracted examples. Uses an LLM to analyze the activating examples and produce concise labels.

**Usage:**

```bash
python generate_labels_from_examples.py \
    --examples_jsonl_path "../domains/finance/scores/feature_examples_finance.jsonl" \
    --output_path "../domains/finance/scores/feature_labels.json" \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --max_examples_per_feature 10
```

**Parameters:**
- `examples_jsonl_path`: Path to JSONL file from `compute_examples_finance.py`
- `output_path`: Path to save labeled features JSON file
- `model_path`: Model to use for label generation (default: "meta-llama/Llama-3.1-8B-Instruct")
- `max_examples_per_feature`: Max examples to use per feature in prompt (default: 10)

## Output Formats

### Examples JSONL Format

The `compute_examples_finance.py` script outputs a JSONL file where each line contains:

```json
{
  "feature_id": 11,
  "finance_score": 0.014546379446983337,
  "examples": [
    {
      "text": "million. Apply that to 10,000 locations and you have $35 billion in annual sales compared to $10 billion today. Of course, there are many years of growth needed to hit these sales figures, but Chiptole has a clear line of sight to achieving these growth goals.Can profit margins keep moving higher?",
      "activation": 5.296875,
      "token_position": 42,
      "sequence_index": 0
    },
    {
      "text": "was a top AI stock to own even before AI was a big buzzword in stocks. The company has long used AI to optimize its e-commerce operations and shorten its delivery times.",
      "activation": 5.25,
      "token_position": 15,
      "sequence_index": 1
    }
  ]
}
```

**Saved to:** `../domains/finance/scores/feature_examples_finance.jsonl` (or path specified in `--output_path`)

### Labels JSON Format

The `generate_labels_from_examples.py` script outputs a JSON file with labels:

```json
{
  "num_features": 21,
  "features": [
    {
      "feature_index": 7,
      "score": 0.002171800471842289,
      "label": "Financial News Context",
      "num_examples": 20
    },
    {
      "feature_index": 11,
      "score": 0.014546379446983337,
      "label": "Financial News Articles",
      "num_examples": 20
    },
    {
      "feature_index": 100,
      "score": 0.0016489948611706495,
      "label": "Market News Sentiment",
      "num_examples": 20
    },
    {
      "feature_index": 313,
      "score": 0.004922049585729837,
      "label": "Electric Vehicle News",
      "num_examples": 20
    }
  ]
}
```

**Saved to:** `../domains/finance/scores/feature_labels.json` (or path specified in `--output_path`)

## Complete Pipeline Example

```bash
# Step 1: Extract examples
python compute_examples_finance.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --sae_path "/path/to/sae" \
    --sae_id "blocks.19.hook_resid_post" \
    --dataset_path "jyanimaulik/yahoo_finance_stockmarket_news" \
    --feature_list_path "../domains/finance/scores/top_features_scores.json" \
    --output_path "../domains/finance/scores/feature_examples_finance.jsonl" \
    --n_samples 5000 \
    --max_examples_per_feature 20

# Step 2: Generate labels from examples
python generate_labels_from_examples.py \
    --examples_jsonl_path "../domains/finance/scores/feature_examples_finance.jsonl" \
    --output_path "../domains/finance/scores/feature_labels.json" \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --max_examples_per_feature 10
```

## Notes

- The scripts handle both local SAE paths and HuggingFace repos
- Uses introspection to handle different versions of `sae_dashboard`
- Follows the same pipeline as the SAE-Reasoning paper but outputs JSONL instead of HTML
- Labels are generated using LLM analysis of activating examples
- Example labels include: "Financial News Context", "Market News Sentiment", "Electric Vehicle News", "Stock Market Sentiment", etc.

