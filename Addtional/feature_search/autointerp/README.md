# AutoInterp: Feature Interpretation

Extract activating examples and generate human-readable labels for discovered features.

## Files

- **compute_examples_finance.py**: Extract top activating examples for features
- **generate_labels_from_examples.py**: Generate labels from extracted examples
- **compute_examples_finance.sh**: Complete pipeline script

## Quick Start

```bash
# Extract examples
python compute_examples_finance.py \
    --model_path <model> \
    --sae_path <sae_path> \
    --sae_id <sae_id> \
    --dataset_path <dataset> \
    --feature_list_path ../domain_finance/results/top_features_scores.json \
    --output_path result/finance_examples.jsonl

# Generate labels
python generate_labels_from_examples.py \
    --examples_jsonl_path result/finance_examples.jsonl \
    --output_path result/finance_labels.json
```

## Results

All results are saved in `result/`:
- `finance_examples.jsonl`: Activating examples per feature
- `finance_labels.json`: Generated feature labels

## Pipeline

1. Extract top activating examples for each feature
2. Generate human-readable labels from examples
3. Use labels for downstream analysis
