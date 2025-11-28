# Domain Common: Core Functionality

Shared core functions for feature search across all domains.

## Files

- **run_feature_search.py**: Unified command-line interface for feature search
- **compute_score.py**: Core feature scoring implementation
- **compute_dashboard.py**: Interactive HTML dashboard generation

## Usage

### Direct Usage

```bash
python run_feature_search.py \
    --model_path <model> \
    --sae_path <sae_path> \
    --dataset_path <dataset> \
    --tokens_str_path <tokens.json> \
    --output_dir <output_dir> \
    --sae_id <sae_id> \
    --score_type fisher \
    --num_features 100
```

### Imported by Domain Scripts

Domain-specific scripts (e.g., `domain_finance/run_search.py`) import and use these functions with domain-specific configurations.

## Scoring Methods

- **simple**: `|μ⁺ - μ⁻|` - Simple mean difference
- **fisher**: `(μ⁺ - μ⁻)² / (σ⁺ + σ⁻ + ε)` - Variance-normalized (recommended)
- **domain**: Token-based with entropy penalty

## Output

- `feature_scores.pt`: All feature scores (PyTorch tensor)
- `top_features.pt`: Top feature indices (PyTorch tensor)
- `feature_list.json`: Top features with metadata (JSON)
- `dashboards/`: HTML dashboards (if `--generate_dashboard` is used)
