# Reasoning Domain Feature Search

Reasoning-specific feature discovery using activation-separation-based scoring.

## Files

- **config.yaml**: Reasoning domain configuration
- **tokens.txt**: Reasoning keywords (therefore, however, because, etc.)
- **run_search.py**: Main script to run reasoning feature search

## Quick Start

```bash
python run_search.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>
```

## Results

All results are saved in `results/`:
- `feature_scores.pt`: All feature scores
- `top_features.pt`: Top feature indices
- `top_features_scores.json`: Top features with scores

## Configuration

Edit `config.yaml` to customize dataset, scoring method, and parameters.

