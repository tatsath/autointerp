# Domain Finance: Feature Search

Finance-specific feature discovery using activation-separation-based scoring.

## Structure

- **Root files** (produce scores, domain-specific):
  - `compute_score.py` - Compute domain scores
  - `run_analysis.sh` - Complete analysis pipeline
  - `run_search.py` - Unified search interface
  - `config.yaml`, `config.json`, `config_big.json` - Domain configurations
  - `tokens.txt`, `tokens.json` - Domain tokens

- **main/** (generic, reusable):
  - `add_labels.py` - Add labels to feature results

- **archive/** (old/unused files):
  - `README_original.md` - Original README

- **results/** (output):
  - Feature scores, top features, labels, etc.

## Quick Start

```bash
# Option 1: Use run_search.py (recommended)
python run_search.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>

# Option 2: Use complete analysis script
bash run_analysis.sh

# Option 3: Use compute_score.py directly
python compute_score.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>
```

## Configuration

Edit `config.yaml` or `config.json` to customize:
- Dataset path
- Scoring method (`simple`, `fisher`, or `domain`)
- Number of features
- Context window size

## Results

All results are saved in `results/`:
- `feature_scores.pt`: All feature scores
- `top_features.pt`: Top feature indices
- `top_features_scores.json`: Top features with scores
- `feature_labels.json`: Feature labels (if generated)
