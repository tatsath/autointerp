# Domain-Specific Feature Extraction

This module extracts domain-specific features from SAEs (Sparse Autoencoders) similar to how ReasonScore works for reasoning data, but adapted for any domain (financial, healthcare, science, math, etc.).

## Overview

The codebase uses a modular architecture: **core functions** (`compute_score.py`, `compute_dashboard.py`) remain domain-agnostic and unchanged, while **domain-specific analyses** are organized in separate folders under `domains/`.

## Quick Start - Unified Interface

The easiest way to run feature search is using the unified command-line interface:

```bash
python run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path domains/finance/finance_tokens.json \
    --output_dir ./results \
    --sae_id blocks.19.hook_resid_post \
    --score_type simple \
    --num_features 100 \
    --n_samples 10000 \
    --generate_dashboard
```

This will:
1. Compute feature scores
2. Select top N features
3. Save feature list to `feature_list.json`
4. Optionally generate HTML dashboard

See `python run_feature_search.py --help` for all available options.

## FinanceScore Implementation

**What was done:** We implemented FinanceScore to identify financial features from a 400-feature SAE trained on Llama-3.1-8B-Instruct layer 19. The system computes feature scores based on activation patterns around financial tokens, applies entropy penalty (α=0.7), and selects top features using quantile filtering.

**Dataset selection:** The dataset `jyanimaulik/yahoo_finance_stockmarket_news` was chosen as it contains financial news and market-related content. Financial keywords (stock, price, market, earnings, revenue, etc.) were tokenized to identify relevant context windows.

**Process:** For each financial keyword occurrence, we analyze a context window of 2 preceding and 3 subsequent tokens. Features that activate highly in these contexts receive higher scores, while high-entropy (less specific) features are penalized. Top features are selected using a quantile threshold (q=0.95 for 400-feature SAE).

**Output:** Results are saved in `domains/finance/scores/`:
- `top_features.pt` - Top feature indices tensor
- `top_features_scores.json` - Feature indices, scores, and quantile threshold
- `feature_scores.pt` - Full score tensor for all features

**Sample output** (`top_features_scores.json`): Top 20 features selected (e.g., features 7, 11, 100, 106, 138...) with scores ranging from 0.001 to 0.015, using quantile threshold 0.95.

### Running FinanceScore

```bash
cd autointerp_domain_features
bash domains/finance/run_finance_analysis.sh
```

**Configuration:** SAE with 400 features, asymmetric window [2,3], α=0.7, q=0.95 (selects ~20 top features), model: Llama-3.1-8B-Instruct, layer 19.

### Creating a New Domain

1. Copy template: `cp -r domains/_template domains/health`
2. Update `config.json` with your dataset and tokens
3. Create domain token file (e.g., `health_tokens.json`)
4. Rename wrapper script and create run script

**Core functions remain unchanged!**

## Direct Usage (Without Domain Folders)

You can also use core functions directly:

### 1. Compute Domain Scores

```bash
python autointerp_domain_features/compute_score.py \
    --model_path <model_path> \
    --sae_path <sae_path> \
    --dataset_path <dataset_path> \
    --tokens_str_path <tokens_json_path> \
    --output_dir <output_dir> \
    --expand_range 2,3 \
    --alpha 0.7 \
    --score_type domain \
    --n_samples 10000
```

### 2. Generate Dashboard

```bash
python autointerp_domain_features/compute_dashboard.py \
    --model_path <model_path> \
    --sae_path <sae_path> \
    --dataset_path <dataset_path> \
    --scores_dir <scores_dir> \
    --output_dir <output_dir> \
    --num_features 200
```

## Parameters

- `tokens_str_path`: Path to JSON file with domain-specific token strings
- `dataset_path`: HuggingFace dataset path or local path
- `score_type`: Scoring method - `"domain"` (default, token-based with entropy), `"simple"` (|μ⁺ - μ⁻|), or `"fisher"` ((μ⁺ - μ⁻)² / (σ⁺ + σ⁻ + ε))
- `alpha`: Weight for entropy term in DomainScore calculation (default: 0.7, only used with `score_type="domain"`)
- `expand_range`: Tuple (left, right) to expand context around matched tokens (e.g., [2,3] for 2 preceding, 3 subsequent)
- `num_features`: Number of top features to return (default: 100)
- `selection_method`: `"topk"` (select top N) or `"quantile"` (select by quantile threshold)
- `quantile_threshold`: Quantile for selecting top features when using `selection_method="quantile"` (default: 0.95)
- `ignore_tokens`: List of token IDs to ignore

## Domain-Specific Folders

- `domains/finance/` - FinanceScore implementation
- `domains/_template/` - Template for creating new domains
- Future: `domains/health/`, `domains/science/`, etc.
