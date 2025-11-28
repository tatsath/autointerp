# 1. Search - Feature Search Functionality

This module implements feature search to identify domain-specific SAE features.

## How Domain is Determined

**The system is domain-agnostic** - it doesn't have a "domain" parameter. The domain is determined by:

1. **`tokens_str_path`** - Points to domain-specific token file (e.g., `domains/finance/finance_tokens.json`)
2. **`dataset_path`** - Points to domain-specific dataset (e.g., `jyanimaulik/yahoo_finance_stockmarket_news`)

**Two ways to run:**

### Option 1: Direct (Domain-Agnostic)
Pass all parameters explicitly - works for any domain:

```bash
python run_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path domains/finance/finance_tokens.json \
    --output_dir ../results/1_search \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100
```

### Option 2: Domain Wrapper (Finance Example)
Use domain-specific wrapper that loads from `config.json`:

```bash
cd domains/finance
python compute_finance_score.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --sae_id blocks.19.hook_resid_post
```

The wrapper reads `domains/finance/config.json` which contains:
- `dataset_path`: Finance dataset
- `tokens_str_path`: Finance tokens file
- Other finance-specific settings

## Quick Start

```bash
# Using the unified entry point (direct method)
python run_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path test_tokens.json \
    --output_dir ../results/1_search \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100

# Or use the shell script
./run_search.sh
```

## Output

Results are saved to `../results/1_search/`:
- `feature_scores.pt` - All feature scores
- `feature_list.json` - Top features with indices and scores
- `top_features.pt` - Top feature indices tensor

## Domain Selection

The system is **domain-agnostic** - there's no "domain" parameter. The domain is determined by:

1. **`tokens_str_path`** parameter - Points to domain-specific token file
   - Finance example: `domains/finance/finance_tokens.json`
   - Contains domain keywords: `["stock", "price", "market", "earnings", ...]`

2. **`dataset_path`** parameter - Points to domain-specific dataset
   - Finance example: `jyanimaulik/yahoo_finance_stockmarket_news`
   - Contains domain-specific text content

**For Finance domain specifically:**
- Use `domains/finance/finance_tokens.json` as `tokens_str_path`
- Use `jyanimaulik/yahoo_finance_stockmarket_news` as `dataset_path`
- Or use the wrapper: `domains/finance/compute_finance_score.py` (loads from `config.json`)

## Files

- `main/run_feature_search.py` - Main search implementation (domain-agnostic)
- `main/compute_score.py` - Core scoring algorithm (domain-agnostic)
- `domains/finance/` - Finance-specific wrappers and configs (optional convenience)
  - `config.json` - Finance configuration
  - `finance_tokens.json` - Finance token keywords
  - `compute_finance_score.py` - Wrapper that uses config.json
- `run_search.py` - Unified entry point
- `run_search.sh` - Shell script wrapper

## Example: Finance Domain

To search for finance features, you need:

1. **Finance token file** (`domains/finance/finance_tokens.json`):
   ```json
   ["stock", " price", "market", " earnings", "revenue", "profit", ...]
   ```

2. **Finance dataset** (`jyanimaulik/yahoo_finance_stockmarket_news`)

3. **Run search**:
   ```bash
   python run_search.py \
       --tokens_str_path domains/finance/finance_tokens.json \
       --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
       ...
   ```

The system identifies finance features by:
- Finding positions in the dataset where finance tokens appear
- Comparing feature activations at those positions vs. other positions
- Scoring features that activate more on finance tokens

## Next Steps

After search completes, proceed to:
- **2. autointerp_lite/** - Basic labeling using collected examples
- **3. autointerp_advance/** - Advanced labeling using SaeVisRunner

