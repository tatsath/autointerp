# FinanceScore Analysis

FinanceScore is computed using the same methodology as ReasonScore, adapted for financial domain data.

## Methodology

Following the ReasonScore approach:
- **Model**: `meta-llama/Llama-3.1-8B-Instruct` (same as run_top10.sh)
- **SAE**: `/home/nvidia/work/autointerp/converted_safetensors` (same as run_top10.sh)
- **Layer**: 19 (same as run_top10.sh)
- **Dataset**: `jyanimaulik/yahoo_finance_stockmarket_news`
- **Window**: Asymmetric window with 2 preceding and 3 subsequent tokens
- **Entropy penalty**: α = 0.7
- **Quantile threshold**: q = 0.997 (results in ~200 top features)
- **Token count**: 10M tokens (same as ReasonScore)

## Usage

Run the complete analysis:

```bash
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_domain_features
bash domains/finance/run_finance_analysis.sh
```

Or run components separately:

```bash
# Compute FinanceScore
python domains/finance/compute_finance_score.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/work/autointerp/converted_safetensors \
    --sae_id blocks.19.hook_resid_post

# Generate dashboard (after scores are computed)
python ../compute_dashboard.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/work/autointerp/converted_safetensors \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --scores_dir domains/finance/scores \
    --sae_id blocks.19.hook_resid_post \
    --topk 200 \
    --output_dir domains/finance/dashboards
```

## Output Files

After running, you'll find:
- `scores/feature_scores.pt` - All feature scores
- `scores/top_features.pt` - Indices of top features (quantile filtered)
- `scores/top_features_scores.json` - Top features with scores and metadata
- `dashboards/topk-200.html` - Interactive dashboard

## Configuration

Edit `config.json` to customize:
- Dataset path
- Token file path
- Window size (`expand_range`)
- Alpha value
- Quantile threshold
- Sample size

## Finance Tokens

The `finance_tokens.json` file contains financial domain keywords:
- Stock market terms (stock, price, market, earnings, etc.)
- Financial indicators (revenue, profit, dividend, IPO)
- Market exchanges (NASDAQ, NYSE, Dow, S&P)

You can customize this file with additional financial terms relevant to your analysis.

