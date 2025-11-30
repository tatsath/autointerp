#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1,2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="ProsusAI/finbert"
SAE="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/finbert_sae_converted/encoder.layer.10.output"
DATASET="jyanimaulik/yahoo_finance_stockmarket_news"
TOKENS="domains/finance/finance_tokens.json"
OUTPUT="../results/1_search"
NUM_FEATURES=100

mkdir -p "$OUTPUT"

echo "🔍 FinBERT Finance Search (Top $NUM_FEATURES)"
echo "============================================="

python run_finbert.py \
    --model_path "$MODEL" \
    --sae_path "$SAE" \
    --dataset_path "$DATASET" \
    --tokens_str_path "$TOKENS" \
    --output_dir "$OUTPUT" \
    --score_type fisher \
    --num_features $NUM_FEATURES \
    --n_samples 1000 \
    --expand_range 1,2

echo "✅ Search complete"

