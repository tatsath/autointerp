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
SEARCH_OUTPUT=$(ls -td ../results/1_search/finbert* 2>/dev/null | head -1)
OUTPUT="../results/3_labeling_advance"

if [ -z "$SEARCH_OUTPUT" ]; then
    echo "❌ No FinBERT search output found"
    exit 1
fi

echo "🏷️  FinBERT Advanced Labeling"
echo "============================="
echo "Model: $MODEL"
echo "SAE: $SAE"
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

python run_finbert.py \
    --model_path "$MODEL" \
    --sae_path "$SAE" \
    --sae_id "encoder.layer.10.output" \
    --dataset_path "$DATASET" \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT" \
    --n_samples 5000 \
    --max_examples_per_feature 20

echo "✅ Advanced labeling complete"

