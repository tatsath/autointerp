#!/bin/bash
# Run finance feature search for Llama model (top 100 features)

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Llama config
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SMALL_SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
SAE_ID="blocks.19.hook_resid_post"
DATASET="jyanimaulik/yahoo_finance_stockmarket_news"
TOKENS="domains/finance/finance_tokens.json"
OUTPUT="../results/1_search"
NUM_FEATURES=100

mkdir -p "$OUTPUT"

echo "🔍 Llama Finance Search (Top $NUM_FEATURES)"
echo "==========================================="
echo "Model: $BASE_MODEL"
echo "SAE: $SMALL_SAE_MODEL"
echo "SAE ID: $SAE_ID"
echo "Output: $OUTPUT"
echo ""

python main/run_feature_search.py \
    --model_path "$BASE_MODEL" \
    --sae_path "$SMALL_SAE_MODEL" \
    --sae_id "$SAE_ID" \
    --dataset_path "$DATASET" \
    --tokens_str_path "$TOKENS" \
    --output_dir "$OUTPUT" \
    --score_type fisher \
    --num_features $NUM_FEATURES \
    --n_samples 1000 \
    --expand_range 1,2

echo ""
echo "✅ Search complete: $OUTPUT"



