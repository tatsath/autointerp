#!/bin/bash
# Run advanced labeling for Llama finance features

set -e

export CUDA_VISIBLE_DEVICES="1,2"
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
SEARCH_OUTPUT="../results/1_search/llama_3_1_8b_instruct_finance_l19_20251128_220137"
OUTPUT="../results/3_labeling_advance"

echo "🏷️  Llama Advanced Labeling"
echo "==========================="
echo "Model: $BASE_MODEL"
echo "SAE: $SMALL_SAE_MODEL"
echo "SAE ID: $SAE_ID"
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

python run_labeling_advanced.py \
    --model_path "$BASE_MODEL" \
    --sae_path "$SMALL_SAE_MODEL" \
    --sae_id "$SAE_ID" \
    --dataset_path "$DATASET" \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT" \
    --n_samples 2000 \
    --max_examples_per_feature 20 \
    --minibatch_size_features 32 \
    --minibatch_size_tokens 16

echo ""
echo "✅ Advanced labeling complete: $OUTPUT"

