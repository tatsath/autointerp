#!/bin/bash
# Run advanced labeling for Nemotron finance features

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Nemotron config
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/nemotron_nano_layer28_features35840_k64/trainer_0"
SAE_ID="blocks.28.hook_resid_post"
DATASET="jyanimaulik/yahoo_finance_stockmarket_news"
SEARCH_OUTPUT="../results/1_search"
OUTPUT="../results/3_labeling_advance"

echo "🏷️  Nemotron Advanced Labeling"
echo "=============================="
echo "Model: $MODEL"
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

python run_labeling_advanced.py \
    --model_path "$MODEL" \
    --sae_path "$SAE" \
    --sae_id "$SAE_ID" \
    --dataset_path "$DATASET" \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT" \
    --n_samples 5000 \
    --max_examples_per_feature 20

echo ""
echo "✅ Advanced labeling complete: $OUTPUT"

