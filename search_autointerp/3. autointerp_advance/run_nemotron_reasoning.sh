#!/bin/bash
# Run advanced labeling for Nemotron reasoning features

set -e

export CUDA_VISIBLE_DEVICES="3,4"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Nemotron config
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_sae_converted"
SAE_ID="blocks.28.hook_resid_post"
DATASET="open-thoughts/OpenThoughts-114k"

# Update this path to match your actual search output directory
SEARCH_OUTPUT="../results/1_search"
OUTPUT="../results/3_labeling_advance"

echo "🏷️  Nemotron Reasoning Advanced Labeling"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""
echo "Note: This script will auto-detect the reasoning search output directory."
echo "Make sure you've run the reasoning search first."
echo ""

# Find the most recent reasoning search output
if [ -d "$SEARCH_OUTPUT" ]; then
    # Look for directories matching reasoning pattern
    REASONING_DIR=$(find "$SEARCH_OUTPUT" -maxdepth 1 -type d -name "*reasoning*" -o -name "*nemotron*reasoning*" | sort -r | head -1)
    
    if [ -n "$REASONING_DIR" ]; then
        SEARCH_OUTPUT="$REASONING_DIR"
        echo "Found reasoning search output: $SEARCH_OUTPUT"
    else
        echo "⚠️  Warning: No reasoning search output found. Using default: $SEARCH_OUTPUT"
    fi
fi

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



