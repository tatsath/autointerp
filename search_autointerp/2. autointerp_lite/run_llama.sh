#!/bin/bash
# Run basic labeling for Llama finance features

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEARCH_OUTPUT="../results/1_search/llama_3_1_8b_instruct_finance_l19_20251128_220137"
OUTPUT="../results/2_labeling_lite"

echo "🏷️  Llama Basic Labeling"
echo "========================"
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

python run_labeling.py \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT"

echo ""
echo "✅ Labeling complete: $OUTPUT"

