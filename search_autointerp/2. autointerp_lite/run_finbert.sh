#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="1,2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Find latest FinBERT search output
SEARCH_OUTPUT=$(ls -td ../results/1_search/finbert* 2>/dev/null | head -1)
OUTPUT="../results/2_labeling_lite"

if [ -z "$SEARCH_OUTPUT" ]; then
    echo "❌ No FinBERT search output found"
    exit 1
fi

echo "🏷️  FinBERT Basic Labeling"
echo "=========================="
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

python run_labeling.py \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT"

echo "✅ Labeling complete"

