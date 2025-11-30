#!/bin/bash
# Run basic labeling for Nemotron reasoning features

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Update this path to match your actual search output directory
# Use the specific reasoning search output directory
SEARCH_OUTPUT="../results/1_search/nvidia_nemotron_nano_9b_v2_reasoning_l28_20251129_201924"
OUTPUT="../results/2_labeling_lite"

echo "🏷️  Nemotron Reasoning Basic Labeling"
echo "======================================="
echo "Search output: $SEARCH_OUTPUT"
echo "Output: $OUTPUT"
echo ""

# Verify the search output directory exists
if [ ! -d "$SEARCH_OUTPUT" ]; then
    echo "⚠️  Error: Search output directory not found: $SEARCH_OUTPUT"
    echo "Looking for alternative reasoning directories..."
    # Try to find the most recent reasoning search output
    BASE_SEARCH="../results/1_search"
    if [ -d "$BASE_SEARCH" ]; then
        REASONING_DIR=$(find "$BASE_SEARCH" -maxdepth 1 -type d -name "*reasoning*" | sort -r | head -1)
        if [ -n "$REASONING_DIR" ]; then
            SEARCH_OUTPUT="$REASONING_DIR"
            echo "Found reasoning search output: $SEARCH_OUTPUT"
        else
            echo "❌ No reasoning search output found. Please run the search first."
            exit 1
        fi
    else
        echo "❌ Search results directory not found: $BASE_SEARCH"
        exit 1
    fi
fi

echo "✅ Using search output: $SEARCH_OUTPUT"

# Run labeling pipeline with reasoning-specific label script
# We need to modify the environment to use label_features_reasoning.py
python run_labeling.py \
    --search_output "$SEARCH_OUTPUT" \
    --output_dir "$OUTPUT"

# After run_labeling.py completes, we need to re-run with reasoning label script
# Check if activating_sentences.json exists
if [ -f "$OUTPUT/activating_sentences.json" ]; then
    echo ""
    echo "🔄 Re-running with reasoning-specific label script..."
    
    # Set environment variables for reasoning label script
    export FEATURE_LIST_JSON="$SEARCH_OUTPUT/feature_list.json"
    export ACTIVATING_CONTEXTS_JSON="$OUTPUT/activating_sentences.json"
    export OUTPUT_JSON="$OUTPUT/feature_labels.json"
    
    # Run reasoning-specific label script
    python label_features_reasoning.py
fi

echo ""
echo "✅ Labeling complete: $OUTPUT"

