#!/bin/bash

# Script to extract finance feature examples using SaeVisRunner
# Similar to SAE-Reasoning paper approach but outputs JSONL instead of HTML

set -e

echo "🚀 Finance Feature Examples Extraction"
echo "======================================"
echo ""

# Set GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment (adjust if needed)
if command -v conda &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reasoning 2>/dev/null || echo "⚠️  Conda environment 'reasoning' not found, continuing..."
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration - adjust these paths as needed
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
SAE_PATH="${SAE_PATH:-/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU}"
SAE_ID="${SAE_ID:-blocks.19.hook_resid_post}"
DATASET_PATH="${DATASET_PATH:-jyanimaulik/yahoo_finance_stockmarket_news}"
FEATURE_LIST_PATH="${FEATURE_LIST_PATH:-../domains/finance/scores/top_features_scores.json}"
OUTPUT_PATH="${OUTPUT_PATH:-../domains/finance/scores/feature_examples_finance.jsonl}"
N_SAMPLES="${N_SAMPLES:-5000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"

echo "📊 Configuration:"
echo "   • Model: $MODEL_PATH"
echo "   • SAE: $SAE_PATH"
echo "   • SAE ID: $SAE_ID"
echo "   • Dataset: $DATASET_PATH"
echo "   • Feature List: $FEATURE_LIST_PATH"
echo "   • Output: $OUTPUT_PATH"
echo "   • Samples: $N_SAMPLES"
echo "   • Max Examples per Feature: $MAX_EXAMPLES"
echo ""

# Check if feature list exists
if [ ! -f "$FEATURE_LIST_PATH" ]; then
    echo "❌ Error: Feature list not found at $FEATURE_LIST_PATH"
    exit 1
fi

# Run the extraction script
echo "🚀 Running feature examples extraction..."
echo ""

python compute_examples_finance.py \
    --model_path "$MODEL_PATH" \
    --sae_path "$SAE_PATH" \
    --sae_id "$SAE_ID" \
    --dataset_path "$DATASET_PATH" \
    --feature_list_path "$FEATURE_LIST_PATH" \
    --output_path "$OUTPUT_PATH" \
    --n_samples "$N_SAMPLES" \
    --max_examples_per_feature "$MAX_EXAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature examples extraction completed successfully!"
    echo ""
    echo "📁 Results saved to: $OUTPUT_PATH"
    echo ""
    
    # Optional: Generate labels from examples
    GENERATE_LABELS="${GENERATE_LABELS:-true}"
    if [ "$GENERATE_LABELS" = "true" ]; then
        LABELS_OUTPUT="${LABELS_OUTPUT:-$(dirname "$OUTPUT_PATH")/feature_labels.json}"
        echo "🏷️  Generating labels from examples..."
        echo ""
        
        python generate_labels_from_examples.py \
            --examples_jsonl_path "$OUTPUT_PATH" \
            --output_path "$LABELS_OUTPUT" \
            --model_path "$MODEL_PATH" \
            --max_examples_per_feature 10
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Labels generated successfully!"
            echo "📁 Labels saved to: $LABELS_OUTPUT"
        else
            echo ""
            echo "⚠️  Label generation failed (examples are still available)"
        fi
    fi
    
    echo ""
    echo "You can view the examples with:"
    echo "   head -n 1 $OUTPUT_PATH | python -m json.tool"
    echo ""
else
    echo ""
    echo "❌ Feature examples extraction failed"
    exit 1
fi

