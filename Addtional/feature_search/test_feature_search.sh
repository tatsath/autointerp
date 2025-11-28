#!/bin/bash

# Test script for feature search
# Tests the main run_feature_search.py with sample parameters

echo "🧪 Testing Feature Search"
echo "=========================="
echo ""

# Set GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment (adjust if needed)
if command -v conda &> /dev/null; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reasoning 2>/dev/null || echo "⚠️  Conda environment 'reasoning' not found, continuing..."
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
SAE_ID="blocks.19.hook_resid_post"
DATASET_PATH="jyanimaulik/yahoo_finance_stockmarket_news"
TOKENS_FILE="test_tokens.json"
OUTPUT_DIR="./test_results"
SCORE_TYPE="fisher"
NUM_FEATURES=20
N_SAMPLES=100

# Create test token file if it doesn't exist
if [ ! -f "$TOKENS_FILE" ]; then
    echo "📝 Creating test token file: $TOKENS_FILE"
    cat > "$TOKENS_FILE" << 'EOF'
[
  "stock", " price", "market", " earnings", "revenue", "profit",
  "dividend", " trading", " investment"
]
EOF
    echo "✅ Created $TOKENS_FILE"
    echo ""
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📊 Configuration:"
echo "   • Model: $MODEL_PATH"
echo "   • SAE: $SAE_PATH"
echo "   • SAE ID: $SAE_ID"
echo "   • Dataset: $DATASET_PATH"
echo "   • Tokens: $TOKENS_FILE"
echo "   • Output: $OUTPUT_DIR"
echo "   • Score Type: $SCORE_TYPE"
echo "   • Num Features: $NUM_FEATURES"
echo "   • Samples: $N_SAMPLES"
echo ""

# Run feature search
echo "🚀 Running feature search..."
echo ""

python main/run_feature_search.py \
    --model_path "$MODEL_PATH" \
    --sae_path "$SAE_PATH" \
    --sae_id "$SAE_ID" \
    --dataset_path "$DATASET_PATH" \
    --tokens_str_path "$TOKENS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --score_type "$SCORE_TYPE" \
    --num_features "$NUM_FEATURES" \
    --n_samples "$N_SAMPLES" \
    --expand_range 2,3

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature search completed successfully!"
    echo ""
    echo "📁 Results saved to: $OUTPUT_DIR"
    echo "   • feature_scores.pt"
    echo "   • top_features.pt"
    echo "   • feature_list.json"
    echo ""
else
    echo ""
    echo "❌ Feature search failed"
    exit 1
fi

