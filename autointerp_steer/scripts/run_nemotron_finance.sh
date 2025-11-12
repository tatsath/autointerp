#!/bin/bash

echo "🚀 AutoInterp Steer - Nemotron SAE Steering & Interpretation Pipeline (Finance)"
echo "=============================================================================="
echo ""
echo "📁 OUTPUT LOCATIONS:"
echo "   • Steering outputs (intermediate): ./nemotron_steering_outputs/"
echo "     - JSON files per feature/prompt with steering results"
echo "   • Feature labels (JSON): ./interpretation_outputs_nemotron/interpretations.json"
echo "   • Feature labels (CSV):  ./nemotron_steering_outputs/nemotron_interpretations_summary.csv"
echo ""
echo "⚡ SPEED OPTIMIZATIONS APPLIED:"
echo "   • Features: Top 10"
echo "   • Prompts: 5 (reduced for speed)"
echo "   • Steering levels: 4 - [-2.0, -1.0, 1.0, 2.0]"
echo "   • Max tokens: 32 (reduced for speed)"
echo "   • Num batches: 1 (optimized for speed)"
echo "   • Dataset: ashraq/financial-news"
echo ""

# Configuration
BASE_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NEMOTRON_SAE_MODEL="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_sae_converted"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE="http://127.0.0.1:8002/v1"
LAYER=28

# Paths
FEATURES_SUMMARY="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_finance_features/top_finance_features_summary.txt"
STEERING_OUTPUTS_DIR="nemotron_steering_outputs"
INTERPRETATIONS_OUTPUT_DIR="interpretation_outputs_nemotron"

# Feature selection - use top 10 features from summary
TOP_N_FEATURES=10

# Speed optimizations
NUM_PROMPTS=5
DATASET_SPLIT="train[:100]"
NUM_BATCHES=1
MAX_NEW_TOKENS=32

# Dataset configuration
DATASET_REPO="ashraq/financial-news"
DATASET_NAME=""

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "🔧 GPU Configuration:"
echo "   • Steering pipeline: GPUs 0,1,2,3 (via CUDA_VISIBLE_DEVICES)"
echo "   • vLLM server: GPUs 4,5,6,7 (for feature labeling)"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Check vLLM server status
EXPLAINER_API_BASE_URL="http://127.0.0.1:8002/v1"
echo "🔍 Checking vLLM server status..."
if curl -s "$EXPLAINER_API_BASE_URL/models" > /dev/null 2>&1; then
    echo "✅ vLLM server is running at $EXPLAINER_API_BASE_URL"
else
    echo "❌ vLLM server is not running at $EXPLAINER_API_BASE_URL"
    echo "Please start the vLLM server separately first"
    exit 1
fi

# Navigate to autointerp_steer
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_steer

echo "📊 Step 1: Running Steering Experiments"
echo "======================================="
echo ""

mkdir -p "$STEERING_OUTPUTS_DIR"

# Extract features from summary file
if [ -f "$FEATURES_SUMMARY" ]; then
    echo "📋 Extracting top features from summary..."
    python3 -c "
import re
features = []
pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
with open('$FEATURES_SUMMARY', 'r') as f:
    for line in f:
        match = pattern.match(line)
        if match:
            features.append(int(match.group(1)))
            if len(features) >= $TOP_N_FEATURES:
                break
print(' '.join(map(str, features)))
" > /tmp/nemotron_features_for_steering.txt
    
    FEATURES_LIST=$(cat /tmp/nemotron_features_for_steering.txt)
    echo "✓ Selected $TOP_N_FEATURES features from $FEATURES_SUMMARY"
else
    echo "⚠️  Features summary not found at $FEATURES_SUMMARY"
    echo "   Using default features 0-9"
    FEATURES_LIST="0 1 2 3 4 5 6 7 8 9"
fi

echo ""
echo "🔬 Running steering experiments..."
echo "   Model: $BASE_MODEL"
echo "   SAE: $NEMOTRON_SAE_MODEL"
echo "   Layer: $LAYER"
echo "   Features: $(echo $FEATURES_LIST | wc -w) features"
echo "   Prompts: $NUM_PROMPTS"
echo ""

# Check if SAE path exists
if [ ! -d "$NEMOTRON_SAE_MODEL" ]; then
    echo "❌ Error: Nemotron SAE path not found: $NEMOTRON_SAE_MODEL"
    exit 1
fi

# Run steering experiment
# Use cuda:1 (maps to physical GPU 1) since GPU 0 may be full
python scripts/run_steering_nemotron.py \
    --output_folder "$STEERING_OUTPUTS_DIR" \
    --sae_path "$NEMOTRON_SAE_MODEL" \
    --model_name "$BASE_MODEL" \
    --layer "$LAYER" \
    --device "cuda:1" \
    --features_list $FEATURES_LIST \
    --features_summary "$FEATURES_SUMMARY" \
    --dataset_repo "$DATASET_REPO" \
    --dataset_name "$DATASET_NAME" \
    --dataset_split "$DATASET_SPLIT" \
    --num_prompts $NUM_PROMPTS \
    --num_batches $NUM_BATCHES \
    --max_new_tokens $MAX_NEW_TOKENS

if [ $? -ne 0 ]; then
    echo "❌ Steering experiments failed"
    exit 1
fi

echo ""
echo "✅ Steering experiments completed!"
echo ""

# Step 2: Run feature labeling/interpretation
echo "📊 Step 2: Running Feature Labeling"
echo "======================================"
echo ""

echo "🔬 Labeling features using $EXPLAINER_MODEL..."
echo ""

python scripts/run_interpretation.py \
    --steering_output_dir "$STEERING_OUTPUTS_DIR" \
    --output_dir "$INTERPRETATIONS_OUTPUT_DIR" \
    --explainer_api_base "$EXPLAINER_API_BASE" \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_max_tokens 256 \
    --explainer_temperature 0.0 \
    --layers "$LAYER"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Feature labeling completed!"
    INTERPRETATIONS_OUTPUT="$INTERPRETATIONS_OUTPUT_DIR/interpretations.json"
    echo "📄 Labels saved to: $INTERPRETATIONS_OUTPUT"
    
    # Generate CSV summary
    echo ""
    echo "📊 Generating CSV summary..."
    if [ -f "$INTERPRETATIONS_OUTPUT" ]; then
        CSV_OUTPUT="$STEERING_OUTPUTS_DIR/nemotron_interpretations_summary.csv"
        python scripts/generate_results_csv.py "$INTERPRETATIONS_OUTPUT" "$STEERING_OUTPUTS_DIR" "$CSV_OUTPUT"
        
        if [ $? -eq 0 ] && [ -f "$CSV_OUTPUT" ]; then
            echo "✅ CSV summary generated: $CSV_OUTPUT"
        else
            echo "⚠️  CSV generation failed (but JSON interpretations are available)"
        fi
    else
        echo "⚠️  Interpretations JSON not found, skipping CSV generation"
    fi
else
    echo "⚠️  Feature labeling failed (but steering outputs are available)"
fi

# Cleanup
rm -f /tmp/nemotron_features_for_steering.txt

echo ""
echo "🎉 Pipeline completed!"
echo "📊 Steering outputs: $STEERING_OUTPUTS_DIR/"
INTERPRETATIONS_OUTPUT="$INTERPRETATIONS_OUTPUT_DIR/interpretations.json"
if [ -f "$INTERPRETATIONS_OUTPUT" ]; then
    echo "📄 Feature labels (JSON): $INTERPRETATIONS_OUTPUT"
    CSV_OUTPUT="$STEERING_OUTPUTS_DIR/nemotron_interpretations_summary.csv"
    if [ -f "$CSV_OUTPUT" ]; then
        echo "📈 Feature labels (CSV): $CSV_OUTPUT"
    fi
fi
echo ""

