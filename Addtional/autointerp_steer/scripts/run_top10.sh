#!/bin/bash

echo "🚀 AutoInterp Steer - Large SAE Steering & Interpretation Pipeline"
echo "================================================================="
echo ""
echo "📁 OUTPUT LOCATIONS:"
echo "   • Steering outputs (intermediate): ./large_sae_steering_outputs/"
echo "     - JSON files per feature/prompt with steering results"
echo "   • Feature labels (JSON): ./large_sae_interpretations.json"
echo "   • Feature labels (CSV):  ./large_sae_interpretations_summary.csv"
echo ""
echo "⚡ RUNNING FEATURES 1 AND 2 WITH OPTIMIZED SETTINGS:"
echo "   • Features: 1, 2 (explicit selection)"
echo "   • Prompts: 5 (reduced for speed)"
echo "   • Steering levels: 4 - [-4.0, -2.0, 2.0, 4.0]"
echo "   • Max tokens: 32 (reduced for speed)"
echo "   • Num batches: 1 (optimized for speed)"
echo "   • Dataset: $DATASET_REPO/$DATASET_NAME"
echo ""

# Configuration from run_large_sae_analysis.sh
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LARGE_SAE_MODEL="/home/nvidia/work/autointerp/converted_safetensors"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE="http://127.0.0.1:8002/v1"
LAYER=19

# Paths
CLUSTERING_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Autointerp_clustering"
STEERING_OUTPUTS_DIR="large_sae_steering_outputs"
INTERPRETATIONS_OUTPUT_DIR="interpretation_outputs"

# Feature selection - explicitly use features 1 and 2
FEATURES_LIST="1 2"

# Prompt diversity - reduced for speed
NUM_PROMPTS=5  # Reduced from 10 to 5 for faster execution
DATASET_SPLIT="train[:100]"  # Reduced from 200 to 100 for faster loading
NUM_BATCHES=1  # Use 1 batch for max activation search (faster)
MAX_NEW_TOKENS=32  # Reduced from 50 to 32 for faster generation

# Dataset configuration - using wikitext-103-raw-v1
DATASET_REPO="wikitext"
DATASET_NAME="wikitext-103-raw-v1"  # wikitext-103-raw-v1 dataset
DATASET_CONFIG_NAME="default"  # For compatibility with script args

# Set environment variables for GPU and memory management FIRST (before conda activation)
# vLLM uses GPUs 4,5,6,7, so steering uses GPUs 0,1,2,3
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "🔧 GPU Configuration:"
echo "   • Steering pipeline: GPUs 0,1,2,3 (via CUDA_VISIBLE_DEVICES)"
echo "   • vLLM server: GPUs 4,5,6,7 (for feature labeling)"
echo "   Note: With CUDA_VISIBLE_DEVICES='0,1,2,3', cuda:0 maps to physical GPU 0"
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
    echo ""
    echo "Please start the vLLM server separately first:"
    echo ""
    echo "  cd $(dirname $0)/.."
    echo "  bash scripts/start_vllm_server.sh [GPU_ID]"
    echo ""
    echo "Or manually:"
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $EXPLAINER_MODEL \\"
    echo "    --port 8002 \\"
    echo "    --gpu-memory-utilization 0.5 \\"
    echo "    --max-model-len 4096 \\"
    echo "    --host 0.0.0.0 \\"
    echo "    --trust-remote-code"
    echo ""
    exit 1
fi

# Navigate to autointerp_steer
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_steer

echo "📊 Step 1: Running Steering Experiments"
echo "======================================="
echo ""

# Create steering output directory
mkdir -p "$STEERING_OUTPUTS_DIR"

# Use explicit feature list (features 1 and 2)
echo "📋 Using explicit feature list: $FEATURES_LIST"
echo "✓ Selected features: $FEATURES_LIST"

echo ""
echo "🔬 Running steering experiments..."
echo "   Model: $BASE_MODEL"
echo "   SAE: $LARGE_SAE_MODEL"
echo "   Layer: $LAYER"
echo "   Features: $(echo $FEATURES_LIST | wc -w) features"
echo "   Prompts: $NUM_PROMPTS"
echo "   Steering levels: 4"
echo "   Max tokens per generation: $MAX_NEW_TOKENS (optimized for speed)"
echo "   Num batches for max activation: $NUM_BATCHES (optimized for speed)"
echo ""

# Use cuda:0 which maps to physical GPU 0 when CUDA_VISIBLE_DEVICES="0,1,2,3"
# Run without background - all output shows directly
python scripts/run_steering_large_sae.py \
    --output_folder "$STEERING_OUTPUTS_DIR" \
    --sae_path "$LARGE_SAE_MODEL" \
    --model_name "$BASE_MODEL" \
    --layer "$LAYER" \
    --device "cuda:0" \
    --features_list $FEATURES_LIST \
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
        # Save CSV in the steering outputs directory
        CSV_OUTPUT="$STEERING_OUTPUTS_DIR/large_sae_interpretations_summary.csv"
        # Pass steering outputs directory to calculate steering effect scores
        python scripts/generate_results_csv.py "$INTERPRETATIONS_OUTPUT" "$STEERING_OUTPUTS_DIR" "$CSV_OUTPUT"
        
        if [ $? -eq 0 ] && [ -f "$CSV_OUTPUT" ]; then
            echo "✅ CSV summary generated: $CSV_OUTPUT"
            echo "   Note: steering_effect column measures feature impact (0.0-1.0)"
            echo "         Higher scores indicate stronger steering effects"
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
rm -f /tmp/large_features_for_steering.txt

echo ""
echo "🎉 Pipeline completed!"
echo "📊 Steering outputs: $STEERING_OUTPUTS_DIR/"
INTERPRETATIONS_OUTPUT="$INTERPRETATIONS_OUTPUT_DIR/interpretations.json"
if [ -f "$INTERPRETATIONS_OUTPUT" ]; then
    echo "📄 Feature labels (JSON): $INTERPRETATIONS_OUTPUT"
    CSV_OUTPUT="$STEERING_OUTPUTS_DIR/large_sae_interpretations_summary.csv"
    if [ -f "$CSV_OUTPUT" ]; then
        echo "📈 Feature labels (CSV): $CSV_OUTPUT"
    fi
fi
echo ""

