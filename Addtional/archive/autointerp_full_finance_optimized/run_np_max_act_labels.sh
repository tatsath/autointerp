#!/bin/bash

echo "🚀 Neuronpedia Max-Activation Label Generation - Nemotron Features"
echo "=================================================================="
echo "🔍 Generating labels using Neuronpedia-style max-activation explainer"
echo "⚡ Using cached activations from Delphi"
echo "📊 Processing top features (Layer 28, K=64)"
echo ""

# Get script directory (autointerp_full_finance_optimized)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to EndtoEnd/Autointerp directory (where SAE and features are)
ENDPOINT_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp"

# Configuration
BASE_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NEMOTRON_SAE_MODEL_DIR="$ENDPOINT_DIR/nemotron_sae_converted"
FEATURES_SUMMARY_PATH="$ENDPOINT_DIR/nemotron_finance_features/top_finance_features_summary.txt"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
LAYER=28
DICT_SIZE=35840
NUM_FEATURES_TO_RUN=5  # Top 5 features for testing
RUN_NAME="nemotron_finance_news_run_np_max_act"

# Results directory (inside results folder)
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🤖 Using vLLM provider: $EXPLAINER_PROVIDER"
echo "🌐 vLLM server URL: $EXPLAINER_API_BASE_URL"
echo "🎯 Processing top $NUM_FEATURES_TO_RUN features (Layer $LAYER, K=64) out of $DICT_SIZE total"
echo "📋 Using Neuronpedia max-activation explainer (K=24, window=12)"
echo "📊 Using cached activations from Delphi (no code changes required)"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Check if vLLM server is running
echo "🔍 Checking vLLM server status..."
if curl -s "$EXPLAINER_API_BASE_URL/models" > /dev/null 2>&1; then
    echo "✅ vLLM server is running at $EXPLAINER_API_BASE_URL"
else
    echo "❌ vLLM server is not running at $EXPLAINER_API_BASE_URL"
    echo "Please start vLLM server first:"
    echo "bash start_vllm_server_72b.sh"
    exit 1
fi
echo ""

# Extract top features from summary file
echo "🔍 Extracting top $NUM_FEATURES_TO_RUN features from summary..."
cd "$ENDPOINT_DIR"
python3 -c "
import re

# Read summary file
with open('$FEATURES_SUMMARY_PATH', 'r') as f:
    lines = f.readlines()

# Extract feature indices
features = []
pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
for line in lines:
    match = pattern.match(line)
    if match:
        features.append(int(match.group(1)))
        if len(features) >= $NUM_FEATURES_TO_RUN:
            break

# Write feature list
with open('nemotron_finance_news_features_list.txt', 'w') as f:
    f.write(' '.join(map(str, features)))

print(f'✅ Extracted {len(features)} features: {features[:10]}...')
"

# Read the feature list
FEATURE_LIST=$(cat "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt")

if [ -z "$FEATURE_LIST" ]; then
    echo "❌ Failed to extract features from summary file"
    exit 1
fi

# Check if cached activations exist (use the system run's cache)
CACHED_LATENTS_PATH="$SCRIPT_DIR/results/nemotron_finance_news_run_system/latents"

if [ ! -d "$CACHED_LATENTS_PATH" ]; then
    echo "❌ Cached activations not found at: $CACHED_LATENTS_PATH"
    echo "Please run the system explainer first to generate cached activations:"
    echo "  bash run_nemotron_finance_news_autointerp_system.sh"
    exit 1
fi

echo "✅ Found cached activations at: $CACHED_LATENTS_PATH"
echo ""

echo "🔍 Running Neuronpedia Max-Activation Label Generation..."
echo "========================================================="

# Navigate to the autointerp_full_finance_optimized directory
cd "$SCRIPT_DIR"

# Create explanations directory
EXPLANATIONS_DIR="$RESULTS_DIR/explanations"
mkdir -p "$EXPLANATIONS_DIR"

# Run the Neuronpedia explainer
python run_np_max_act_explainer.py \
    --latents_path "$CACHED_LATENTS_PATH" \
    --explanations_path "$EXPLANATIONS_DIR" \
    --hookpoints "backbone.layers.$LAYER" \
    --feature_num $FEATURE_LIST \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
    --model "$BASE_MODEL" \
    --k_max_act 24 \
    --window 12 \
    --verbose 2>&1 | tee "$RESULTS_DIR/run.log" || true

# Check if results were generated
if [ -d "$EXPLANATIONS_DIR" ] && [ "$(ls -A $EXPLANATIONS_DIR 2>/dev/null)" ]; then
    echo "✅ Label generation completed (results found)"
    
    # Count explanations
    EXPLANATION_COUNT=$(find "$EXPLANATIONS_DIR" -name "*.txt" 2>/dev/null | wc -l)
    echo "📊 Generated $EXPLANATION_COUNT feature labels"
    
    # Run scoring on the generated labels
    echo ""
    echo "🔍 Running scoring on generated labels..."
    echo "=========================================="
    cd "$SCRIPT_DIR"
    
    # Run scoring using the same cached activations and the new explanations
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$NEMOTRON_SAE_MODEL_DIR" \
        --n_tokens 3000000 \
        --cache_ctx_len 1024 \
        --batch_size 1 \
        --feature_num $FEATURE_LIST \
        --hookpoints "layers.$LAYER" \
        --scorers detection \
        --explainer_model "$EXPLAINER_MODEL" \
        --explainer_provider "$EXPLAINER_PROVIDER" \
        --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
        --explainer_model_max_len 8192 \
        --num_gpus 4 \
        --num_examples_per_scorer_prompt 10 \
        --n_non_activating 50 \
        --min_examples 5 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "FinLang/finance-embeddings-investopedia" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --dataset_repo ashraq/financial-news \
        --dataset_name default \
        --dataset_split "train[:50%]" \
        --dataset_column headline \
        --filter_bos \
        --verbose \
        --overwrite scores \
        --name "$RUN_NAME" 2>&1 | tee -a "$RESULTS_DIR/scoring.log" || true
    
    # Generate CSV summary with scores
    if [ -d "$EXPLANATIONS_DIR" ] && [ "$(ls -A $EXPLANATIONS_DIR 2>/dev/null)" ]; then
        echo ""
        echo "📊 Generating CSV summary for Neuronpedia labels with scores..."
        cd "$SCRIPT_DIR"
        
        # Generate enhanced CSV
        echo "📊 Generating enhanced CSV with labels and scores..."
        python generate_nemotron_enhanced_csv.py "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/csv_generation.log" || true
        
        if [ -f "$RESULTS_DIR/nemotron_finance_results_summary_enhanced.csv" ]; then
            echo "✅ Enhanced CSV saved: $RESULTS_DIR/nemotron_finance_results_summary_enhanced.csv"
        else
            echo "⚠️  Enhanced CSV generation may have failed, check logs"
        fi
    fi
else
    echo "❌ Label generation failed - no results found"
    exit 1
fi

# Clean up temporary file
rm -f "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt"

echo ""
echo "🔬 Neuronpedia max-activation label generation completed!"
echo "📊 Analyzed top $NUM_FEATURES_TO_RUN features (out of $DICT_SIZE total) from the Nemotron SAE model (Layer $LAYER, K=64)"
echo "📁 Results saved in: $RESULTS_DIR"
echo "📄 CSV files generated in results folder:"
echo "   - nemotron_finance_results_summary_enhanced.csv (if generated)"
echo ""
echo "📝 Note: Using Neuronpedia-style max-activation explainer"
echo "   - Uses top 24 max-activation examples with ±12 token context"
echo "   - Generates concise, precise financial labels (≤18 words)"
echo "   - Works with cached activations from Delphi (no code changes)"
echo ""

