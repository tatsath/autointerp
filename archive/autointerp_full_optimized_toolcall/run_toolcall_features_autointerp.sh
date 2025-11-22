#!/bin/bash

echo "🚀 AutoInterp Tool-Call Feature Analysis - Llama 3.1 8B"
echo "=================================================================================="
echo "🔍 Analyzing tool-call features from Llama 3.1 8B SAE"
echo "⚡ Using vLLM server for faster inference"
echo "📊 Processing features from tool_features.json"
echo "🎯 Using tool-call prompts for agent tool-use analysis"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL_DIR="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
TOOL_FEATURES_JSON="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/AgenticTracing/tool_features.json"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=5000000  # 5M tokens for better F1 scores
LAYER=19
DICT_SIZE=400
RUN_NAME="toolcall_features_run"

# Results directory (no timestamp - reuse same location)
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🤖 Using vLLM provider: $EXPLAINER_PROVIDER"
echo "🌐 vLLM server URL: $EXPLAINER_API_BASE_URL"
echo "🎯 Processing features from tool_features.json (Layer $LAYER, K=32, Dict Size=$DICT_SIZE)"
echo "📋 Using tool-call prompts for agent tool-use analysis"
echo "🔍 Using ContrastiveExplainer with FAISS for hard negatives"
echo "📊 Using 20 examples per feature + 60 non-overlapping examples for scoring"
echo "🎯 Using Detection scorer only (faster execution)"
echo "📈 Using 5M tokens for analysis (cache will be reused if available)"
echo "🎯 Optimized for high F1 scores: min_examples=8, n_non_activating=60, examples_per_prompt=20"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Check if SAE exists
if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "❌ SAE not found at: $SAE_MODEL_DIR/layers.$LAYER"
    exit 1
fi

# Check if tool_features.json exists
if [ ! -f "$TOOL_FEATURES_JSON" ]; then
    echo "❌ tool_features.json not found at: $TOOL_FEATURES_JSON"
    exit 1
fi

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

# Extract features from tool_features.json
echo "🔍 Extracting features from tool_features.json..."
cd "$SCRIPT_DIR"
python3 -c "
import json

# Read tool_features.json
with open('$TOOL_FEATURES_JSON', 'r') as f:
    data = json.load(f)

# Extract all feature IDs from both market_features and news_features
features = []
for category in ['market_features', 'news_features']:
    if category in data:
        for item in data[category]:
            features.append(item['feature_id'])

# Remove duplicates and sort
features = sorted(list(set(features)))

# Write feature list
with open('toolcall_features_list.txt', 'w') as f:
    f.write(' '.join(map(str, features)))

print(f'✅ Extracted {len(features)} unique features: {features}')
print(f'   Market features: {len(data.get(\"market_features\", []))}')
print(f'   News features: {len(data.get(\"news_features\", []))}')
"

# Read the feature list
FEATURE_LIST=$(cat "$SCRIPT_DIR/toolcall_features_list.txt")

if [ -z "$FEATURE_LIST" ]; then
    echo "❌ Failed to extract features from tool_features.json"
    exit 1
fi

echo "🔍 Running AutoInterp Tool-Call Feature Analysis..."
echo "====================================================================="

# Navigate to the autointerp_full_optimized_toolcall directory
cd "$SCRIPT_DIR"

# Delete old explanations and scores (keep latents cache for reuse)
if [ -d "results/$RUN_NAME/explanations" ]; then
    echo "🗑️  Removing old explanations to force regeneration with updated prompts..."
    rm -rf "results/$RUN_NAME/explanations"
    echo "✅ Old explanations removed"
fi
if [ -d "results/$RUN_NAME/scores" ]; then
    echo "🗑️  Removing old scores to force regeneration..."
    rm -rf "results/$RUN_NAME/scores"
    echo "✅ Old scores removed"
fi
echo "💾 Keeping latents cache for reuse (using existing activations)"

# Run AutoInterp for tool-call features
# Using lmsys-chat-1m dataset (has more data for 5M tokens)
python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL_DIR" \
    --n_tokens "$N_TOKENS" \
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
    --num_examples_per_scorer_prompt 20 \
    --n_non_activating 60 \
    --min_examples 8 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo lmsys/lmsys-chat-1m \
    --dataset_name default \
    --dataset_split "train[:50%]" \
    --dataset_column conversation \
    --filter_bos \
    --verbose \
    --overwrite scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run.log" || true

# Check if results were generated (even if logging step failed)
RESULTS_SOURCE_DIR="$SCRIPT_DIR/results/$RUN_NAME"

if [ -d "$RESULTS_SOURCE_DIR" ] && [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
    echo "✅ AutoInterp scoring completed (results found)"
    
    # Count explanations
    EXPLANATION_COUNT=$(find "$RESULTS_SOURCE_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "📊 Generated $EXPLANATION_COUNT feature explanations"
    
    # Generate CSV summary directly in results folder
    if [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
        echo "📊 Generating CSV summary for tool-call feature results..."
        cd "$SCRIPT_DIR"
        
        # Generate enhanced CSV
        echo "📊 Generating enhanced CSV with scorer metrics..."
        python generate_toolcall_enhanced_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        
        if [ -f "$RESULTS_SOURCE_DIR/toolcall_results_summary_enhanced.csv" ]; then
            echo "✅ Enhanced CSV saved: $RESULTS_SOURCE_DIR/toolcall_results_summary_enhanced.csv"
            
            # Also generate basic CSV for compatibility
            python generate_results_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log" || true
            if [ -f "$RESULTS_SOURCE_DIR/results_summary.csv" ]; then
                echo "📈 Basic CSV saved: $RESULTS_SOURCE_DIR/results_summary.csv"
            fi
        else
            echo "⚠️  Enhanced CSV generation may have failed, check logs"
        fi
    fi
else
    echo "❌ AutoInterp run failed - no results found"
    exit 1
fi

# Clean up temporary file
rm -f "$SCRIPT_DIR/toolcall_features_list.txt"

# Navigate back to the original directory
cd "$SCRIPT_DIR"

echo ""
echo "🔬 Tool-call feature analysis completed! Check the results in: $RESULTS_DIR/"
echo "📊 Analyzed features from tool_features.json (Layer $LAYER, K=32, Dict Size=$DICT_SIZE)"
echo "📁 Results saved in: $RESULTS_DIR"
echo "💾 Using existing latents cache (5M tokens) - only regenerating scores/explanations"
echo ""
echo "📝 Note: Using detection scorer only for faster execution"
echo "   - Enhanced CSV includes detection F1 scores"
echo "   - Results format matches tool-call feature analysis"
echo ""

