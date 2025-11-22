#!/bin/bash

echo "🚀 AutoInterp Analysis - Llama-3.1-8B-Instruct All Features"
echo "============================================================"
echo "🔍 Analyzing ALL features from Llama-3.1-8B-Instruct SAE"
echo "⚡ Using vLLM server for faster inference"
echo "📊 Processing all 400 features (Layer 19, K=32)"
echo "🎯 Using granular prompts with ENTITY/SECTOR/MACRO/EVENT/STRUCTURAL/LEXICAL classification"
echo ""

# Get script directory (autointerp_full_finance_optimized)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL_DIR="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=5000000  # 5M tokens for analysis (reduced for faster execution)
LAYER=19
DICT_SIZE=400
NUM_FEATURES_TO_RUN=400  # All features
RUN_NAME="llama31_8b_all_features_run"

# Results directory (inside results folder)
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🤖 Using vLLM provider: $EXPLAINER_PROVIDER"
echo "🌐 vLLM server URL: $EXPLAINER_API_BASE_URL"
echo "🎯 Processing ALL $NUM_FEATURES_TO_RUN features (Layer $LAYER, K=32) out of $DICT_SIZE total"
echo "📋 Using granular prompts: ENTITY | SECTOR | MACRO | EVENT | STRUCTURAL | LEXICAL"
echo "🔍 Using ContrastiveExplainer with FAISS (finance-embeddings-investopedia) for hard negatives"
echo "📊 Using 15 examples per feature + 60 non-overlapping examples for scoring"
echo "🎯 Using Detection scorer only (faster execution)"
echo "📈 Using 5M tokens for analysis (cache will be regenerated)"
echo "🎯 Optimized for higher F1: min_examples=8, n_non_activating=60, examples_per_prompt=15"
echo "⚙️  Processing all 400 features - this may take several hours"
echo "📰 Using ashraq/financial-news dataset (train[:80%]) for financial text patterns"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Check if SAE exists
if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "❌ SAE not found in expected format at: $SAE_MODEL_DIR/layers.$LAYER"
    exit 1
fi

echo "✅ SAE found at: $SAE_MODEL_DIR"
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

# Generate feature list for all features (0 to 399)
echo "🔍 Generating feature list for all $NUM_FEATURES_TO_RUN features..."
FEATURE_LIST=$(seq 0 $((NUM_FEATURES_TO_RUN - 1)) | tr '\n' ' ')

if [ -z "$FEATURE_LIST" ]; then
    echo "❌ Failed to generate feature list"
    exit 1
fi

echo "✅ Generated feature list: 0 to $((NUM_FEATURES_TO_RUN - 1))"
echo ""

echo "🔍 Running AutoInterp Analysis for Llama-3.1-8B-Instruct Features..."
echo "===================================================================="

# Navigate to the autointerp_full_finance_optimized directory
cd "$SCRIPT_DIR"

# Delete old explanations to force regeneration
if [ -d "results/$RUN_NAME/explanations" ]; then
    echo "🗑️  Removing old explanations to force regeneration with updated prompts..."
    rm -rf "results/$RUN_NAME/explanations"
    echo "✅ Old explanations removed"
fi

# Run AutoInterp for all features with financial-news dataset
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
    --num_examples_per_scorer_prompt 15 \
    --n_non_activating 60 \
    --min_examples 8 \
    --pipeline_num_proc 4 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "FinLang/finance-embeddings-investopedia" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo ashraq/financial-news \
    --dataset_name default \
    --dataset_split "train" \
    --dataset_column headline \
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
        echo "📊 Generating CSV summary for Llama-3.1-8B-Instruct results..."
        cd "$SCRIPT_DIR"
        
        # Generate enhanced CSV
        echo "📊 Generating enhanced CSV with scorer metrics..."
        python generate_nemotron_enhanced_csv.py "$RESULTS_SOURCE_DIR" --output_name "llama31_8b_results_summary_enhanced.csv" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        
        if [ -f "$RESULTS_SOURCE_DIR/llama31_8b_results_summary_enhanced.csv" ]; then
            echo "✅ Enhanced CSV saved: $RESULTS_SOURCE_DIR/llama31_8b_results_summary_enhanced.csv"
            
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

echo ""
echo "🔬 Llama-3.1-8B-Instruct feature analysis completed! Check the results in: $RESULTS_SOURCE_DIR/"
echo "📊 Analyzed all $NUM_FEATURES_TO_RUN features (out of $DICT_SIZE total) from the Llama-3.1-8B-Instruct SAE model (Layer $LAYER, K=32)"
echo "📁 Results saved in: $RESULTS_SOURCE_DIR"
echo "📄 CSV files generated in results folder:"
echo "   - llama31_8b_results_summary_enhanced.csv"
echo "   - results_summary.csv (if generated)"
echo ""
echo "📝 Note: Using detection scorer only for faster execution"
echo "   - Enhanced CSV includes detection F1 scores"
echo "   - Results format matches topk_sae_results_summary_enhanced.csv"
echo ""

