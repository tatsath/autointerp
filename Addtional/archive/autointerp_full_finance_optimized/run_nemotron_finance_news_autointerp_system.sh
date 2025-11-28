#!/bin/bash

echo "🚀 AutoInterp Financial News Analysis - Nemotron Top 5 Features (System Prompts)"
echo "=================================================================================="
echo "🔍 Analyzing top 5 features from Nemotron SAE with system prompts"
echo "⚡ Using vLLM server for faster inference"
echo "📊 Processing top 5 features (Layer 28, K=64)"
echo "🎯 Using system prompts with contrastive search (FAISS)"
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
N_TOKENS=3000000  # 3M tokens for analysis
LAYER=28
DICT_SIZE=35840
NUM_FEATURES_TO_RUN=5  # Top 5 features for testing system prompts
RUN_NAME="nemotron_finance_news_run_system"

# Results directory (inside results folder)
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🤖 Using vLLM provider: $EXPLAINER_PROVIDER"
echo "🌐 vLLM server URL: $EXPLAINER_API_BASE_URL"
echo "🎯 Processing top $NUM_FEATURES_TO_RUN features (Layer $LAYER, K=64) out of $DICT_SIZE total"
echo "📋 Using system prompts with contrastive search (FAISS)"
echo "🔍 Using ContrastiveExplainer with FAISS (finance-embeddings-investopedia) for hard negatives"
echo "📊 Using 10 examples per feature + 50 non-overlapping examples for scoring"
echo "🎯 Using Detection scorer only (faster execution)"
echo "📈 Using 3M tokens for analysis (cache will be regenerated)"
echo "🎯 Optimized for high F1 scores: min_examples=5, n_non_activating=50, examples_per_prompt=10"
echo "📰 Using ashraq/financial-news dataset for financial text patterns"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Check if SAE is converted
if [ ! -d "$NEMOTRON_SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "⚠️  SAE not found in expected format. Converting..."
    cd "$ENDPOINT_DIR"
    python convert_nemotron_sae_for_autointerp.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to convert SAE"
        exit 1
    fi
    echo ""
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

echo "🔍 Running AutoInterp Financial News Analysis for Nemotron Features..."
echo "====================================================================="

# Navigate to the autointerp_full_finance_optimized directory
cd "$SCRIPT_DIR"

# Delete old explanations to force regeneration every time
if [ -d "results/$RUN_NAME/explanations" ]; then
    echo "🗑️  Removing old explanations to force regeneration with updated prompts..."
    rm -rf "results/$RUN_NAME/explanations"
    echo "✅ Old explanations removed"
fi

# Run AutoInterp for Nemotron features with financial-news dataset (with FAISS/contrastive)
python -m autointerp_full \
    "$BASE_MODEL" \
    "$NEMOTRON_SAE_MODEL_DIR" \
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
        echo "📊 Generating CSV summary for Nemotron financial news results..."
        cd "$SCRIPT_DIR"
        
        # Generate enhanced CSV directly (handles backbone.layers format and works without fuzz)
        echo "📊 Generating enhanced CSV with scorer metrics..."
        python generate_nemotron_enhanced_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        
        if [ -f "$RESULTS_SOURCE_DIR/nemotron_finance_results_summary_enhanced.csv" ]; then
            echo "✅ Enhanced CSV saved: $RESULTS_SOURCE_DIR/nemotron_finance_results_summary_enhanced.csv"
            
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
rm -f "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt"

echo ""
echo "🔬 Nemotron financial news feature analysis completed! Check the results in: $RESULTS_SOURCE_DIR/"
echo "📊 Analyzed top $NUM_FEATURES_TO_RUN features (out of $DICT_SIZE total) from the Nemotron SAE model (Layer $LAYER, K=64)"
echo "📁 Results saved in: $RESULTS_SOURCE_DIR"
echo "📄 CSV files generated in results folder:"
echo "   - nemotron_finance_results_summary_enhanced.csv"
echo "   - results_summary.csv (if generated)"
echo ""
echo "📝 Note: Using detection scorer only for faster execution"
echo "   - Enhanced CSV includes detection F1 scores"
echo "   - Using system prompts with contrastive search (FAISS)"
echo ""

