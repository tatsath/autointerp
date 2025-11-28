#!/bin/bash

# AutoInterp Analysis - Goodfire Llama-3.1-8B-Instruct-SAE-l19 (5 Features Test)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL_DIR="$SCRIPT_DIR/goodfire_sae_converted"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=15000000
LAYER=19
DICT_SIZE=65536  # Goodfire SAE dict size
NUM_FEATURES_TO_RUN=5
RUN_NAME="goodfire_llama31_8b_5_features_test"
PROMPT_CONFIG_FILE="$SCRIPT_DIR/prompts_finance.yaml"

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if SAE is converted, if not convert it
if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "SAE not found at $SAE_MODEL_DIR/layers.$LAYER"
    echo "Converting Goodfire SAE from HuggingFace..."
    python "$SCRIPT_DIR/convert_goodfire_sae.py"
    if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
        echo "Error: Failed to convert SAE"
        exit 1
    fi
fi

echo "Using converted SAE: $SAE_MODEL_DIR"

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Set HuggingFace token if available
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

if ! curl -s "$EXPLAINER_API_BASE_URL/models" > /dev/null 2>&1; then
    echo "vLLM server is not running at $EXPLAINER_API_BASE_URL"
    echo "Please start vLLM server first: bash start_vllm_server_72b.sh"
    exit 1
fi

FEATURE_LIST=$(seq 0 $((NUM_FEATURES_TO_RUN - 1)) | tr '\n' ' ')

if [ -z "$FEATURE_LIST" ]; then
    echo "Failed to generate feature list"
    exit 1
fi

cd "$SCRIPT_DIR"

if [ -d "results/$RUN_NAME/explanations" ]; then
    rm -rf "results/$RUN_NAME/explanations"
fi

echo "Starting AutoInterp analysis..."
echo "  Base Model: $BASE_MODEL"
echo "  SAE Model: $SAE_MODEL_DIR"
echo "  Layer: $LAYER"
echo "  Features: $FEATURE_LIST"
echo "  Run Name: $RUN_NAME"
echo ""

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
    --num_examples_per_scorer_prompt 25 \
    --n_non_activating 100 \
    --min_examples 30 \
    --pipeline_num_proc 4 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo lmsys/lmsys-chat-1m \
    --dataset_name default \
    --dataset_split "train[:10%]" \
    --dataset_column conversation \
    --filter_bos \
    --verbose \
    --overwrite scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run.log" || true

RESULTS_SOURCE_DIR="$SCRIPT_DIR/results/$RUN_NAME"

if [ -d "$RESULTS_SOURCE_DIR" ] && [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
    EXPLANATION_COUNT=$(find "$RESULTS_SOURCE_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "Generated $EXPLANATION_COUNT feature explanations"
    
    if [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
        cd "$SCRIPT_DIR"
        python generate_nemotron_enhanced_csv.py "$RESULTS_SOURCE_DIR" --output_name "goodfire_llama31_8b_results_summary_enhanced.csv" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        python generate_results_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log" || true
    fi
else
    echo "AutoInterp run failed - no results found"
    exit 1
fi

echo "Results saved in: $RESULTS_SOURCE_DIR"

