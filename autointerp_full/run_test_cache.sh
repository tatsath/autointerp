#!/bin/bash

# AutoInterp Test - Cache Reuse and Prompt Override
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL_DIR="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=500000
LAYER=19
NUM_FEATURES_TO_RUN=10
RUN_NAME="test_cache_prompts_run"
PROMPT_CONFIG_FILE="$SCRIPT_DIR/prompts_finance.yaml"

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "SAE not found at $SAE_MODEL_DIR/layers.$LAYER"
    exit 1
fi

if [ ! -f "$PROMPT_CONFIG_FILE" ]; then
    echo "Prompt config file not found at $PROMPT_CONFIG_FILE"
    exit 1
fi

# Set environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Generate feature list
FEATURE_LIST=$(seq 0 $((NUM_FEATURES_TO_RUN - 1)) | tr '\n' ' ')

cd "$SCRIPT_DIR"

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL_DIR" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 256 \
    --batch_size 8 \
    --feature_num $FEATURE_LIST \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
    --explainer_model_max_len 4096 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 5 \
    --n_non_activating 20 \
    --min_examples 5 \
    --pipeline_num_proc 4 \
    --non_activating_source "random" \
    --dataset_repo "EleutherAI/SmolLM2-135M-10B" \
    --dataset_split "train[:1%]" \
    --dataset_column "text" \
    --filter_bos \
    --verbose \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/test1.log" || true

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL_DIR" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 256 \
    --batch_size 8 \
    --feature_num $FEATURE_LIST \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
    --explainer_model_max_len 4096 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 5 \
    --n_non_activating 20 \
    --min_examples 5 \
    --pipeline_num_proc 4 \
    --non_activating_source "random" \
    --dataset_repo "EleutherAI/SmolLM2-135M-10B" \
    --dataset_split "train[:1%]" \
    --dataset_column "text" \
    --filter_bos \
    --verbose \
    --prompt_override \
    --prompt_config_file "$PROMPT_CONFIG_FILE" \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/test2.log" || true

if [ -d "$RESULTS_DIR/explanations" ] && [ "$(ls -A $RESULTS_DIR/explanations 2>/dev/null)" ]; then
    if [ -f "$SCRIPT_DIR/generate_nemotron_enhanced_csv.py" ]; then
        python "$SCRIPT_DIR/generate_nemotron_enhanced_csv.py" "$RESULTS_DIR" --output_name "test_results_summary_enhanced.csv" 2>&1 | tee -a "$RESULTS_DIR/csv_generation.log" || true
    fi
    if [ -f "$SCRIPT_DIR/generate_results_csv.py" ]; then
        python "$SCRIPT_DIR/generate_results_csv.py" "$RESULTS_DIR" 2>&1 | tee -a "$RESULTS_DIR/csv_generation.log" || true
    fi
    EXPLANATION_COUNT=$(find "$RESULTS_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "Generated $EXPLANATION_COUNT explanations. Results: $RESULTS_DIR"
fi

