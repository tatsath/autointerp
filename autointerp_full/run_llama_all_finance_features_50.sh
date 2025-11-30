#!/bin/bash

# AutoInterp Analysis - Llama-3.1-8B-Instruct All Finance Features
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL_DIR="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=15000000
LAYER=19
DICT_SIZE=400
RUN_NAME="llama31_8b_top10_finance_features"
PROMPT_CONFIG_FILE="$SCRIPT_DIR/prompts_finance.yaml"

# Load top 10 feature IDs from CSV file
FEATURE_CSV_FILE="/home/nvidia/Documents/Hariom/autointerp/search_autointerp/llama_finance_feature.csv"
if [ -f "$FEATURE_CSV_FILE" ]; then
    # Extract top 10 feature IDs from CSV (skip header)
    FEATURE_LIST=$(python3 -c "
import csv
with open('$FEATURE_CSV_FILE', 'r') as f:
    reader = csv.DictReader(f)
    feature_ids = [row['feature_id'] for row in reader][:10]
    print(' '.join(feature_ids))
")
    echo "Loaded top 10 finance features from CSV"
    echo "Features: $FEATURE_LIST"
else
    echo "Error: Feature CSV file not found: $FEATURE_CSV_FILE"
    exit 1
fi

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

if [ ! -d "$SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "SAE not found at $SAE_MODEL_DIR/layers.$LAYER"
    exit 1
fi

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

if ! curl -s "$EXPLAINER_API_BASE_URL/models" > /dev/null 2>&1; then
    echo "vLLM server is not running at $EXPLAINER_API_BASE_URL"
    echo "Please start vLLM server first: bash start_vllm_server_72b.sh"
    exit 1
fi

if [ -z "$FEATURE_LIST" ]; then
    echo "Failed to generate feature list"
    exit 1
fi

cd "$SCRIPT_DIR"

# Remove old results to force regeneration
if [ -d "results/$RUN_NAME" ]; then
    echo "Removing old results to regenerate..."
    rm -rf "results/$RUN_NAME"
fi

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL_DIR" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 1024 \
    --batch_size 1 \
    --feature_num $FEATURE_LIST \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer np_max_act \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
    --explainer_model_max_len 8192 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 50 \
    --n_non_activating 200 \
    --min_examples 30 \
    --n_examples_train 80 \
    --n_examples_test 120 \
    --pipeline_num_proc 4 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
    --dataset_name default \
    --dataset_split "train[:10%]" \
    --dataset_column text \
    --filter_bos \
    --verbose \
    --prompt_override \
    --prompt_config_file "$PROMPT_CONFIG_FILE" \
    --overwrite cache scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run.log" || true

RESULTS_SOURCE_DIR="$SCRIPT_DIR/results/$RUN_NAME"

if [ -d "$RESULTS_SOURCE_DIR" ] && [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
    EXPLANATION_COUNT=$(find "$RESULTS_SOURCE_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "Generated $EXPLANATION_COUNT feature explanations"
    
    if [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
        cd "$SCRIPT_DIR"
        python generate_nemotron_enhanced_csv.py "$RESULTS_SOURCE_DIR" --output_name "llama31_8b_top10_finance_results_summary_enhanced.csv" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        python generate_results_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log" || true
        
        # Extract top sentences for all features
        echo "Extracting top activating and non-activating sentences..."
        python extract_top_sentences.py "$RESULTS_SOURCE_DIR" \
            --hookpoint "layers.$LAYER" \
            --base_model "$BASE_MODEL" \
            --top_k_positive 20 \
            --top_k_negative 10 \
            2>&1 | tee -a "$RESULTS_SOURCE_DIR/sentence_extraction.log" || true
        
        # Show top tokens for all features
        echo "Extracting top activating tokens..."
        for feature_id in $FEATURE_LIST; do
            python show_top_tokens.py "$RESULTS_SOURCE_DIR" \
                --hookpoint "layers.$LAYER" \
                --base_model "$BASE_MODEL" \
                --feature_id "$feature_id" \
                --top_k 30 \
                2>&1 | tee -a "$RESULTS_SOURCE_DIR/top_tokens.log" || true
        done
    fi
else
    echo "AutoInterp run failed - no results found"
    exit 1
fi

echo "Results saved in: $RESULTS_SOURCE_DIR"

