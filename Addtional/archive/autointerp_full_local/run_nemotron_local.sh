#!/bin/bash

# AutoInterp Financial News Analysis - Nemotron with Local TXT File
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENDPOINT_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp"

BASE_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
NEMOTRON_SAE_MODEL_DIR="$ENDPOINT_DIR/nemotron_sae_converted"
FEATURES_SUMMARY_PATH="$ENDPOINT_DIR/nemotron_finance_features/top_finance_features_summary.txt"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=20000000
LAYER=28
DICT_SIZE=35840
NUM_FEATURES_TO_RUN=100
RUN_NAME="nemotron_local_finance_run"
PROMPT_CONFIG_FILE="$SCRIPT_DIR/prompts_finance.yaml"

# Local TXT file path - update this to your finance TXT file
LOCAL_FINANCE_TXT="$SCRIPT_DIR/data/finance_tokens.txt"

RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"
mkdir -p "$SCRIPT_DIR/data"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if local TXT file exists, create sample if not
if [ ! -f "$LOCAL_FINANCE_TXT" ]; then
    echo "Local finance TXT file not found at $LOCAL_FINANCE_TXT"
    echo "Creating sample file..."
    python3 << 'EOF'
import os
finance_texts = [
    "The Federal Reserve raised interest rates by 0.25% to combat inflation.",
    "Stock prices surged 15% following the merger announcement between two tech giants.",
    "The company reported quarterly earnings of $2.5 billion, beating analyst expectations.",
    "Bank of America's loan loss provisions increased by $500 million this quarter.",
    "Tesla's market capitalization reached $800 billion after strong delivery numbers.",
    "Gold prices hit a new record high amid economic uncertainty.",
    "The Dow Jones Industrial Average closed at 35,000 points today.",
    "Cryptocurrency markets experienced significant volatility this week.",
    "The housing market shows signs of cooling after years of growth.",
    "Oil prices dropped 3% following OPEC production increase announcement.",
] * 60  # Repeat to get ~600 rows
os.makedirs(os.path.dirname("$LOCAL_FINANCE_TXT"), exist_ok=True)
with open("$LOCAL_FINANCE_TXT", "w") as f:
    f.write("\n".join(finance_texts))
print(f"Created sample finance TXT file with {len(finance_texts)} lines")
EOF
fi

if [ ! -d "$NEMOTRON_SAE_MODEL_DIR/layers.$LAYER" ]; then
    echo "Converting SAE..."
    cd "$ENDPOINT_DIR"
    python convert_nemotron_sae_for_autointerp.py
    if [ $? -ne 0 ]; then
        echo "Failed to convert SAE"
        exit 1
    fi
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

# Extract top features from summary file
cd "$ENDPOINT_DIR"
python3 -c "
import re

with open('$FEATURES_SUMMARY_PATH', 'r') as f:
    lines = f.readlines()

features = []
pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
for line in lines:
    match = pattern.match(line)
    if match:
        features.append(int(match.group(1)))
        if len(features) >= $NUM_FEATURES_TO_RUN:
            break

with open('nemotron_finance_news_features_list.txt', 'w') as f:
    f.write(' '.join(map(str, features)))
"

FEATURE_LIST=$(cat "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt")

if [ -z "$FEATURE_LIST" ]; then
    echo "Failed to extract features from summary file"
    exit 1
fi

cd "$SCRIPT_DIR"

if [ -d "results/$RUN_NAME/explanations" ]; then
    rm -rf "results/$RUN_NAME/explanations"
fi

# Use local file instead of HuggingFace dataset
python -m autointerp_full_local \
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
    --num_examples_per_scorer_prompt 15 \
    --n_non_activating 60 \
    --min_examples 8 \
    --pipeline_num_proc 4 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo "$LOCAL_FINANCE_TXT" \
    --dataset_split "train" \
    --dataset_column "text" \
    --filter_bos \
    --verbose \
    --prompt_override \
    --prompt_config_file "$PROMPT_CONFIG_FILE" \
    --overwrite scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run.log" || true

RESULTS_SOURCE_DIR="$SCRIPT_DIR/results/$RUN_NAME"

if [ -d "$RESULTS_SOURCE_DIR" ] && [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
    EXPLANATION_COUNT=$(find "$RESULTS_SOURCE_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "Generated $EXPLANATION_COUNT feature explanations"
    
    # Run analysis script
    if [ -f "$SCRIPT_DIR/analyze_feature_activations.py" ]; then
        echo "Running feature activation analysis..."
        python "$SCRIPT_DIR/analyze_feature_activations.py" \
            "$RESULTS_SOURCE_DIR" \
            --base-model "$BASE_MODEL" \
            --hookpoint "layers.$LAYER" \
            --features-from-list "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt" \
            --output-dir "$RESULTS_SOURCE_DIR/feature_analysis" \
            2>&1 | tee "$RESULTS_SOURCE_DIR/feature_analysis.log"
    fi
else
    echo "AutoInterp run failed - no results found"
    exit 1
fi

rm -f "$ENDPOINT_DIR/nemotron_finance_news_features_list.txt"
echo "Results saved in: $RESULTS_SOURCE_DIR"
