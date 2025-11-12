#!/bin/bash

echo "🚀 AutoInterp Analysis with vLLM Provider"
echo "=========================================="

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL="/path/to/your/sae/model"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_URL="http://localhost:8002/v1"
N_TOKENS=200000
LAYER=19

# Check vLLM server status
echo "🔍 Checking vLLM server status..."
if curl --output /dev/null --silent --head --fail "$VLLM_URL/models"; then
    echo "✅ vLLM server is running at $VLLM_URL"
else
    echo "❌ vLLM server is NOT running at $VLLM_URL"
    echo "Please start vLLM server first:"
    echo "python -m vllm.entrypoints.openai.api_server --model $EXPLAINER_MODEL --port 8002"
    exit 1
fi

# Run AutoInterp analysis
echo "🔍 Running AutoInterp Analysis with vLLM Provider..."

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --feature_num 0 1 2 3 4 \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider vllm \
    --explainer_api_base_url "$VLLM_URL" \
    --explainer_model_max_len 4096 \
    --num_examples_per_scorer_prompt 10 \
    --n_non_activating 20 \
    --min_examples 1 \
    --non_activating_source FAISS \
    --verbose \
    --name vllm_run

echo "✅ Analysis completed! Check results in: results/vllm_run/"
