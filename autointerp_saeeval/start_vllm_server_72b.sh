#!/bin/bash

# Script to start vLLM server for feature labeling with Qwen 72B model
# Uses GPUs 6 and 7 by default
# Usage: bash start_vllm_server_72b.sh

EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
VLLM_PORT=8002
GPU_LIST="6,7"
GPU_COUNT=2

echo "🚀 Starting vLLM Server for Feature Labeling (72B Model)"
echo "========================================================="
echo ""
echo "Model: $EXPLAINER_MODEL"
echo "Port: $VLLM_PORT"
echo "GPUs: $GPU_LIST"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Check if server already running
API_URL="http://localhost:${VLLM_PORT}/v1"
if curl -s "$API_URL/models" > /dev/null 2>&1; then
    echo "⚠️  vLLM server already running at $API_URL"
    echo "   Stopping existing server..."
    pkill -f "vllm.entrypoints.openai.api_server.*port.*${VLLM_PORT}"
    sleep 3
    if curl -s "$API_URL/models" > /dev/null 2>&1; then
        echo "❌ Failed to stop existing server"
        exit 1
    fi
    echo "✅ Existing server stopped"
    echo ""
fi

# Check GPU availability
echo "🔍 Checking GPU availability..."
for gpu in 6 7; do
    info=$(nvidia-smi -i $gpu --query-gpu=memory.used,utilization.gpu,memory.free \
        --format=csv,noheader,nounits 2>/dev/null)
    mem_used=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
    gpu_util=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
    mem_free=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
    printf "   GPU %s: %s MB used, %s MB free, %s%% util\n" \
        "$gpu" "$mem_used" "$mem_free" "$gpu_util"
done
echo ""

# Start vLLM server with tensor parallelism
echo "🚀 Starting vLLM server on GPUs: $GPU_LIST"
echo "   Model: $EXPLAINER_MODEL"
echo "   Tensor parallel size: $GPU_COUNT"
echo "   Logs will be written to: /tmp/vllm_server_${VLLM_PORT}.log"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_LIST python -m vllm.entrypoints.openai.api_server \
    --model "$EXPLAINER_MODEL" \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size $GPU_COUNT 2>&1 | tee /tmp/vllm_server_${VLLM_PORT}.log




