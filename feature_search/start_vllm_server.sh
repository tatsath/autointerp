#!/bin/bash

# Script to start vLLM server for feature labeling
# Uses 4 GPUs: GPUs 4, 5, 6, 7 (as specified by user for better performance)
# Usage: bash scripts/start_vllm_server.sh [GPU_IDs comma-separated]

EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_PORT=8002
REQUESTED_GPUS=${1:-""}
PREFERRED_GPUS="4,5,6,7"
FALLBACK_GPUS="6,7"

echo "🚀 Starting vLLM Server for Feature Labeling"
echo "============================================="
echo ""
echo "Model: $EXPLAINER_MODEL"
echo "Port: $VLLM_PORT"
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

# Function to check if specific GPU is free
check_gpu_free() {
    local gpu_id=$1
    local min_free_memory_gb=${2:-40}
    
    nvidia-smi -i $gpu_id --query-gpu=memory.used,utilization.gpu,memory.free \
        --format=csv,noheader,nounits 2>/dev/null | \
        awk -F',' -v min_free=$((min_free_memory_gb * 1024)) '
        {
            mem_used = $1; gsub(/^[ \t]+|[ \t]+$/, "", mem_used)
            gpu_util = $2; gsub(/^[ \t]+|[ \t]+$/, "", gpu_util)
            mem_free = $3; gsub(/^[ \t]+|[ \t]+$/, "", mem_free)
            
            # Check if GPU is free: low memory used (<1GB), low utilization (<5%), enough free memory
            if (mem_used < 1024 && gpu_util < 5 && mem_free >= min_free) {
                print "free"
            } else {
                print "busy"
            }
        }'
}

# Find free GPUs if not specified
if [ -z "$REQUESTED_GPUS" ]; then
    echo "🔍 Checking for free GPUs..."
    echo "   Preferred: GPUs $PREFERRED_GPUS (4 GPUs for tensor parallelism)"
    echo "   Fallback: GPUs $FALLBACK_GPUS (2 GPUs)"
    echo "   Criteria: < 1GB memory used, < 5% GPU utilization, >= 40GB free"
    echo ""
    
    # Check preferred GPUs (4, 5, 6, 7)
    GPU4_STATUS=$(check_gpu_free 4 40)
    GPU5_STATUS=$(check_gpu_free 5 40)
    GPU6_STATUS=$(check_gpu_free 6 40)
    GPU7_STATUS=$(check_gpu_free 7 40)
    
    if [ "$GPU4_STATUS" = "free" ] && [ "$GPU5_STATUS" = "free" ] && [ "$GPU6_STATUS" = "free" ] && [ "$GPU7_STATUS" = "free" ]; then
        GPU_LIST="4,5,6,7"
        echo "✅ Using preferred GPUs: 4, 5, 6, 7 (4 GPUs for maximum performance)"
        GPU_COUNT=4
    else
        # Try fallback GPUs (6 and 7)
        if [ "$GPU6_STATUS" = "free" ] && [ "$GPU7_STATUS" = "free" ]; then
            GPU_LIST="6,7"
            echo "⚠️  Preferred GPUs not all available, using fallback: 6 and 7"
            GPU_COUNT=2
        else
            echo "❌ Neither preferred (4 GPUs) nor fallback (2 GPUs) are available!"
            echo ""
            echo "GPU Status:"
            for gpu in 4 5 6 7; do
                status=$(check_gpu_free $gpu 40)
                info=$(nvidia-smi -i $gpu --query-gpu=memory.used,utilization.gpu,memory.free \
                    --format=csv,noheader,nounits 2>/dev/null)
                mem_used=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
                gpu_util=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
                mem_free=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
                printf "  GPU %s: %s (%s MB used, %s%% util, %s MB free)\n" \
                    "$gpu" "$status" "$mem_used" "$gpu_util" "$mem_free"
            done
            exit 1
        fi
    fi
    
    # Show GPU details
    echo ""
    IFS=',' read -ra GPUS <<< "$GPU_LIST"
    for gpu in "${GPUS[@]}"; do
        info=$(nvidia-smi -i $gpu --query-gpu=memory.used,utilization.gpu,memory.free \
            --format=csv,noheader,nounits 2>/dev/null)
        mem_used=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $1); print $1}')
        gpu_util=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')
        mem_free=$(echo $info | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $3); print $3}')
        printf "   GPU %s: %s MB used, %s MB free, %s%% util\n" \
            "$gpu" "$mem_used" "$mem_free" "$gpu_util"
    done
    echo ""
else
    GPU_LIST="$REQUESTED_GPUS"
    GPU_COUNT=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
    echo "✓ Using specified GPUs: $GPU_LIST"
    echo ""
fi

# Check if server already running
API_URL="http://localhost:${VLLM_PORT}/v1"
if curl -s "$API_URL/models" > /dev/null 2>&1; then
    echo "✅ vLLM server already running at $API_URL"
    echo "   Stop it first if you want to restart"
    exit 0
fi

# Start vLLM server with tensor parallelism
echo "🚀 Starting vLLM server on GPUs: $GPU_LIST"
echo "   Tensor parallel size: $GPU_COUNT"
echo "   Logs will be written to: /tmp/vllm_server_${VLLM_PORT}.log"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_LIST python -m vllm.entrypoints.openai.api_server \
    --model "$EXPLAINER_MODEL" \
    --port $VLLM_PORT \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --tensor-parallel-size $GPU_COUNT 2>&1 | tee /tmp/vllm_server_${VLLM_PORT}.log

