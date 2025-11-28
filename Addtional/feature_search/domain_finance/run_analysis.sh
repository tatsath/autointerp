#!/bin/bash

# Domain Score Analysis Script
# Computes DomainScore similar to ReasonScore methodology:
# - Dataset: jyanimaulik/yahoo_finance_stockmarket_news
# - Model: meta-llama/Llama-3.1-8B-Instruct
# - SAE: /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU (400 features)
# - Layer: 19
# - Asymmetric window: 2 preceding, 3 subsequent tokens
# - α = 0.7 for entropy penalty
# - q = 0.997 quantile threshold
# - 10M tokens (target)
# - Expected ~200 top features

echo "🚀 Domain Score Analysis - Computing domain-specific features"
echo "=============================================================="
echo ""
echo "📊 Configuration:"
echo "   • Model: meta-llama/Llama-3.1-8B-Instruct"
echo "   • SAE: /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU (400 features)"
echo "   • Layer: 19"
echo "   • Dataset: jyanimaulik/yahoo_finance_stockmarket_news"
echo "   • Target tokens: 10M"
echo "   • Window: [2, 3] (asymmetric)"
echo "   • α = 0.7, q = 0.997"
echo ""

# Set GPU configuration - use GPUs 0,1,2,3 (vLLM is using 4,5,6,7)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
echo "🐍 Activating conda environment: reasoning"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate reasoning
echo "🔧 GPU Configuration: Using GPUs 0,1,2,3 (vLLM uses 4,5,6,7)"
echo ""

# Get absolute paths - updated for new structure
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Compute DomainScore
echo "📊 Step 1: Computing DomainScore"
echo "==================================="
echo ""

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Try config.yaml first, then config.json
if [ -f "$SCRIPT_DIR/config.yaml" ]; then
    CONFIG_PATH="$SCRIPT_DIR/config.yaml"
elif [ -f "$SCRIPT_DIR/config.json" ]; then
    CONFIG_PATH="$SCRIPT_DIR/config.json"
else
    echo "❌ Error: No config file found (config.yaml or config.json)"
    exit 1
fi

python "$SCRIPT_DIR/compute_score.py" \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU \
    --sae_id blocks.19.hook_resid_post \
    --config_path "$CONFIG_PATH" \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --num_chunks 1 \
    --chunk_num 0

if [ $? -ne 0 ]; then
    echo "❌ DomainScore computation failed"
    exit 1
fi

echo ""
echo "✅ DomainScore computation complete!"
echo ""

# Step 2: Generate dashboard for top features
echo "📊 Step 2: Generating Dashboard for Top Features"
echo "================================================="
echo ""

python "$BASE_DIR/domain_common/compute_dashboard.py" \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --scores_dir "$SCRIPT_DIR/results" \
    --sae_id blocks.19.hook_resid_post \
    --num_features 200 \
    --n_samples 10000 \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --output_dir "$SCRIPT_DIR/results/dashboards"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dashboard generation complete!"
else
    echo "⚠️  Dashboard generation failed (but scores are available)"
fi

echo ""
echo "🎉 Domain Score analysis complete!"
echo ""
echo "📁 Results location:"
echo "   • Scores: $SCRIPT_DIR/results/"
echo "   • Top features: $SCRIPT_DIR/results/top_features.pt"
echo "   • Dashboard: $SCRIPT_DIR/results/dashboards/features-200.html"
echo ""
