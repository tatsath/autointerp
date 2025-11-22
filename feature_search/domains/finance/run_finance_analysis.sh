#!/bin/bash

# FinanceScore Analysis Script
# Computes FinanceScore similar to ReasonScore methodology:
# - Dataset: jyanimaulik/yahoo_finance_stockmarket_news
# - Model: meta-llama/Llama-3.1-8B-Instruct
# - SAE: /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU (400 features)
# - Layer: 19
# - Asymmetric window: 2 preceding, 3 subsequent tokens
# - α = 0.7 for entropy penalty
# - q = 0.997 quantile threshold
# - 10M tokens (target)
# - Expected ~200 top features

echo "🚀 FinanceScore Analysis - Computing domain-specific features"
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

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$BASE_DIR"

# Step 1: Compute FinanceScore
echo "📊 Step 1: Computing FinanceScore"
echo "==================================="
echo ""

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$BASE_DIR/domains/finance/compute_finance_score.py" \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU \
    --sae_id blocks.19.hook_resid_post \
    --config_path "$BASE_DIR/domains/finance/config.json" \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --num_chunks 1 \
    --chunk_num 0

if [ $? -ne 0 ]; then
    echo "❌ FinanceScore computation failed"
    exit 1
fi

echo ""
echo "✅ FinanceScore computation complete!"
echo ""

# Step 2: Generate dashboard for top features
echo "📊 Step 2: Generating Dashboard for Top Features"
echo "================================================="
echo ""

python "$BASE_DIR/main/compute_dashboard.py" \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --scores_dir "$BASE_DIR/domains/finance/scores" \
    --sae_id blocks.19.hook_resid_post \
    --num_features 200 \
    --n_samples 10000 \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --output_dir "$BASE_DIR/domains/finance/dashboards"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dashboard generation complete!"
else
    echo "⚠️  Dashboard generation failed (but scores are available)"
fi

echo ""
echo "🎉 FinanceScore analysis complete!"
echo ""
echo "📁 Results location:"
echo "   • Scores: $BASE_DIR/domains/finance/scores/"
echo "   • Top features: $BASE_DIR/domains/finance/scores/top_features.pt"
echo "   • Dashboard: $BASE_DIR/domains/finance/dashboards/features-200.html"
echo ""

