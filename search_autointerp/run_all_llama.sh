#!/bin/bash
# Run complete pipeline: search → basic labeling → advanced labeling (Llama SAE)

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

echo "🚀 Llama SAE Finance Pipeline (Top 100)"
echo "========================================"
echo ""

# Step 1: Search
echo "📊 Step 1/3: Feature Search"
echo "---------------------------"
cd "1. search"
bash run_llama.sh
cd ..

# Step 2: Basic Labeling
echo ""
echo "📊 Step 2/3: Basic Labeling"
echo "---------------------------"
cd "2. autointerp_lite"
bash run_llama.sh
cd ..

# Step 3: Advanced Labeling
echo ""
echo "📊 Step 3/3: Advanced Labeling"
echo "------------------------------"
cd "3. autointerp_advance"
bash run_llama.sh
cd ..

echo ""
echo "🎉 Pipeline Complete!"
echo "===================="
echo "Results:"
echo "  • Search: results/1_search/"
echo "  • Basic Labels: results/2_labeling_lite/"
echo "  • Advanced Labels: results/3_labeling_advance/"

