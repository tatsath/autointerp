#!/bin/bash

# Single-Layer AutoInterp Analysis with OpenRouter API
# This script uses OpenRouter API for better explanations and processes features in a consolidated run

LAYER=4
echo "🚀 Running AutoInterp analysis for Layer $LAYER with OpenRouter API..."

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="openai/gpt-4o-mini"
N_TOKENS=10000

# Set environment variables for better performance
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1
export OPENROUTER_API_KEY="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"

# Get top 2 features for faster testing
FEATURES=$(python3 -c "
import pandas as pd
df = pd.read_csv('multi_layer_lite_results/features_layer${LAYER}.csv')
features = df['feature'].head(2).tolist() if 'feature' in df.columns else df.iloc[:2, 1].tolist()
print(' '.join(map(str, features)))
")

echo "🎯 Features: $FEATURES"
echo "🔧 Using OpenRouter API for better explanations"

# Process all features in a single run to avoid separate folders
echo "🔍 Processing all features in a single run..."
cd ../../autointerp_full
RUN_NAME="single_layer_openrouter_layer${LAYER}_consolidated"

# Convert features array to space-separated list for --feature_num
FEATURE_LIST=$(echo $FEATURES | tr ' ' ' ')

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --feature_num $FEATURE_LIST \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "openrouter" \
    --explainer_model_max_len 512 \
    --num_gpus 4 \
    --num_examples_per_scorer_prompt 1 \
    --n_non_activating 2 \
    --min_examples 1 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo jyanimaulik/yahoo_finance_stockmarket_news \
    --dataset_name default \
    --dataset_split "train[:1%]" \
    --filter_bos \
    --verbose \
    --name "$RUN_NAME"

# Copy results directly to FinanceLabeling directory
if [ -d "results/$RUN_NAME" ]; then
    echo "📋 Copying consolidated results..."
    # Remove existing results directory if it exists
    rm -rf "../use_cases/FinanceLabeling/single_layer_openrouter_results"
    # Copy the entire results directory
    cp -r "results/$RUN_NAME" "../use_cases/FinanceLabeling/single_layer_openrouter_results"
    
    # Generate CSV summary
    if [ -d "results/$RUN_NAME/explanations" ] && [ "$(ls -A results/$RUN_NAME/explanations)" ]; then
        echo "📊 Generating CSV summary..."
        python generate_results_csv.py "results/$RUN_NAME"
        if [ -f "results/$RUN_NAME/results_summary.csv" ]; then
            cp "results/$RUN_NAME/results_summary.csv" "../use_cases/FinanceLabeling/single_layer_openrouter_results/results_summary.csv"
        fi
    fi
fi

# Go back to FinanceLabeling directory
cd ../use_cases/FinanceLabeling
echo "✅ Completed consolidated analysis for all features"

echo "🎉 All features processed successfully with OpenRouter API!"
echo "📁 Results saved in: single_layer_openrouter_results/"
echo "📊 CSV summary generated for all features"

# Clean up unnecessary directories to save space
echo "🧹 Cleaning up unnecessary directories to save space..."
python3 -c "
import os
import shutil

# Clean up latents and log directories to save space
latents_dir = 'single_layer_openrouter_results/latents'
log_dir = 'single_layer_openrouter_results/log'
if os.path.exists(latents_dir):
    shutil.rmtree(latents_dir)
    print('🗑️  Removed latents directory')
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print('🗑️  Removed log directory')
print('✅ Cleanup completed')
"
