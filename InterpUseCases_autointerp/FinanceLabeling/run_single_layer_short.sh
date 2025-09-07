#!/bin/bash

# Short Single-Layer AutoInterp Analysis - Process Features One by One
# Usage: ./run_single_layer_short.sh [LAYER_NUMBER]

LAYER=${1:-4}
echo "🚀 Running AutoInterp analysis for Layer $LAYER (one feature at a time)..."

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
N_TOKENS=50000

# Get top 10 features
FEATURES=$(python3 -c "
import pandas as pd
df = pd.read_csv('multi_layer_lite_results/features_layer${LAYER}.csv')
features = df['feature'].head(10).tolist() if 'feature' in df.columns else df.iloc[:10, 1].tolist()
print(' '.join(map(str, features)))
")

echo "🎯 Features: $FEATURES"

# Process features one at a time to avoid memory issues
FEATURE_ARRAY=($FEATURES)
for i in "${!FEATURE_ARRAY[@]}"; do
    FEATURE=${FEATURE_ARRAY[$i]}
    echo "🔍 Processing feature $((i+1))/10: $FEATURE"
    
    # Run analysis for this single feature
    cd ../../autointerp_full
    RUN_NAME="single_layer_short_layer${LAYER}_feature${FEATURE}"
    
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$SAE_MODEL" \
        --n_tokens "$N_TOKENS" \
        --feature_num $FEATURE \
        --hookpoints "layers.$LAYER" \
        --scorers detection \
        --explainer_model "$EXPLAINER_MODEL" \
        --explainer_provider "offline" \
        --explainer_model_max_len 1024 \
        --num_gpus 4 \
        --num_examples_per_scorer_prompt 1 \
        --n_non_activating 2 \
        --min_examples 1 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --example_ctx_len 256 \
        --dataset_repo wikitext \
        --dataset_name wikitext-103-raw-v1 \
        --dataset_split "train[:1%]" \
        --filter_bos \
        --verbose \
        --name "$RUN_NAME"
    
    # Copy results for this feature
    if [ -d "results/$RUN_NAME" ]; then
        echo "📋 Copying results for feature $FEATURE..."
        cp -r "results/$RUN_NAME" "../use_cases/FinanceLabeling/single_layer_full_results/"
    fi
    
    # Go back to FinanceLabeling directory
    cd ../use_cases/FinanceLabeling
    echo "✅ Completed feature $FEATURE"
    echo ""
done

echo "🎉 All features processed successfully!"
echo "📁 Results saved in: single_layer_full_results/"