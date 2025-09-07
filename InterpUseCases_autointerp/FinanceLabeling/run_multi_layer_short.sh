#!/bin/bash

# Short Multi-Layer AutoInterp Analysis
# Usage: ./run_multi_layer_short.sh

echo "🚀 Running Multi-Layer AutoInterp Analysis..."

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="meta-llama/Llama-2-7b-chat-hf"
N_TOKENS=50000
LAYERS=(4 10 16 22 28)

# Create results directory
mkdir -p "multi_layer_full_results"

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Processing Layer $layer..."
    
    # Get top 10 features
    FEATURES=$(python3 -c "
import pandas as pd
df = pd.read_csv('multi_layer_lite_results/features_layer${layer}.csv')
features = df['feature'].head(10).tolist() if 'feature' in df.columns else df.iloc[:10, 1].tolist()
print(' '.join(map(str, features)))
")
    
    echo "🎯 Features: $FEATURES"
    
    # Run analysis
    cd ../../autointerp_full
    RUN_NAME="multi_layer_short_layer${layer}"
    
    python -m autointerp_full \
        "$BASE_MODEL" \
        "$SAE_MODEL" \
        --n_tokens "$N_TOKENS" \
        --feature_num $FEATURES \
        --hookpoints "layers.$layer" \
        --scorers detection \
        --explainer_model "$EXPLAINER_MODEL" \
        --explainer_provider "offline" \
        --explainer_model_max_len 2048 \
        --num_gpus 4 \
        --num_examples_per_scorer_prompt 1 \
        --n_non_activating 2 \
        --min_examples 1 \
        --non_activating_source "FAISS" \
        --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
        --faiss_embedding_cache_dir ".embedding_cache" \
        --faiss_embedding_cache_enabled \
        --dataset_repo wikitext \
        --dataset_name wikitext-103-raw-v1 \
        --dataset_split "train[:1%]" \
        --filter_bos \
        --verbose \
        --name "$RUN_NAME"
    
    # Copy results
    if [ $? -eq 0 ] && [ -d "results/$RUN_NAME" ]; then
        echo "✅ Layer $layer completed"
        cp -r "results/$RUN_NAME" "../use_cases/FinanceLabeling/multi_layer_full_results/"
    else
        echo "❌ Layer $layer failed"
    fi
    
    cd ../use_cases/FinanceLabeling
    echo ""
done

echo "🎯 Multi-Layer Analysis Complete!"
echo "📊 Results saved in: multi_layer_full_results/"
