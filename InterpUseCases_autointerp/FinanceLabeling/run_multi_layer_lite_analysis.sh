#!/bin/bash

# Multi-Layer AutoInterp Lite Analysis
# This script runs AutoInterp Lite analysis across multiple layers to find top 10 features
# Usage: ./run_multi_layer_lite_analysis.sh

echo "🚀 Multi-Layer AutoInterp Lite Analysis"
echo "======================================="
echo "🔍 Running analysis across multiple layers:"
echo "  • Layers: 4, 10, 16, 22, 28"
echo "  • Top 10 features per layer"
echo "  • LLM labeling with local model"
echo "  • Financial domain analysis"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
DOMAIN_DATA="../autointerp_lite/financial_texts.txt"
GENERAL_DATA="../autointerp_lite/general_texts.txt"
LABELING_MODEL="Qwen/Qwen2.5-7B-Instruct"
TOP_N=10

# Layers to analyze
LAYERS=(4 10 16 22 28)

# Create results directory
RESULTS_DIR="multi_layer_lite_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Run analysis for each layer
for layer in "${LAYERS[@]}"; do
    echo "🔍 Analyzing Layer $layer..."
    echo "----------------------------------------"
    
    # Run AutoInterp Lite for this layer
    cd ../autointerp_lite
    python run_analysis.py \
        --base_model "$BASE_MODEL" \
        --sae_model "$SAE_MODEL" \
        --domain_data "$DOMAIN_DATA" \
        --general_data "$GENERAL_DATA" \
        --top_n "$TOP_N" \
        --layer_idx "$layer" \
        --enable_labeling \
        --labeling_model "$LABELING_MODEL" \
        --output_dir "../use_cases/$RESULTS_DIR"
    
    # Check if analysis was successful
    if [ $? -eq 0 ]; then
        echo "✅ Layer $layer analysis completed successfully"
        
        # Find the most recent results directory for this layer
        LATEST_RESULT=$(ls -t results/analysis_*/features_layer${layer}.csv 2>/dev/null | head -1)
        if [ -n "$LATEST_RESULT" ]; then
            # Copy results to our organized directory
            cp "$LATEST_RESULT" "../use_cases/$RESULTS_DIR/features_layer${layer}.csv"
            echo "📋 Results copied to: $RESULTS_DIR/features_layer${layer}.csv"
        fi
    else
        echo "❌ Layer $layer analysis failed"
    fi
    
    echo ""
    cd ../use_cases
done

echo "🎯 Multi-Layer Analysis Summary"
echo "==============================="
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Files created:"
for layer in "${LAYERS[@]}"; do
    if [ -f "$RESULTS_DIR/features_layer${layer}.csv" ]; then
        echo "   ✅ features_layer${layer}.csv"
    else
        echo "   ❌ features_layer${layer}.csv (failed)"
    fi
done

echo ""
echo "📋 Next Steps:"
echo "1. Review the top 10 features for each layer"
echo "2. Run: ./run_multi_layer_full_analysis.sh"
echo "3. Compare results across layers"
echo ""
echo "✅ Multi-layer AutoInterp Lite analysis completed!"
