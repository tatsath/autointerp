#!/bin/bash

# Single-Layer AutoInterp Full Analysis
# This script runs AutoInterp Full analysis on top 10 features from a single layer
# Usage: ./run_single_layer_full_analysis.sh [LAYER_NUMBER]
# Example: ./run_single_layer_full_analysis.sh 4

echo "🚀 Single-Layer AutoInterp Full Analysis"
echo "========================================"

# Check if layer number is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide a layer number"
    echo "Usage: ./run_single_layer_full_analysis.sh [LAYER_NUMBER]"
    echo "Available layers: 4, 10, 16, 22, 28"
    exit 1
fi

LAYER=$1

# Validate layer number
if [[ ! "$LAYER" =~ ^(4|10|16|22|28)$ ]]; then
    echo "❌ Error: Invalid layer number. Must be one of: 4, 10, 16, 22, 28"
    exit 1
fi

echo "🔍 Running detailed analysis on top features from layer $LAYER:"
echo "  • Layer: $LAYER"
echo "  • Top 10 features"
echo "  • LLM-based explanations with confidence scores"
echo "  • F1 scores, precision, and recall metrics"
echo ""

# Configuration
BASE_MODEL="meta-llama/Llama-2-7b-hf"
SAE_MODEL="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
EXPLAINER_PROVIDER="offline"
N_TOKENS=50000

# Create results directory
RESULTS_DIR="single_layer_full_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo ""

# Set environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if lite results exist
LITE_RESULTS_DIR="multi_layer_lite_results"
if [ ! -d "$LITE_RESULTS_DIR" ]; then
    echo "❌ Error: Lite results directory not found: $LITE_RESULTS_DIR"
    echo "Please run ./run_multi_layer_lite_analysis.sh first"
    exit 1
fi

echo "🔍 Analyzing Layer $LAYER with AutoInterp Full..."
echo "----------------------------------------"

# Check if lite results exist for this layer
LITE_CSV="$LITE_RESULTS_DIR/features_layer${LAYER}.csv"
if [ ! -f "$LITE_CSV" ]; then
    echo "❌ Error: No lite results found for layer $LAYER: $LITE_CSV"
    exit 1
fi

# Extract top 10 feature numbers from the CSV
echo "📋 Extracting top 10 features from: $LITE_CSV"
FEATURES=$(python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$LITE_CSV')
    # Get top 10 features, skip header if it exists
    if 'feature' in df.columns:
        features = df['feature'].head(10).tolist()
    else:
        features = df.iloc[:10, 1].tolist()  # Second column is usually feature number
    print(' '.join(map(str, features)))
except Exception as e:
    print('Error reading CSV:', e)
    exit(1)
")

if [ -z "$FEATURES" ]; then
    echo "❌ Failed to extract features from $LITE_CSV"
    exit 1
fi

echo "🎯 Features to analyze: $FEATURES"

# Run AutoInterp Full for this layer
cd ../../autointerp_full

# Create layer-specific run name
RUN_NAME="single_layer_full_layer${LAYER}"

echo "🚀 Starting AutoInterp Full analysis..."
echo "📊 Run name: $RUN_NAME"
echo "🎯 Features: $FEATURES"
echo "🔧 Hookpoint: layers.$LAYER"
echo ""

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --feature_num $FEATURES \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
        --explainer_model_max_len 4096 \
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

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo "✅ Layer $LAYER AutoInterp Full analysis completed successfully"
    
    # Copy results to our organized directory
    if [ -d "results/$RUN_NAME" ]; then
        cp -r "results/$RUN_NAME" "../use_cases/FinanceLabeling/$RESULTS_DIR/"
        echo "📋 Results copied to: $RESULTS_DIR/$RUN_NAME/"
        
        # Generate CSV summary if results exist
        if [ -d "results/$RUN_NAME/explanations" ] && [ "$(ls -A results/$RUN_NAME/explanations)" ]; then
            echo "📊 Generating CSV summary for layer $LAYER..."
            python generate_results_csv.py "results/$RUN_NAME"
            if [ -f "results/$RUN_NAME/results_summary.csv" ]; then
                cp "results/$RUN_NAME/results_summary.csv" "../use_cases/FinanceLabeling/$RESULTS_DIR/results_summary_layer${LAYER}.csv"
                echo "📈 Summary saved: $RESULTS_DIR/results_summary_layer${LAYER}.csv"
            fi
        fi
        
        # Display sample explanations to verify quality
        echo ""
        echo "🔍 Sample Explanations from Layer $LAYER:"
        echo "========================================"
        if [ -d "results/$RUN_NAME/explanations" ]; then
            for explanation_file in results/$RUN_NAME/explanations/*.txt; do
                if [ -f "$explanation_file" ]; then
                    echo "📄 $(basename "$explanation_file"):"
                    head -3 "$explanation_file" | sed 's/^/   /'
                    echo ""
                fi
            done
        fi
        
    fi
else
    echo "❌ Layer $LAYER AutoInterp Full analysis failed"
    exit 1
fi

cd ../use_cases/FinanceLabeling

echo ""
echo "🎯 Single-Layer AutoInterp Full Analysis Summary"
echo "================================================"
echo "📊 Results saved in: $RESULTS_DIR/"
echo "📁 Directory created: $RESULTS_DIR/$RUN_NAME/"
if [ -f "$RESULTS_DIR/results_summary_layer${LAYER}.csv" ]; then
    echo "📈 Summary: $RESULTS_DIR/results_summary_layer${LAYER}.csv"
fi

echo ""
echo "📋 Analysis Complete!"
echo "🔍 Check the following for detailed results:"
echo "   • explanations/: Human-readable feature explanations"
echo "   • scores/detection/: F1 scores and metrics"
echo "   • results_summary_layer${LAYER}.csv: CSV summary"
echo ""
echo "✅ Single-layer AutoInterp Full analysis completed!"
echo ""
echo "💡 Next steps:"
echo "   • Review the explanations to verify they are meaningful"
echo "   • Check the F1 scores and metrics"
echo "   • If satisfied, run the full multi-layer analysis"
