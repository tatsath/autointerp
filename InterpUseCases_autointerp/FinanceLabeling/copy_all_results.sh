#!/bin/bash

# Copy All Results to FinanceLabeling Directory
# This script ensures all analysis results are copied to the FinanceLabeling directory

echo "🔄 Copying all results to FinanceLabeling directory..."
echo "=================================================="

# Create results directories if they don't exist
mkdir -p "multi_layer_full_results"
mkdir -p "single_layer_full_results"

# Copy from autointerp_full results directory
AUTOINTERP_RESULTS_DIR="../../autointerp_full/results"

if [ -d "$AUTOINTERP_RESULTS_DIR" ]; then
    echo "📁 Found autointerp_full results directory"
    
    # Copy multi-layer results
    for layer in 4 10 16 22 28; do
        if [ -d "$AUTOINTERP_RESULTS_DIR/multi_layer_full_layer${layer}" ]; then
            echo "📋 Copying multi_layer_full_layer${layer}..."
            cp -r "$AUTOINTERP_RESULTS_DIR/multi_layer_full_layer${layer}" "multi_layer_full_results/"
        fi
        
        if [ -d "$AUTOINTERP_RESULTS_DIR/single_layer_full_layer${layer}" ]; then
            echo "📋 Copying single_layer_full_layer${layer}..."
            cp -r "$AUTOINTERP_RESULTS_DIR/single_layer_full_layer${layer}" "single_layer_full_results/"
        fi
    done
    
    # Copy any CSV summaries
    for csv_file in "$AUTOINTERP_RESULTS_DIR"/*.csv; do
        if [ -f "$csv_file" ]; then
            echo "📈 Copying CSV: $(basename "$csv_file")"
            cp "$csv_file" "multi_layer_full_results/"
        fi
    done
    
else
    echo "⚠️  autointerp_full results directory not found: $AUTOINTERP_RESULTS_DIR"
fi

echo ""
echo "✅ Copy operation completed!"
echo "📊 All results are now in the FinanceLabeling directory:"
echo "   • multi_layer_full_results/"
echo "   • single_layer_full_results/"
echo "   • multi_layer_lite_results/"

# Show what we have
echo ""
echo "📁 Current contents:"
ls -la
