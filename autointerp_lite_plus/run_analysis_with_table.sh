#!/bin/bash

# AutoInterp Lite Plus - Comprehensive Analysis with Quality Assessment Table
# This script runs the analysis and generates a detailed quality assessment table

echo "🚀 Starting AutoInterp Lite Plus Comprehensive Analysis..."
echo "=================================================================================="

# Run the comprehensive analysis
echo "📊 Running comprehensive analysis..."
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --domain_data "financial_texts.txt" \
    --general_data "general_texts.txt" \
    --top_n 5 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct" \
    --comprehensive

echo ""
echo "📋 Generating quality assessment table..."
echo "=================================================================================="

# Generate the results table
python generate_results_table.py

echo ""
echo "✅ Analysis complete! Check the results directory for detailed outputs."
echo "📁 Results saved in: results/comprehensive_analysis_*/"
