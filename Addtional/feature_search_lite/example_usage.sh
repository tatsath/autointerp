#!/bin/bash

# AutoInterp Lite - Simple Example Usage
# This script runs a complete analysis with top 10 features and LLM labeling using local models

echo "🚀 AutoInterp Lite - Domain-Agnostic Analysis Example"
echo "====================================================="
echo "🔍 Running Complete Analysis:"
echo "  • Top 10 most specialized features"
echo "  • LLM labeling with local model"
echo "  • Domain-specific vs General text comparison"
echo ""


python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "✅ Analysis completed!"
echo "📁 Check the results/ directory for output files:"
echo "   - results/analysis_YYYYMMDD_HHMMSS/features_layer16.csv (feature analysis with labels)"
echo "   - results/analysis_YYYYMMDD_HHMMSS/summary_layer16.json (analysis summary)"
echo "   - Each analysis creates a unique timestamped folder to avoid overwriting"