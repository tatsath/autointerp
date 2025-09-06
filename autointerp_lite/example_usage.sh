#!/bin/bash

# AutoInterp Lite - Example Usage Scripts
# This script demonstrates various ways to use the flexible command-line interface

echo "🚀 AutoInterp Lite - Example Usage"
echo "=================================="

# Check if sample text files exist
if [ ! -f "financial_texts.txt" ] || [ ! -f "general_texts.txt" ]; then
    echo "❌ Sample text files not found!"
    echo "💡 Make sure financial_texts.txt and general_texts.txt exist in the current directory"
    exit 1
fi

echo ""
echo "1️⃣  Basic Analysis (Top 10 Features, No Labeling)"
echo "------------------------------------------------"
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10

echo ""
echo "2️⃣  Analysis with LLM Labeling (Offline Model)"
echo "----------------------------------------------"
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct"

echo ""
echo "3️⃣  Analysis with Custom Prompt"
echo "------------------------------"
python run_analysis.py \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "meta-llama/Llama-2-7b-chat-hf" \
    --prompt_file core/sample_labeling_prompt.txt

echo ""
echo "4️⃣  Analysis with OpenRouter API (if API key is set)"
echo "---------------------------------------------------"
if [ -n "$OPENROUTER_API_KEY" ]; then
    python run_analysis.py \
        --domain_data financial_texts.txt \
        --general_data general_texts.txt \
        --top_n 10 \
        --enable_labeling \
        --labeling_provider openrouter \
        --labeling_model "openai/gpt-3.5-turbo"
else
    echo "⚠️  OPENROUTER_API_KEY not set, skipping API example"
fi

echo ""
echo "5️⃣  Using HuggingFace Models"
echo "---------------------------"
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "EleutherAI/sae-llama-3-8b-32x" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10

echo ""
echo "✅ All examples completed!"
echo "📁 Check the results/ directory for output files:"
echo "   - results/features_layer16.csv (feature analysis results)"
echo "   - results/summary_layer16.json (analysis summary)"
