#!/bin/bash

# AutoInterp Full - LLM API Example
# Usage: ./example_LLM_API.sh
# Output: runs/llm_api_example/

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"  # Base language model (Llama-2-7B)
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"  # Path to SAE model files

# Feature numbers to analyze - modify this line to change features
FEATURES="27 133 220"

# Set environment variables (same as working Delphi script)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting AutoInterp Full Analysis"
echo "📊 Features to analyze: $FEATURES"
echo "🔧 Model: $MODEL"
echo "📁 SAE Path: $SAE_PATH"
echo "⏱️  This may take several minutes..."

# Run AutoInterp Full interpretability analysis
python -m autointerp_full \
  "$MODEL" \
  "$SAE_PATH" \
  --n_tokens 20000 \
  --feature_num $FEATURES \
  --hookpoints layers.16 \
  --scorers detection \
  --explainer_model "openai/gpt-3.5-turbo" \
  --explainer_provider "openrouter" \
  --explainer_model_max_len 512 \
  --num_gpus 4 \
  --num_examples_per_scorer_prompt 1 \
  --n_non_activating 5 \
  --min_examples 2 \
  --non_activating_source "FAISS" \
  --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
  --faiss_embedding_cache_dir ".embedding_cache" \
  --faiss_embedding_cache_enabled \
  --dataset_repo wikitext \
  --dataset_name wikitext-103-raw-v1 \
  --dataset_split "train[:1%]" \
  --filter_bos \
  --verbose \
  --name "llm_api_example"

echo ""
echo "✅ Analysis Complete!"
echo "📁 Results saved to: results/llm_api_example/"
echo "📊 Check explanations/ and scores/ directories for results"

# Generate CSV summary if results exist
if [ -d "results/llm_api_example/explanations" ] && [ "$(ls -A results/llm_api_example/explanations)" ]; then
    echo ""
    echo "📋 Generating CSV summary..."
    python generate_results_csv.py results/llm_api_example
else
    echo ""
    echo "⚠️  No explanations found - skipping CSV generation"
fi

# Output Location: results/llm_api_example/
# - explanations/: Human-readable feature explanations
# - scores/detection/: F1 scores and metrics
# - latents/: Cached model activations
# - run_config.json: Configuration used
# - results_summary.csv: CSV summary of all results
