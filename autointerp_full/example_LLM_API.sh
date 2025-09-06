#!/bin/bash

# AutoInterp Full - LLM API Example
# Usage: ./example_LLM_API.sh
# Output: runs/llm_api_example/

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"  # Base language model (Llama-2-7B)
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"  # Path to SAE model files
FEATURES="27,133,220,17,333"  # Specific feature numbers to analyze

# Run Delphi interpretability analysisWhy do we need to have two
python -m autointerp_full \
  "$MODEL" \  # Use the base model
  "$SAE_PATH" \  # Use the SAE model path
  --n_tokens 100000 \  # Process 100K tokens for analysis
  --feature_num "$FEATURES" \  # Analyze specific features (27,133,220,17,333)
  --hookpoints layers.16 \  # Extract features from layer 16
  --scorers detection \  # Use detection scoring (F1, precision, recall)
  --explainer_model "openai/gpt-3.5-turbo" \  # Use GPT-3.5 for explanations
  --explainer_provider "openrouter" \  # Use OpenRouter API
  --explainer_model_max_len 512 \  # Max length for explainer model
  --num_gpus 4 \  # Use 4 GPUs for processing
  --num_examples_per_scorer_prompt 1 \  # Number of examples per prompt
  --n_non_activating 50 \  # Use 50 non-activating examples
  --non_activating_source "FAISS" \  # Use FAISS for contrastive examples
  --faiss_embedding_model "sentence-transformers/all-MiniLM-L6-v2" \  # FAISS embedding model
  --faiss_embedding_cache_dir ".embedding_cache" \  # FAISS cache directory
  --faiss_embedding_cache_enabled \  # Enable FAISS caching
  --dataset_repo wikitext \  # Use WikiText dataset
  --dataset_name wikitext-103-raw-v1 \  # Specific dataset name
  --dataset_split "train[:1%]" \  # Use 1% of training data
  --filter_bos \  # Filter beginning-of-sequence tokens
  --name "llm_api_example"  # Name for this run

# Output Location: runs/llm_api_example/
# - explanations/: Human-readable feature explanations
# - scores/detection/: F1 scores and metrics
# - latents/: Cached model activations
# - run_config.json: Configuration used
