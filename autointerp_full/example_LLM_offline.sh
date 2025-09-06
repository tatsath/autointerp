#!/bin/bash

# AutoInterp Full - LLM Offline Example
# Usage: ./example_LLM_offline.sh
# Output: runs/llm_offline_example/

# Configuration
MODEL="meta-llama/Llama-2-7b-hf"  # Base language model (Llama-2-7B)
SAE_PATH="/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"  # Path to SAE model files
FEATURES="27,133,220,17,333"  # Specific feature numbers to analyze

# Run Delphi interpretability analysis
python -m autointerp_full \
  "$MODEL" \  # Use the base model
  "$SAE_PATH" \  # Use the SAE model path
  --n_tokens 1000000 \  # Process 1M tokens for analysis
  --feature_num "$FEATURES" \  # Analyze specific features (27,133,220,17,333)
  --hookpoints layers.16 \  # Extract features from layer 16
  --scorers detection \  # Use detection scoring (F1, precision, recall)
  --explainer_model "meta-llama/Llama-2-7b-chat-hf" \  # Use Llama-2-7B-chat for explanations
  --explainer_provider "offline" \  # Use offline model (no API)
  --num_gpus 1 \  # Use 1 GPU for processing
  --explainer_model_max_len 512 \  # Max length for explainer model
  --dataset_repo wikitext \  # Use WikiText dataset
  --dataset_name wikitext-103-raw-v1 \  # Specific dataset name
  --dataset_split "train[:1%]" \  # Use 1% of training data
  --filter_bos \  # Filter beginning-of-sequence tokens
  --name "llm_offline_example"  # Name for this run

# Output Location: runs/llm_offline_example/
# - explanations/: Human-readable feature explanations
# - scores/detection/: F1 scores and metrics
# - latents/: Cached model activations
# - run_config.json: Configuration used
