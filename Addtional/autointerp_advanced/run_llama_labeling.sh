#!/bin/bash
# Sample script for running feature labeling pipeline with Llama model
# Usage: bash run_llama_labeling.sh [feature_numbers] [sae_path] [sae_id]
#   feature_numbers: Comma-separated list of feature indices (e.g., "11,313,251") or path to JSON file

set -e

# Default values (can be overridden by arguments or environment variables)
# Using first 5 features by default
FEATURE_INPUT="${1:-11,313,251,165,28}"
SAE_PATH="${2:-/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU}"
SAE_ID="${3:-blocks.19.hook_resid_post}"

# Optional parameters (edit these or set as environment variables)
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
DATASET_PATH="${DATASET_PATH:-jyanimaulik/yahoo_finance_stockmarket_news}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
N_SAMPLES="${N_SAMPLES:-1000}"
MAX_EXAMPLES="${MAX_EXAMPLES:-20}"

# Activate conda environment
echo ">>> Activating conda environment: sae"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sae

# Run the pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Convert comma-separated feature numbers to Python list
if [[ "$FEATURE_INPUT" == *","* ]]; then
    # It's comma-separated numbers
    FEATURE_LIST="[$(echo $FEATURE_INPUT | sed 's/,/, /g')]"
elif [[ -f "$FEATURE_INPUT" ]]; then
    # It's a file path
    FEATURE_LIST="'$FEATURE_INPUT'"
else
    # Single number or space-separated
    FEATURE_LIST="[$(echo $FEATURE_INPUT | sed 's/ /, /g')]"
fi

python -c "
from main.run_labeling import run_labeling
run_labeling(
    feature_indices=$FEATURE_LIST,
    model_path='$MODEL_PATH',
    sae_path='$SAE_PATH',
    dataset_path='$DATASET_PATH',
    sae_id='$SAE_ID',
    output_dir='$OUTPUT_DIR',
    n_samples=$N_SAMPLES,
    max_examples_per_feature=$MAX_EXAMPLES
)
"

echo ">>> Done!"
