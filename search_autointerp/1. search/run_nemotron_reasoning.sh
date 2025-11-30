#!/bin/bash
# Run reasoning feature search for Nemotron model (top 50 features)

set -e

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Nemotron config
MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_sae_converted"
SAE_ID="blocks.28.hook_resid_post"
DATASET="open-thoughts/OpenThoughts-114k"
TOKENS="domains/reasoning/reasoning_tokens.json"
OUTPUT="../results/1_search"
NUM_FEATURES=50

mkdir -p "$OUTPUT"

echo "🔍 Nemotron Reasoning Search (Top $NUM_FEATURES)"
echo "================================================"
echo "Model: $MODEL"
echo "SAE: $SAE"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT"
echo ""

# Preprocess OpenThoughts dataset to create 'text' column
# The dataset has 'system' and 'conversations' fields, need to combine them
PREPROCESSED_DATASET="/tmp/openthoughts_114k_preprocessed"

if [ ! -d "$PREPROCESSED_DATASET" ]; then
    echo "Preprocessing OpenThoughts-114k dataset..."
    python3 -c "
from datasets import load_dataset, Dataset
from datasets import load_from_disk
import json
import os

print('Loading OpenThoughts-114k dataset...')
dataset = load_dataset('open-thoughts/OpenThoughts-114k', split='train', streaming=False)

def extract_text(example):
    text_parts = []
    if 'system' in example and example.get('system'):
        text_parts.append(example['system'])
    if 'conversations' in example and isinstance(example['conversations'], list):
        for conv in example['conversations']:
            if isinstance(conv, dict) and 'value' in conv:
                text_parts.append(conv['value'])
    return {'text': ' '.join(text_parts) if text_parts else ''}

print(f'Processing {len(dataset)} samples...')
dataset = dataset.map(extract_text, remove_columns=['system', 'conversations'], batched=False)
print(f'Preprocessed dataset has {len(dataset)} samples with text column')
dataset.save_to_disk('$PREPROCESSED_DATASET')
print('Saved preprocessed dataset to $PREPROCESSED_DATASET')
"
else
    echo "Using existing preprocessed dataset at $PREPROCESSED_DATASET"
fi

# Now run the search with the preprocessed dataset
# Note: We need to use load_from_disk in the code, but for now we'll modify the approach
# by creating a simple script that loads and passes the data
python main/run_feature_search.py \
    --model_path "$MODEL" \
    --sae_path "$SAE" \
    --sae_id "$SAE_ID" \
    --dataset_path "$PREPROCESSED_DATASET" \
    --tokens_str_path "$TOKENS" \
    --output_dir "$OUTPUT" \
    --score_type fisher \
    --num_features $NUM_FEATURES \
    --n_samples 1000 \
    --expand_range 1,2 \
    --column_name text

echo ""
echo "✅ Search complete: $OUTPUT"

