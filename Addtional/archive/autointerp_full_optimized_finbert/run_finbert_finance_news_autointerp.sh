#!/bin/bash

echo "🚀 AutoInterp Financial News Analysis - FinBERT Top 100 Features (Granular Prompts)"
echo "====================================================================================="
echo "🔍 Analyzing top 100 finance-focused features from FinBERT Layer 10 SAE"
echo "⚡ Using vLLM server for faster LLM explanations"
echo "📊 Processing top 100 features (Layer 10, K=24)"
echo "🎯 Using granular prompts with ENTITY/SECTOR/MACRO/EVENT/STRUCTURAL/LEXICAL classification"
echo ""

# Get script directory (autointerp_full_finance_optimized)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Path to EndtoEnd/Autointerp directory (where SAE and features are stored)
ENDPOINT_DIR="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp"

# Configuration
BASE_MODEL="ProsusAI/finbert"
FINBERT_SAE_MODEL_DIR="$ENDPOINT_DIR/finbert_sae_converted"
FINBERT_SAE_CHECKPOINT="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24/trainer_0/ae.pt"
FINBERT_SAE_CONFIG="/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24/trainer_0/config.json"
FEATURES_SUMMARY_PATH="$ENDPOINT_DIR/finbert_finance_features/top_finance_features_summary.txt"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"
N_TOKENS=1000000  # 1M tokens for analysis
LAYER=10
DICT_SIZE=3072
NUM_FEATURES_TO_RUN=100  # Top 100 features
HOOKPOINT="encoder.layer.${LAYER}.output"
RUN_NAME="finbert_finance_news_run"

# Results directory (inside results folder)
RESULTS_DIR="$SCRIPT_DIR/results/$RUN_NAME"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🤖 Using vLLM provider: $EXPLAINER_PROVIDER"
echo "🌐 vLLM server URL: $EXPLAINER_API_BASE_URL"
echo "🎯 Processing top $NUM_FEATURES_TO_RUN features (Layer $LAYER, K=24) out of $DICT_SIZE total"
echo "📋 Using granular prompts: ENTITY | SECTOR | MACRO | EVENT | STRUCTURAL | LEXICAL"
echo "🔍 Using ContrastiveExplainer with FAISS (finance-embeddings-investopedia) for hard negatives"
echo "📊 Using 15 examples per feature + 60 non-overlapping examples for scoring"
echo "🎯 Using Detection scorer only (faster execution)"
echo "📈 Using 1M tokens for analysis (cache will be regenerated)"
echo "⚙️  Processing 100 features - this may take several hours"
echo "📰 Using ashraq/financial-news dataset (train[:80%]) for financial text patterns"
echo ""

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Ensure FinBERT SAE is converted for autointerp_full
FINBERT_LAYER_DIR="$FINBERT_SAE_MODEL_DIR/$HOOKPOINT"
if [ ! -d "$FINBERT_LAYER_DIR" ]; then
    echo "⚠️  FinBERT SAE not found in autointerp_full format. Converting..."
    python3 - <<'PY'
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

sae_checkpoint = Path("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24/trainer_0/ae.pt")
config_path = Path("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24/trainer_0/config.json")
output_dir = Path("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/finbert_sae_converted")
layer = 10
hook_name = f"encoder.layer.{layer}.output"

if not sae_checkpoint.exists():
    raise FileNotFoundError(f"SAE checkpoint not found: {sae_checkpoint}")
if not config_path.exists():
    raise FileNotFoundError(f"SAE config not found: {config_path}")

with open(config_path, "r") as f:
    config = json.load(f)

trainer_config = config["trainer"]
dict_size = trainer_config["dict_size"]
k = trainer_config["k"]
activation_dim = trainer_config["activation_dim"]

checkpoint = torch.load(sae_checkpoint, map_location="cpu")
encoder_weight = checkpoint["encoder.weight"]
encoder_bias = checkpoint["encoder.bias"]
decoder_weight = checkpoint["decoder.weight"]
b_dec = checkpoint.get("b_dec", torch.zeros(activation_dim))

layer_dir = output_dir / hook_name
layer_dir.mkdir(parents=True, exist_ok=True)

cfg = {
    "d_in": activation_dim,
    "activation": "topk",
    "k": k,
    "num_latents": dict_size,
    "expansion_factor": dict_size / activation_dim,
    "normalize_decoder": True,
    "multi_topk": False,
    "skip_connection": False,
    "transcode": False,
    "hook_layer": layer,
    "hook_name": hook_name,
}

with open(layer_dir / "cfg.json", "w") as f:
    json.dump(cfg, f, indent=2)

state_dict = {
    "encoder.weight": encoder_weight.contiguous(),
    "encoder.bias": encoder_bias.contiguous(),
    "W_dec": decoder_weight.T.contiguous(),
    "b_dec": b_dec.contiguous(),
}

save_file(state_dict, layer_dir / "sae.safetensors")
print(f"✅ FinBERT SAE converted and saved to: {layer_dir}")
PY

    if [ $? -ne 0 ]; then
        echo "❌ Failed to convert FinBERT SAE"
        exit 1
    fi
    echo ""
fi

# Set environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="0"
export VLLM_USE_DEEP_GEMM=1
export VLLM_GPU_MEMORY_UTILIZATION=0.7
export VLLM_MAX_MODEL_LEN=4096
export VLLM_BLOCK_SIZE=16
export VLLM_SWAP_SPACE=0

# Check if vLLM server is running
echo "🔍 Checking vLLM server status..."
if curl -s "$EXPLAINER_API_BASE_URL/models" > /dev/null 2>&1; then
    echo "✅ vLLM server is running at $EXPLAINER_API_BASE_URL"
else
    echo "❌ vLLM server is not running at $EXPLAINER_API_BASE_URL"
    echo "Please start vLLM server first:"
    echo "bash start_vllm_server_72b.sh"
    exit 1
fi
echo ""

# Extract top features from summary file
echo "🔍 Extracting top $NUM_FEATURES_TO_RUN features from summary..."
cd "$ENDPOINT_DIR"
python3 -c "
import re

with open('$FEATURES_SUMMARY_PATH', 'r') as f:
    lines = f.readlines()

features = []
pattern = re.compile(r'^\\s*\\d+\\.\\s+Feature\\s+(\\d+):')
for line in lines:
    match = pattern.match(line)
    if match:
        features.append(int(match.group(1)))
        if len(features) >= $NUM_FEATURES_TO_RUN:
            break

with open('finbert_finance_news_features_list.txt', 'w') as f:
    f.write(' '.join(map(str, features)))

print(f'✅ Extracted {len(features)} features: {features[:10]}...')
"

# Read the feature list
FEATURE_LIST=$(cat "$ENDPOINT_DIR/finbert_finance_news_features_list.txt")

if [ -z "$FEATURE_LIST" ]; then
    echo "❌ Failed to extract features from summary file"
    exit 1
fi

echo "🔍 Running AutoInterp Financial News Analysis for FinBERT Features..."
echo "====================================================================="

# Navigate to the autointerp_full_finance_optimized directory (with granular prompts)
cd "$SCRIPT_DIR"

# Delete old explanations and cache to force regeneration
if [ -d "results/$RUN_NAME/explanations" ]; then
    echo "🗑️  Removing old explanations to force regeneration with updated prompts..."
    rm -rf "results/$RUN_NAME/explanations"
    echo "✅ Old explanations removed"
fi

# Run AutoInterp for FinBERT features with financial-news dataset
python -m autointerp_full \
    "$BASE_MODEL" \
    "$FINBERT_SAE_MODEL_DIR" \
    --n_tokens "$N_TOKENS" \
    --cache_ctx_len 512 \
    --batch_size 1 \
    --feature_num $FEATURE_LIST \
    --hookpoints "$HOOKPOINT" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider "$EXPLAINER_PROVIDER" \
    --explainer_api_base_url "$EXPLAINER_API_BASE_URL" \
    --explainer_model_max_len 8192 \
    --num_gpus 1 \
    --num_examples_per_scorer_prompt 15 \
    --n_non_activating 60 \
    --min_examples 8 \
    --pipeline_num_proc 4 \
    --non_activating_source "FAISS" \
    --faiss_embedding_model "FinLang/finance-embeddings-investopedia" \
    --faiss_embedding_cache_dir ".embedding_cache" \
    --faiss_embedding_cache_enabled \
    --dataset_repo ashraq/financial-news \
    --dataset_name default \
    --dataset_split "train" \
    --dataset_column headline \
    --filter_bos \
    --verbose \
    --overwrite scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run.log" || true

# Check if results were generated (even if logging step failed)
RESULTS_SOURCE_DIR="$SCRIPT_DIR/results/$RUN_NAME"

if [ -d "$RESULTS_SOURCE_DIR" ] && [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A "$RESULTS_SOURCE_DIR/explanations" 2>/dev/null)" ]; then
    echo "✅ AutoInterp scoring completed (results found)"
    
    # Count explanations
    EXPLANATION_COUNT=$(find "$RESULTS_SOURCE_DIR/explanations" -name "*.txt" 2>/dev/null | wc -l)
    echo "📊 Generated $EXPLANATION_COUNT feature explanations"
    
    # Generate CSV summary directly in results folder
    if [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A "$RESULTS_SOURCE_DIR/explanations" 2>/dev/null)" ]; then
        echo "📊 Generating CSV summary for FinBERT financial news results..."
        cd "$SCRIPT_DIR"
        
        # Generate enhanced CSV with scorer metrics
        echo "📊 Generating enhanced CSV with scorer metrics..."
        python generate_nemotron_enhanced_csv.py "$RESULTS_SOURCE_DIR" --output_name finbert_finance_results_summary_enhanced.csv 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log"
        
        if [ -f "$RESULTS_SOURCE_DIR/finbert_finance_results_summary_enhanced.csv" ]; then
            echo "✅ Enhanced CSV saved: $RESULTS_SOURCE_DIR/finbert_finance_results_summary_enhanced.csv"
            
            # Also generate basic CSV for compatibility
            python generate_results_csv.py "$RESULTS_SOURCE_DIR" 2>&1 | tee -a "$RESULTS_SOURCE_DIR/csv_generation.log" || true
            if [ -f "$RESULTS_SOURCE_DIR/results_summary.csv" ]; then
                echo "📈 Basic CSV saved: $RESULTS_SOURCE_DIR/results_summary.csv"
            fi
        else
            echo "⚠️  Enhanced CSV generation may have failed, check logs"
        fi
    fi
else
    echo "❌ AutoInterp run failed - no results found"
    exit 1
fi

# Clean up temporary file
rm -f "$ENDPOINT_DIR/finbert_finance_news_features_list.txt"

echo ""
echo "🔬 FinBERT financial news feature analysis completed! Check the results in: $RESULTS_SOURCE_DIR/"
echo "📊 Analyzed top $NUM_FEATURES_TO_RUN features (out of $DICT_SIZE total) from the FinBERT SAE model (Layer $LAYER, K=24)"
echo "📁 Results saved in: $RESULTS_SOURCE_DIR"
echo "📄 CSV files generated in results folder:"
echo "   - finbert_finance_results_summary_enhanced.csv"
echo "   - results_summary.csv (if generated)"
echo ""
echo "📝 Note: Using detection scorer only for faster execution"
echo "   - Enhanced CSV includes detection F1 scores"
echo "   - Results format matches nemotron_finance_results_summary_enhanced.csv"
echo ""


