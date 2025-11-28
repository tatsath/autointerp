#!/bin/bash

# FinBERT All Features Analysis with AutoInterp (ACTION-ORIENTED)
# This script runs AutoInterp evaluation on all features found in results.json
# Uses autointerp_full CLI (not Python API)
# Usage: ./run_finbert_autointerp.sh

echo "🚀 FinBERT All Features Analysis - ACTION-ORIENTED"
echo "=================================================="
echo "🔍 Running AutoInterp evaluation on all features from results.json:"
echo "  • Model: ProsusAI/finbert"
echo "  • Layer: 10"
echo "  • All features from results.json"
echo "  • Financial dataset (ashraq/financial-news)"
echo "  • ACTION-ORIENTED explanations with sentiment/market reaction focus"
echo ""

# Configuration
MODEL_NAME="ProsusAI/finbert"
LAYER=10
N_TOKENS=1000000  # 1M tokens
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Use the SAE from train/trained_models - need to convert it first
SAE_SOURCE_DIR="$SCRIPT_DIR/../../InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24"
SAE_AE_PT="$SAE_SOURCE_DIR/trainer_0/ae.pt"
SAE_CONFIG_JSON="$SAE_SOURCE_DIR/trainer_0/config.json"
SAE_CONVERTED_DIR="$SCRIPT_DIR/../../InterpUseCases_autointerp/EndtoEnd/Autointerp/finbert_sae_converted"
HOOKPOINT="encoder.layer.${LAYER}.output"
SAE_PATH="$SAE_CONVERTED_DIR"  # Use converted SAE path
RESULTS_JSON="$SCRIPT_DIR/../../InterpUseCases_autointerp/FinbertSentiment/FeatureAlign/results.json"
OUTPUT_DIR="$SCRIPT_DIR/autointerp_results"
AUTOINTERP_DIR="$SCRIPT_DIR"  # Use local autointerp_full
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_PROVIDER="vllm"
EXPLAINER_API_BASE_URL="http://localhost:8002/v1"

# Extract feature IDs from results.json
echo "📊 Extracting feature IDs from results.json..."
if [ ! -f "$RESULTS_JSON" ]; then
    echo "❌ Error: Results JSON not found: $RESULTS_JSON"
    exit 1
fi

# Extract unique feature IDs using Python
FEATURE_IDS=$(python3 << EOF
import json
with open("$RESULTS_JSON") as f:
    data = json.load(f)
feature_ids = set()
for item in data:
    for feature in item.get("top_features", []):
        feature_id = feature.get("feature_id")
        if feature_id is not None:
            feature_ids.add(int(feature_id))
print(" ".join(str(f) for f in sorted(feature_ids)))
EOF
)

if [ -z "$FEATURE_IDS" ]; then
    echo "❌ Error: No feature IDs found in results.json"
    exit 1
fi

NUM_FEATURES=$(echo $FEATURE_IDS | wc -w)
echo "📊 Found $NUM_FEATURES unique feature IDs"
echo "🎯 Features: $(echo $FEATURE_IDS | tr ' ' '\n' | head -10 | tr '\n' ' ')..."
echo ""

# Ensure FinBERT SAE is converted for autointerp_full
FINBERT_LAYER_DIR="$SAE_PATH/$HOOKPOINT"
if [ ! -d "$FINBERT_LAYER_DIR" ]; then
    echo "⚠️  FinBERT SAE not found in autointerp_full format. Converting..."
    python3 <<PY
import json
import os
from pathlib import Path
import torch
from safetensors.torch import save_file

sae_checkpoint = Path("$SAE_AE_PT")
config_path = Path("$SAE_CONFIG_JSON")
output_dir = Path("$SAE_CONVERTED_DIR")
layer = $LAYER
hook_name = "$HOOKPOINT"

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

# Activate conda environment
echo "🐍 Activating conda environment: sae"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sae
echo ""

# Set environment variables
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="2"
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
    echo "python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $EXPLAINER_MODEL \\"
    echo "    --port 8002 \\"
    echo "    --gpu-memory-utilization 0.7 \\"
    echo "    --max-model-len 4096 \\"
    echo "    --tensor-parallel-size 4 \\"
    echo "    --host 0.0.0.0"
    exit 1
fi
echo ""

# Create results directory
RESULTS_DIR="autointerp_results"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR/"
echo "🎯 Features to analyze: $NUM_FEATURES features from $RESULTS_JSON"
echo "📊 Using financial dataset: ashraq/financial-news"
echo "🎯 Using ACTION-ORIENTED prompts (sentiment/market reaction focus)"
echo ""

# Run AutoInterp evaluation using CLI
echo "🔍 Running AutoInterp evaluation for FinBERT Layer $LAYER..."
echo "   Using ACTION-ORIENTED prompts (sentiment/market reaction focus)"
echo "----------------------------------------"

cd "$AUTOINTERP_DIR"

RUN_NAME="finbert_layer${LAYER}_all_features_action"

# Run autointerp_full CLI
python -m autointerp_full \
    "$MODEL_NAME" \
    "$SAE_PATH" \
    --n_tokens $N_TOKENS \
    --cache_ctx_len 512 \
    --batch_size 1 \
    --feature_num $FEATURE_IDS \
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
    --dataset_repo "ashraq/financial-news" \
    --dataset_name default \
    --dataset_split "train" \
    --dataset_column headline \
    --filter_bos \
    --verbose \
    --overwrite scores \
    --name "$RUN_NAME" 2>&1 | tee "$RESULTS_DIR/run_${RUN_NAME}.log" || true

# Check if analysis was successful
ANALYSIS_EXIT_CODE=${PIPESTATUS[0]}
if [ $ANALYSIS_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ AutoInterp evaluation completed successfully"
    
    # Check for results
    RESULTS_SOURCE_DIR="$AUTOINTERP_DIR/results/$RUN_NAME"
    if [ -d "$RESULTS_SOURCE_DIR" ]; then
        # Move results to our output directory
        if [ -d "$RESULTS_SOURCE_DIR/explanations" ] && [ "$(ls -A $RESULTS_SOURCE_DIR/explanations 2>/dev/null)" ]; then
            echo "📋 Results found in: $RESULTS_SOURCE_DIR"
            
            # Generate CSV summary if generate_results_csv.py exists
            if [ -f "$AUTOINTERP_DIR/generate_results_csv.py" ]; then
                echo "📊 Generating CSV summary..."
                python "$AUTOINTERP_DIR/generate_results_csv.py" "$RESULTS_SOURCE_DIR"
            fi
            
            CSV_FILE="$RESULTS_SOURCE_DIR/results_summary.csv"
            if [ -f "$CSV_FILE" ]; then
                echo "📈 CSV summary: $CSV_FILE"
                echo "   Showing first 10 rows:"
                head -11 "$CSV_FILE" | column -t -s, || head -11 "$CSV_FILE"
            fi
        fi
    fi
    
    echo ""
    echo "📋 Analysis Complete!"
    echo "🔍 Check the following for detailed results:"
    echo "   • $RESULTS_SOURCE_DIR/ - All results and logs"
    echo "   • CSV summary: $RESULTS_SOURCE_DIR/results_summary.csv"
    echo "   • Artifacts: $RESULTS_SOURCE_DIR/"
    echo ""
    echo "✅ FinBERT All Features analysis completed successfully!"
    echo "   Using ACTION-ORIENTED prompts (sentiment/market reaction focus)"
else
    echo ""
    echo "❌ Analysis failed with exit code $ANALYSIS_EXIT_CODE"
    echo "🔍 Check the logs above for error details"
    exit 1
fi
