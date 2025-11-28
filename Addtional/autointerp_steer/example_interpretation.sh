#!/bin/bash
# Example script for running feature interpretation
# This demonstrates using vLLM HTTP API

# Configuration
STEERING_OUTPUTS="steering_outputs"
OUTPUT_DIR="interpretation_outputs"

echo "🚀 AutoInterp Steer - Feature Interpretation Example"
echo "======================================================"
echo ""

# Check vLLM server status
EXPLAINER_API_BASE="http://127.0.0.1:8002/v1"
EXPLAINER_MODEL="Qwen/Qwen2.5-72B-Instruct"

echo "🔍 Checking vLLM server status..."
if curl -s "$EXPLAINER_API_BASE/models" > /dev/null 2>&1; then
    echo "✅ vLLM server is running at $EXPLAINER_API_BASE"
else
    echo "❌ vLLM server is not running at $EXPLAINER_API_BASE"
    echo ""
    echo "Please start the vLLM server first:"
    echo ""
    echo "  python -m vllm.entrypoints.openai.api_server \\"
    echo "    --model $EXPLAINER_MODEL --port 8002"
    echo ""
    exit 1
fi

echo ""
echo "Using vLLM HTTP API for feature interpretation"
echo "----------------------------------------------"
echo "  API Base: $EXPLAINER_API_BASE"
echo "  Model: $EXPLAINER_MODEL"
echo ""

python scripts/run_interpretation.py \
    --steering_output_dir "$STEERING_OUTPUTS" \
    --output_dir "$OUTPUT_DIR" \
    --explainer_api_base "$EXPLAINER_API_BASE" \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_max_tokens 256 \
    --explainer_temperature 0.0 \
    --max_features 50

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Done! Check interpretation results in:"
    echo "   - $OUTPUT_DIR/interpretations.json"
    echo ""
else
    echo ""
    echo "❌ Interpretation failed"
    echo ""
fi
