#!/bin/bash
# Example script for running feature interpretation
# This demonstrates different LLM provider options

# Configuration
STEERING_OUTPUTS="steering_outputs"
OUTPUT="interpretations.json"

echo "🚀 AutoInterp Steer - Feature Interpretation Examples"
echo "======================================================"
echo ""

# Option 1: OpenRouter (GPT-4o) - Recommended
echo "Option 1: Using OpenRouter with GPT-4o (Recommended)"
echo "-----------------------------------------------------"
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "⚠️  Warning: OPENROUTER_API_KEY not set. Set it or use --api_key flag."
    echo ""
fi

python scripts/run_interpretation.py \
    --steering_outputs "$STEERING_OUTPUTS" \
    --output "$OUTPUT" \
    --explainer_provider openrouter \
    --explainer_model openai/gpt-4o \
    --api_key "${OPENROUTER_API_KEY:-}" \
    || echo "⚠️  Skipping OpenRouter example (set OPENROUTER_API_KEY or provide --api_key)"

echo ""
echo "======================================================"
echo ""

# Option 2: vLLM (Local)
echo "Option 2: Using vLLM (Local deployment)"
echo "----------------------------------------"
echo "⚠️  Make sure vLLM server is running:"
echo "   python -m vllm.entrypoints.openai.api_server \\"
echo "     --model Qwen/Qwen2.5-7B-Instruct --port 8002"

python scripts/run_interpretation.py \
    --steering_outputs "$STEERING_OUTPUTS" \
    --output "${OUTPUT%.json}_vllm.json" \
    --explainer_provider vllm \
    --explainer_model Qwen/Qwen2.5-7B-Instruct \
    --explainer_api_base_url http://localhost:8002/v1 \
    || echo "⚠️  Skipping vLLM example (start vLLM server first)"

echo ""
echo "======================================================"
echo ""

# Option 3: Offline Transformers
echo "Option 3: Using Offline Transformers (Local)"
echo "----------------------------------------------"

python scripts/run_interpretation.py \
    --steering_outputs "$STEERING_OUTPUTS" \
    --output "${OUTPUT%.json}_offline.json" \
    --explainer_provider offline \
    --explainer_model meta-llama/Llama-2-7b-chat-hf \
    || echo "⚠️  Skipping offline example (model may not be available)"

echo ""
echo "✅ Done! Check interpretation results in:"
echo "   - $OUTPUT (or variant files)"
echo ""

