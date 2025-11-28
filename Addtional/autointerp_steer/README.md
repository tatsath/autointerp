# AutoInterp Steer - SAE Feature Steering & Interpretation Pipeline

This pipeline implements automatic feature interpretation for Sparse Autoencoder (SAE) features through steering experiments and LLM-based labeling.

## 📋 Table of Contents

1. [Installation](#installation)
2. [Pipeline Overview](#pipeline-overview)
3. [How It Works](#how-it-works)
4. [Running the Pipeline](#running-the-pipeline)
5. [Parameters & Configuration](#parameters--configuration)
6. [File Structure](#file-structure)
7. [Output Format](#output-format)
8. [GPU Configuration](#gpu-configuration)

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPUs (minimum 2 GPUs recommended)
- Conda (recommended)

### Setup

```bash
# Clone repository
cd autointerp/autointerp_steer

# Create conda environment
conda create -n sae python=3.10 pytorch cudatoolkit -c pytorch -y
conda activate sae

# Install dependencies
pip install -r requirements.txt

# Install SAELens and TransformerLens
pip install sae-lens transformer-lens

# Install vLLM for feature labeling (optional but recommended)
pip install vllm
```

---

## Pipeline Overview

The pipeline performs two main steps:

1. **Steering Experiments**: Amplifies SAE features and generates text to observe effects
2. **Feature Labeling**: Uses LLM analysis to interpret and label features based on steering outputs

```
Input: SAE Model + Base Model + Dataset
    ↓
[Step 1: Steering] → Generate texts with steered features
    ↓
[Step 2: Labeling] → Analyze steering outputs with LLM
    ↓
Output: CSV with feature labels + steering effect scores
```

---

## How It Works

### Steering (Step 1)

**Purpose**: Observe how amplifying specific SAE features affects text generation.

**Process**:
1. Loads SAE features for specified layer(s)
2. For each feature:
   - Finds maximum activation value (`A_max`) for the feature
   - Selects prompts from dataset (stratified sampling)
   - Generates text with feature steered at different strengths: `[-2.0, -1.0, 1.0, 2.0]`
   - Saves all generated texts as JSON files

**Key Function**: `generate_with_steering()` in `sae_pipeline/steering.py`
- Amplifies feature activations by `strength × A_max`
- Generates continuation text
- Compares steered vs original outputs

### Labeling (Step 2)

**Purpose**: Automatically generate human-readable labels for features.

**Process**:
1. Loads steering output JSON files
2. For each feature, builds prompt with:
   - 2 example prompts showing original + steered texts at 4 strengths
   - Truncated text snippets (200 chars) for efficiency
3. Sends to LLM (vLLM/OpenRouter/offline) for analysis
4. Extracts short label (< 20 words) from LLM response
5. Calculates steering effect score (0.0-1.0) based on text diversity
6. Generates CSV summary

**Key Function**: `interpret_all_features()` in `sae_pipeline/feature_interpreter.py`

**Steering Effect Score**:
- Measures how much outputs vary when feature is steered
- Based on token diversity (60%) + text uniqueness (40%)
- Higher score = stronger steering effect = more interpretable feature

---

## Running the Pipeline

### Quick Start

```bash
# 1. Start vLLM server (in separate terminal)
bash scripts/start_vllm_server.sh

# 2. Run full pipeline
bash scripts/run_large_sae_steer_interpret.sh
```

### Manual Execution

```bash
# Activate environment
conda activate sae

# Step 1: Run steering
python scripts/run_steering_large_sae.py \
    --output_folder large_sae_steering_outputs \
    --sae_path /path/to/sae \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --layer 19 \
    --features_list 6 9 \
    --dataset_repo wikitext \
    --dataset_name wikitext-103-raw-v1 \
    --num_prompts 10

# Step 2: Run labeling
python scripts/run_interpretation.py \
    --steering_outputs large_sae_steering_outputs \
    --output large_sae_interpretations.json \
    --explainer_provider vllm \
    --explainer_model Qwen/Qwen2.5-72B-Instruct \
    --explainer_api_base_url http://localhost:8002/v1 \
    --layers 19

# Step 3: Generate CSV
python scripts/generate_results_csv.py \
    large_sae_interpretations.json \
    large_sae_steering_outputs \
    large_sae_steering_outputs/large_sae_interpretations_summary.csv
```

---

## Parameters & Configuration

### Main Script Parameters (`run_large_sae_steer_interpret.sh`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Base transformer model to steer |
| `LARGE_SAE_MODEL` | `/path/to/converted_safetensors` | Path to local SAE model directory |
| `LAYER` | `19` | Layer number to analyze |
| `TOP_N_FEATURES` | `2` | Number of features to analyze (reduced for speed) |
| `NUM_PROMPTS` | `10` | Number of prompts per feature (methodology uses 30) |
| `DATASET_REPO` | `wikitext` | HuggingFace dataset repository |
| `DATASET_NAME` | `wikitext-103-raw-v1` | Specific dataset config/name |
| `EXPLAINER_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | LLM for feature labeling |
| `EXPLAINER_PROVIDER` | `vllm` | Labeling provider: `vllm`, `openrouter`, `offline` |

### Steering Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steering_strengths` | `[-2.0, -1.0, 1.0, 2.0]` | Multipliers for feature amplification (4 levels, reduced from 14) |
| `max_tokens` | `50` | Maximum tokens to generate per prompt (reduced from 95) |
| `device` | `cuda:0` | Device for steering experiments |

### Labeling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_examples` | `2` | Number of prompt examples shown to LLM (reduced from 5) |
| `max_tokens` | `500` | Max tokens for LLM response (reduced for context limits) |
| `temperature` | `0.7` | Sampling temperature for LLM |

### Speed Optimizations (Current Configuration)

- **Features**: 2 (methodology uses 10)
- **Prompts**: 10 (methodology uses 30)
- **Steering levels**: 4 (methodology uses 14)
- **LLM examples**: 2 (methodology uses 5)
- **Max tokens**: 50 generated, 200 chars truncated in prompts

---

## File Structure

```
autointerp_steer/
├── scripts/
│   ├── run_large_sae_steer_interpret.sh    # Main pipeline script
│   ├── start_vllm_server.sh                 # Start vLLM server for labeling
│   ├── run_steering_large_sae.py            # Steering execution
│   ├── run_interpretation.py                # Feature labeling execution
│   └── generate_results_csv.py               # CSV generation from JSON
├── sae_pipeline/
│   ├── steering.py                          # Core steering functions
│   ├── feature_interpreter.py               # LLM labeling logic
│   └── steering_utils.py                    # SAE loading utilities
├── large_sae_steering_outputs/              # Steering outputs (JSON files)
│   ├── generated_texts_layer_19_feature_6_0_prompt_0.json
│   ├── generated_texts_layer_19_feature_9_1_prompt_0.json
│   └── large_sae_interpretations_summary.csv  # Final CSV output
├── large_sae_interpretations.json           # Full interpretation JSON
└── README.md                                 # This file
```

### Steering Output Files

Format: `generated_texts_layer_{LAYER}_feature_{FEATURE_ID}_{SUFFIX}_prompt_{PROMPT_ID}.json`

Structure:
```json
{
  "19": {
    "6": {
      "prompt_text": {
        "original": "text without steering...",
        "-2.0": "text with feature at -2.0 strength...",
        "-1.0": "text with feature at -1.0 strength...",
        "1.0": "text with feature at 1.0 strength...",
        "2.0": "text with feature at 2.0 strength..."
      }
    }
  }
}
```

---

## Output Format

### CSV Summary (`large_sae_interpretations_summary.csv`)

Columns:
- `layer`: Layer number (integer)
- `feature`: Feature ID (integer)
- `label`: Short descriptive phrase (< 20 words)
- `steering_effect`: Score 0.0-1.0 measuring steering impact

**Example**:
```csv
layer,feature,label,steering_effect
19,6,Historical and temporal context in narratives,0.282
19,9,Mathematical problem statements and solutions,0.294
```

### JSON Interpretations (`large_sae_interpretations.json`)

Structure:
```json
{
  "19": {
    "6": {
      "feature_number": 6,
      "layer": 19,
      "interpretation": "SUMMARY: Historical and temporal context...\n\nDETAILED ANALYSIS: ...",
      "status": "success"
    }
  }
}
```

---

## GPU Configuration

### Recommended Setup

- **Steering Pipeline**: GPUs 0, 1, 2, 3
- **vLLM Server**: GPUs 4, 5, 6, 7 (for tensor parallelism)

### Environment Variables

```bash
# Set in run_large_sae_steer_interpret.sh
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # For steering
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce fragmentation
```

### vLLM Server Configuration

- **Preferred**: 4 GPUs (4,5,6,7) for tensor parallelism
- **Fallback**: 2 GPUs (6,7) if preferred unavailable
- **Port**: 8002
- **Memory**: 70% GPU memory utilization

---

## Troubleshooting

### vLLM Server Not Running
```bash
# Check if running
curl http://localhost:8002/v1/models

# Start manually
bash scripts/start_vllm_server.sh
```

### Out of Memory Errors
- Reduce `NUM_PROMPTS` or `TOP_N_FEATURES`
- Reduce `max_tokens` in steering
- Use fewer GPUs or reduce batch sizes

### Import Errors
```bash
# Ensure you're in the correct directory
cd /path/to/autointerp/autointerp_steer

# Check Python path
python -c "import sys; print(sys.path)"
```

---

## Citation

Based on methodology from:
- **Paper**: *Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders*
- **Venue**: Findings of ACL 2025
- **Authors**: Kuznetsov et al.
