# AutoInterp - SAE Interpretability System

Comprehensive toolkit for understanding what features in your Sparse Autoencoder (SAE) model have learned. Multiple complementary approaches for feature discovery, explanation, and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation & Dependencies](#installation--dependencies)
4. [Main Approaches](#main-approaches)
   - [Search AutoInterp (Recommended)](#search-autointerp-recommended)
   - [AutoInterp SAEEval](#autointerp-saeeval)
   - [AutoInterp Full](#autointerp-full)
5. [Side-by-Side Comparison](#side-by-side-comparison)
6. [Folder Descriptions](#folder-descriptions)
7. [Running Examples](#running-examples)
8. [Sample Results](#sample-results)
9. [Additional Components](#additional-components)
10. [Repository Structure](#repository-structure)

## Overview

AutoInterp provides multiple approaches for SAE feature interpretability, with **Search AutoInterp** being the recommended primary method for most use cases. The system enables you to:

- **Search** for domain-relevant features using activation-separation-based scoring
- **Label** features simultaneously during the search process
- **Evaluate** feature explanations using model-agnostic tools
- **Understand** features through detailed analysis (when needed)

## Quick Start

```bash
# Install dependencies
pip install -e .

# Recommended: Search and label features simultaneously
cd search_autointerp
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path "1. search/test_tokens.json" \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100

# Results saved to:
# - results/1_search/ - Feature search results
# - results/2_labeling_lite/ - Basic labeling (recommended)
# - results/3_labeling_advance/ - Advanced labeling (optional)
```

## Installation & Dependencies

### Basic Installation

```bash
# From main directory
pip install -e .

# For visualization support (optional)
pip install -e ".[visualize]"
```

### Core Dependencies

- **Python**: 3.10+
- **PyTorch**: Latest stable version
- **Transformers**: HuggingFace transformers library
- **SAE Lens**: For SAE model loading and processing
- **vLLM**: For high-throughput inference (optional, for labeling)
- **FAISS**: For contrastive learning (optional, for AutoInterp Full)
- **NumPy, Pandas**: Data processing
- **Safetensors**: Model loading

### Additional Requirements by Component

**Search AutoInterp**:
- `torch`, `transformers`, `sae_lens`, `datasets` (HuggingFace)

**AutoInterp SAEEval**:
- `torch`, `safetensors`, `transformer_lens`, `sae_lens`, `requests`
- vLLM server (for API-based labeling)

**AutoInterp Full**:
- All above dependencies
- Additional: `plotly` (for visualization), `openai` (for OpenRouter provider)

## Main Approaches

### Search AutoInterp (Recommended) ⭐

**Location**: `search_autointerp/`

**Purpose**: Training-free approach to identify and label sparse features aligned to domain concepts (finance, reasoning, healthcare, etc.) using activation-separation-based scoring. **This is the preferred method** for most use cases.

**Key Features**:
- ✅ **Search and label simultaneously** - Find relevant features and generate labels in one pipeline
- ✅ Training-free (no probes, no classifiers)
- ✅ Works with 20-100 examples per concept
- ✅ Multiple scoring methods (simple, Fisher-style, domain-specific)
- ✅ Domain-agnostic - works with any HuggingFace dataset
- ✅ Three-component pipeline: Search → Lite Labeling → Advanced Labeling (optional)

**How It Works**:

1. **Step 1: Feature Search** (`1. search/`)
   - Defines domain tokens (e.g., financial keywords)
   - Identifies C⁺ (text with tokens) vs C⁻ (text without)
   - Computes activation separation scores (Fisher recommended)
   - Selects top features encoding the concept

2. **Step 2: Basic Labeling** (`2. autointerp_lite/`) ⭐ **Important**
   - Collects positive/negative context windows
   - Generates labels using vLLM API
   - Fast and reliable for most use cases

3. **Step 3: Advanced Labeling** (`3. autointerp_advance/`) ⚠️ **Optional**
   - Extracts examples using SaeVisRunner
   - Generates labels from extracted examples
   - **Note**: May not work for some models - use only if Step 2 doesn't provide sufficient detail

**Usage - Complete Pipeline**:

```bash
cd search_autointerp
conda activate sae

# Run all three steps seamlessly
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path "1. search/test_tokens.json" \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100
```

**Usage - Individual Steps**:

```bash
# Step 1: Search only
cd "1. search"
python run_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path test_tokens.json \
    --output_dir ../results/1_search \
    --score_type fisher \
    --num_features 20

# Step 2: Basic labeling (recommended)
cd "../2. autointerp_lite"
python run_labeling.py \
    --search_output ../results/1_search \
    --output_dir ../results/2_labeling_lite

# Step 3: Advanced labeling (optional - may not work for all models)
cd "../3. autointerp_advance"
python run_labeling_advanced.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --search_output ../results/1_search \
    --output_dir ../results/3_labeling_advance
```

**Output**:
- `results/1_search/feature_list.json` - Feature scores and rankings
- `results/2_labeling_lite/feature_labels.json` - Generated labels with scores
- `results/2_labeling_lite/activating_sentences.json` - Top activating sentences
- `results/3_labeling_advance/feature_labels.json` - Advanced labels (if Step 3 succeeds)

**Example Output Files**:
- See `search_autointerp/nemotron_finance_detailed_labels.csv` for sample results with top 50 and bottom 50 activating sentences
- CSV format: `feature_id,label,label_score,positive_sentences,negative_sentences`

### AutoInterp SAEEval

**Location**: `autointerp_saeeval/`

**Purpose**: Model-agnostic tool for automatically producing human-readable explanations for SAE features. Based on the eval folder methodology, this tool evaluates how well an LLM explains and predicts feature activations.

**Key Features**:
- ✅ Domain-agnostic and model-agnostic evaluation
- ✅ Works with any LLM (Nemotron, GPT-OSS, Llama, Gemma, FinBERT, etc.)
- ✅ Three-stage process: activation evidence collection, LLM explanation, evaluation/scoring
- ✅ Provides interpretability scores (AutoInterp score: 0-1)
- ✅ Standalone tool with minimal dependencies

**How It Works**:

1. **Collecting Activation Evidence**: Samples text sequences, computes SAE activations, selects strong (top-k), medium (importance-weighted), and low (random) activation examples

2. **Natural-Language Explanation**: Passes collected examples to LLM using structured, domain-neutral prompt to generate explanations

3. **Evaluation and Scoring**: Tests how well the explanation predicts activations by asking LLM to identify which sequences will activate the feature

**Usage**:

```bash
cd autointerp_saeeval

# Start vLLM server first
bash start_vllm_server_72b.sh

# Run evaluation
python run_nemotron_top100_finance_eval.py  # Or other model-specific scripts
```

**Key Scripts**:
- `run_finbert.py` - FinBERT evaluation
- `run_nemotron_top100_finance_eval.py` - Nemotron top 100 finance features
- `run_nemotron_finance_features_eval.py` - Nemotron finance features
- `run_llama.py` - Llama evaluation

**Output**:
- CSV summaries: `Results/<model>_layer<num>_features_summary_<timestamp>.csv`
- JSON results: `Results/<sae_id>_eval_results.json` (contains full logs with top activating examples)
- Log files: `Results/autointerp_<model>_layer<num>_features<list>_<timestamp>.txt`

**Limitations**:
- Less detailed than AutoInterp Full
- No contrastive learning
- Script-based configuration (less flexible than YAML)

### AutoInterp Full

**Location**: `autointerp_full/`

**Purpose**: Production LLM-based feature explanation system with comprehensive metrics. **Note**: This approach is **not recommended** for most use cases due to complexity, time requirements, and generic output quality.

**Key Features**:
- LLM-based feature explanation with F1, precision, recall metrics
- External YAML prompt configuration (`prompts.yaml`, `prompts_finance.yaml`)
- FAISS-based contrastive learning for better explanations
- Cache management for efficient re-runs
- Multiple explainer providers (vLLM, OpenRouter, offline)

**Limitations** ⚠️:
- ❌ **Very slow**: 30-60 minutes per feature
- ❌ **Generic output**: Explanations often lack specificity and detail
- ❌ **Complex setup**: Requires extensive configuration and tuning
- ❌ **Not robust**: Overall activation analysis is not very robust
- ❌ **Difficult to use**: Requires significant expertise to get meaningful results

**When to Use**:
- Only if you need F1, precision, recall metrics for research validation
- Only if you have extensive time and computational resources
- Only if other methods (Search AutoInterp, SAEEval) don't meet your needs

**Usage** (not recommended for most users):

```bash
cd autointerp_full

# FinBERT analysis
./run_finbert.sh

# Nemotron analysis
./run_nemotron.sh

# Llama all features
./run_llama_all.sh
```

**Output**: Detailed explanations with confidence scores, F1 metrics, CSV summaries (but often generic)

## Side-by-Side Comparison

| Feature | Search AutoInterp ⭐ | AutoInterp SAEEval | AutoInterp Full |
|---------|---------------------|-------------------|-----------------|
| **Primary Use** | Search + label simultaneously | Standalone evaluation | Production LLM explanations |
| **Speed** | Fast (2-5 min for search + labeling) | Variable (depends on setup) | Very slow (30-60 min/feature) |
| **Output Quality** | Specific, domain-focused labels | Natural-language labels + scores | Generic explanations |
| **Output Detail** | Labels + top/bottom sentences | Labels + AutoInterp scores | Explanations + F1/precision/recall |
| **Labeling Method** | vLLM-based (Steps 1-2) | LLM explanation + scoring | LLM with contrastive learning |
| **Configuration** | Command-line args | Script-based | YAML prompts, complex config |
| **Robustness** | ✅ High - activation-separation based | ✅ High - model-agnostic | ⚠️ Low - not very robust |
| **Ease of Use** | ✅ Easy - unified pipeline | ✅ Moderate - script-based | ❌ Difficult - complex setup |
| **Model Support** | Any base model | Any LLM (model-agnostic) | Any with chat format |
| **Domain Support** | Any domain (finance, reasoning, etc.) | Any domain | Any domain |
| **Recommended For** | ⭐ **Most use cases** | Model comparison, evaluation | Research validation only |
| **Dependencies** | Minimal | Moderate | Extensive |

### What They Do and How They Label

**Search AutoInterp**:
- **What it does**: Finds domain-relevant features using activation-separation scoring, then labels them
- **How it labels**: Collects top activating sentences, sends to vLLM with context, generates concise labels
- **Output format**: CSV with `feature_id,label,label_score,positive_sentences,negative_sentences`

**AutoInterp SAEEval**:
- **What it does**: Evaluates feature explanations by testing LLM's ability to predict activations
- **How it labels**: LLM generates explanation from top examples, then scores explanation by testing predictions
- **Output format**: CSV with `feature,label,autointerp_score` and JSON with full details

**AutoInterp Full**:
- **What it does**: Comprehensive explanation system with multiple scorers and contrastive learning
- **How it labels**: Uses FAISS for negative examples, multiple prompt types, comprehensive scoring
- **Output format**: Multiple files (explanations, scores, cached activations)

## Folder Descriptions

### `search_autointerp/` ⭐ **Recommended**

**Purpose**: Complete pipeline for searching and labeling features simultaneously.

**Structure**:
- `1. search/` - Feature search functionality (activation-separation scoring)
- `2. autointerp_lite/` - Basic labeling pipeline (recommended, works reliably)
- `3. autointerp_advance/` - Advanced labeling pipeline (optional, may not work for all models)
- `pipeline.py` - Seamless chaining script for all three steps
- `utils.py` - Utility functions

**Key Files**:
- `pipeline.py` - Main entry point for complete pipeline
- `1. search/run_search.py` - Feature search implementation
- `2. autointerp_lite/run_labeling.py` - Basic labeling
- `3. autointerp_advance/run_labeling_advanced.py` - Advanced labeling

### `autointerp_saeeval/` - Standalone SAE Evaluation Tool

**Purpose**: Model-agnostic tool for automatically producing human-readable explanations for SAE features.

**Key Features**:
- Domain-agnostic and model-agnostic evaluation
- Works with any LLM (Nemotron, GPT-OSS, Llama, Gemma, FinBERT, etc.)
- Three-stage process: activation evidence collection, LLM explanation, evaluation/scoring
- Standalone scripts: `run_finbert.py`, `run_nemotron_top100_finance_eval.py`, etc.

**Output**: Interpretability scores, natural-language labels, CSV summaries

### `autointerp_full/` - Production LLM-Based Feature Explanation

**Purpose**: Generate human-readable explanations with confidence scores using LLM analysis.

**Key Features**:
- LLM-based feature explanation with F1, precision, recall metrics
- External YAML prompt configuration (`prompts.yaml`, `prompts_finance.yaml`)
- FAISS-based contrastive learning for better explanations
- Cache management for efficient re-runs
- Multiple explainer providers (vLLM, OpenRouter, offline)

**Output**: Detailed explanations with confidence scores, F1 metrics, CSV summaries

**Note**: Not recommended for most use cases due to complexity and generic output.

### `autointerp_lite/` - Fast Feature Discovery

**Purpose**: Quickly find domain-relevant features by comparing activations on domain-specific vs general text.

**Key Features**:
- Fast execution (2-5 minutes for 1000+ features)
- Domain specialization scoring
- Optional LLM labeling
- Compares domain vs general text activations

**Output**: Ranked list of domain-relevant features with specialization scores

### `autointerp_steer/` - Feature Steering Analysis

**Purpose**: Understand feature functions through controlled activation intervention experiments.

**Key Features**:
- Activation steering (`x' = x + λ·A_max·d_i`)
- Text generation with varying steering strengths
- Based on Kuznetsov et al. (2025) methodology

**Output**: Generated texts with different steering strengths (JSON format)

### `archive/` - Historical Versions

Contains archived versions: `autointerp_full_finance/`, `autointerp_full_optimized_finbert/`, `autointerp_full_reasoning/`, etc. Reference only.

## Running Examples

### Search AutoInterp (Recommended) ⭐

```bash
cd search_autointerp
conda activate sae

# Complete pipeline (recommended)
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path "1. search/test_tokens.json" \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100

# Individual steps
cd "1. search"
python run_search.py --model_path ... --sae_path ... --output_dir ../results/1_search

cd "../2. autointerp_lite"
python run_labeling.py --search_output ../results/1_search --output_dir ../results/2_labeling_lite
```

### AutoInterp SAEEval

```bash
cd autointerp_saeeval

# Start vLLM server
bash start_vllm_server_72b.sh

# Run evaluation
python run_nemotron_top100_finance_eval.py
# Or
python run_finbert.py
# Or
python run_llama.py
```

### AutoInterp Full (Not Recommended)

```bash
cd autointerp_full

# FinBERT analysis
./run_finbert.sh

# Nemotron analysis
./run_nemotron.sh

# Llama all features
./run_llama_all.sh
```

### AutoInterp Lite

```bash
cd autointerp_lite
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct"
```

### AutoInterp Steer

```bash
conda activate sae
cd autointerp_steer
python scripts/run_steering.py --output_folder steering_outputs
```

## Sample Results

### Search AutoInterp Output

See `search_autointerp/nemotron_finance_detailed_labels.csv` for complete example with:
- Feature IDs and labels
- Label scores (0-1)
- Top 50 positive activating sentences
- Top 50 negative (non-activating) sentences

**Example**:
```csv
feature_id,label,label_score,positive_sentences,negative_sentences
22981,"Analyst sentiment, specific reference: Fed",0.75,"said he believes analysts are ""Fed watching way too|...","company figures to have a bright future in AI as its|..."
35158,"Post-earnings reactions, specific reference: earnings",0.72,"indexes dropped last week afterinflation came in hotter|...","sell-off in artificial intelligence (AI) stocks. Arm|..."
```

### AutoInterp SAEEval Output

**CSV Summary** (`Results/nemotron_layer28_features_summary_<timestamp>.csv`):
```csv
feature,label,autointerp_score
2189,Earnings Reports and Financial Updates,0.7143
2330,CEO Earnings Updates,0.5714
2485,Stock market performance updates,0.7143
10628,Earnings Call Transcripts,0.9286
```

**Interpreting Scores**:
- **0.85+**: Excellent - explanation accurately predicts activations
- **0.70-0.85**: Good - explanation is mostly reliable
- **0.50-0.70**: Moderate - some predictive power
- **<0.50**: Poor - explanation doesn't reliably predict activations

### AutoInterp Full Output

```csv
feature,label,F1_score,precision,recall
27,"-ing" forms,0.745,0.82,0.68
220,Conceptual ideas and alternatives,0.527,0.61,0.46
```

**Quality thresholds**: F1 > 0.7 (good), Precision > 0.8 (reliable), Recall > 0.6 (catches cases well)

## Additional Components

### `Addtional/` Folder

The `Addtional/` folder contains supplementary tools and archived implementations:

- **`autointerp_advanced/`**: Advanced labeling implementation with SaeVisRunner
- **`autointerp_full_local/`**: Local version of AutoInterp Full
- **`autointerp_steer/`**: Feature steering analysis (also in main directory)
- **`feature_search_lite/`**: Lightweight feature search implementation
- **`archive/`**: Historical versions and experimental implementations
  - `autointerp_full_finance/`: Finance-specific AutoInterp Full
  - `autointerp_full_optimized_finbert/`: Optimized FinBERT version
  - `autointerp_full_reasoning/`: Reasoning-specific version
  - `autointerp_lite_plus/`: Enhanced AutoInterp Lite

**Key Documentation**:
- `QUICK_START.md`: Quick start guide for generic system
- `GENERIC_SYSTEM_README.md`: Generic system documentation
- `MULTI_LAYER_README.md`: Multi-layer analysis documentation
- `RESTRUCTURE_SUMMARY.md`: Restructuring information

**Note**: These are supplementary tools. For most use cases, use the main `search_autointerp/` pipeline.

## Repository Structure

```
autointerp/
├── search_autointerp/          # ⭐ RECOMMENDED - Search + label pipeline
│   ├── 1. search/              # Feature search (activation-separation)
│   ├── 2. autointerp_lite/     # Basic labeling (recommended)
│   ├── 3. autointerp_advance/  # Advanced labeling (optional)
│   ├── pipeline.py             # Main entry point
│   └── nemotron_finance_detailed_labels.csv  # Example output
│
├── autointerp_saeeval/         # Standalone evaluation tool
│   ├── autointerp/             # Core evaluation module
│   ├── run_finbert.py          # FinBERT evaluation
│   ├── run_nemotron_top100_finance_eval.py
│   └── Results/                # Evaluation outputs
│
├── autointerp_full/            # Production LLM explanations (not recommended)
│   ├── run_finbert.sh
│   ├── run_nemotron.sh
│   ├── prompts.yaml
│   └── prompts_finance.yaml
│
├── autointerp_lite/            # Fast feature discovery
│   └── run_analysis.py
│
├── autointerp_steer/           # Feature steering analysis
│   └── scripts/run_steering.py
│
├── Addtional/                  # Supplementary tools
│   ├── autointerp_advanced/
│   ├── autointerp_full_local/
│   ├── feature_search_lite/
│   └── archive/
│
└── archive/                    # Historical versions (reference only)
```

## Key Configuration Files

- **`search_autointerp/pipeline.py`** - Main pipeline script
- **`search_autointerp/1. search/test_tokens.json`** - Example token file for domain matching
- **`autointerp_full/prompts.yaml`** - Domain-agnostic prompts
- **`autointerp_full/prompts_finance.yaml`** - Finance-specific prompts
- **`.gitignore`** - Configured to allow CSV files and archive folder

## Documentation Links

- [search_autointerp/README.md](search_autointerp/README.md) - Complete Search AutoInterp documentation
- [autointerp_saeeval/README.md](autointerp_saeeval/README.md) - AutoInterp SAEEval documentation
- [autointerp_full/README.md](autointerp_full/README.md) - AutoInterp Full documentation
- [autointerp_lite/README.md](autointerp_lite/README.md) - AutoInterp Lite documentation
- [autointerp_steer/README.md](autointerp_steer/README.md) - AutoInterp Steer documentation

---

## Quick Decision Guide

- **⭐ Need to search and label features?** → **Search AutoInterp** (recommended)
- **Need model-agnostic evaluation?** → AutoInterp SAEEval
- **Need fast feature discovery?** → AutoInterp Lite
- **Need intervention experiments?** → AutoInterp Steer
- **Need detailed explanations with metrics?** → AutoInterp Full (not recommended - slow, generic output)

## Recommended Workflow

1. **Discovery & Labeling**: `search_autointerp/` → Search and label features simultaneously (2-5 min)
2. **Evaluation** (optional): `autointerp_saeeval/` → Compare across models or validate labels
3. **Intervention** (optional): `autointerp_steer/` → Understand feature effects through steering

**For most users**: Start with `search_autointerp/pipeline.py` - it provides the best balance of speed, quality, and ease of use.
