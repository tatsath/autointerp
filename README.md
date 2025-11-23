# AutoInterp - SAE Interpretability System

Comprehensive toolkit for understanding what features in your Sparse Autoencoder (SAE) model have learned. Multiple complementary approaches for feature discovery, explanation, and evaluation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Folder Descriptions](#folder-descriptions)
4. [Detailed Comparison](#detailed-comparison)
   - [AutoInterp Full vs AutoInterp SAEEval vs AutoInterp Lite](#autointerp-full-vs-autointerp-saeeval-vs-autointerp-lite)
5. [When to Use Which](#when-to-use-which)
6. [Installation](#installation)
7. [Running Examples](#running-examples)
8. [Sample Results](#sample-results)
9. [Repository Structure](#repository-structure)

## Overview

AutoInterp provides five main approaches for SAE feature interpretability:

- **AutoInterp Lite**: Fast feature discovery (2-5 min) - Find domain-relevant features quickly
- **AutoInterp Full**: LLM-based detailed explanations (30-60 min/feature) - Understand what features actually do
- **AutoInterp SAEEval**: Standalone evaluation tool - Model-agnostic feature explanation
- **AutoInterp Steer**: Feature steering analysis - Understand features through controlled intervention
- **Feature Search**: Training-free concept discovery - Identify features aligned to domain concepts (finance, reasoning, healthcare, etc.)

## Quick Start

```bash
# Install
pip install -e .

# Find relevant features (Step 1)
cd autointerp_lite
python run_analysis.py --mode financial

# Get detailed explanations (Step 2)
cd autointerp_full
./run_finbert.sh  # or ./run_nemotron.sh, ./run_llama_all.sh
```

## Folder Descriptions

### `autointerp_full/` - Production LLM-Based Feature Explanation
**Purpose**: Generate human-readable explanations with confidence scores using LLM analysis.

**Key Features**:
- LLM-based feature explanation with F1, precision, recall metrics
- External YAML prompt configuration (`prompts.yaml`, `prompts_finance.yaml`)
- FAISS-based contrastive learning for better explanations
- Cache management for efficient re-runs
- Multiple explainer providers (vLLM, OpenRouter, offline)
- Short script names: `run_finbert.sh`, `run_nemotron.sh`, `run_llama_all.sh`

**Output**: Detailed explanations with confidence scores, F1 metrics, CSV summaries

### `autointerp_saeeval/` - Standalone SAE Evaluation Tool
**Purpose**: Model-agnostic tool for automatically producing human-readable explanations for SAE features.

**Key Features**:
- Domain-agnostic and model-agnostic evaluation
- Works with any LLM (Nemotron, GPT-OSS, Llama, Gemma, FinBERT, etc.)
- Three-stage process: activation evidence collection, LLM explanation, evaluation/scoring
- Standalone scripts: `run_autointerp_features.py`, `run_nemotron_autointerp_vllm.py`

**Output**: Interpretability scores, natural-language labels, CSV summaries

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

### `feature_search/` - Training-Free Concept Feature Discovery
**Purpose**: Training-free approach to identify sparse features aligned to reasoning, finance, healthcare, science, math, or other domain concepts using activation-separation-based scoring.

**Key Features**:
- No training required (no probes, no classifiers)
- Works with 20-100 examples per concept
- Multiple scoring methods (simple, Fisher-style, domain-specific)
- Domain-agnostic - works with any HuggingFace dataset
- Unified CLI interface (`main/run_feature_search.py`)

**How it works**:
1. Define domain tokens (e.g., financial keywords)
2. System identifies C⁺ (text with tokens) vs C⁻ (text without)
3. Computes activation separation scores (Fisher recommended)
4. Selects top features encoding the concept

**Usage**:
```bash
cd feature_search
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path finance_tokens.json \
    --output_dir ./results \
    --score_type fisher \
    --num_features 100
```

**Output**: Feature scores, top feature indices, feature list JSON, optional HTML dashboard

### `archive/` - Historical Versions
Contains archived versions: `autointerp_full_finance/`, `autointerp_full_optimized_finbert/`, `autointerp_full_reasoning/`, etc. Reference only.

## Detailed Comparison

### AutoInterp Full vs AutoInterp SAEEval vs AutoInterp Lite

| Feature | AutoInterp Full | AutoInterp SAEEval | AutoInterp Lite |
|---------|----------------|-------------------|----------------|
| **Primary Use** | Production LLM explanations | Standalone evaluation | Quick feature discovery |
| **Speed** | 30-60 min/feature | Variable (depends on setup) | 2-5 min for 1000+ features |
| **Output Detail** | Detailed explanations + F1 scores | Natural-language labels + scores | Specialization scores |
| **LLM Required** | Yes (chat models) | Yes (any LLM) | Optional (for labeling) |
| **Configuration** | YAML prompts, cache management | Script-based | Command-line args |
| **Contrastive Learning** | ✅ FAISS-based | ❌ | ❌ |
| **Model Support** | Any with chat format | Any LLM (model-agnostic) | Any base model |
| **Best For** | Research, validation, detailed analysis | Standalone evaluation, model comparison | Initial exploration, screening |

#### AutoInterp Full - Deep Dive

**Strengths**:
- Most comprehensive explanation system
- FAISS contrastive learning improves explanation quality
- External prompt configuration (YAML) for easy customization
- Cache management for efficient re-runs
- Multiple provider support (vLLM, OpenRouter, offline)
- Production-ready with comprehensive metrics

**Limitations**:
- Slowest (30-60 minutes per feature)
- Requires chat-formatted models
- More complex setup

**Key Scripts**:
```bash
./run_finbert.sh              # FinBERT analysis (top 100 features)
./run_nemotron.sh              # Nemotron analysis (top 100 features)
./run_llama_all.sh             # Llama-3.1-8B all features
./run_test_cache.sh            # Cache testing
```

**Configuration Files**:
- `prompts.yaml` - Domain-agnostic prompts
- `prompts_finance.yaml` - Finance-specific prompts (8-15 words regular, 5-7 contrastive)

#### AutoInterp SAEEval - Deep Dive

**Strengths**:
- Model-agnostic (works with any LLM)
- Domain-agnostic (finance, sports, legal, etc.)
- Standalone tool (no complex dependencies)
- Three-stage evaluation process
- Good for comparing different models

**Limitations**:
- Less detailed than AutoInterp Full
- No contrastive learning
- Script-based configuration (less flexible)

**Key Scripts**:
```bash
run_autointerp_features.py              # General evaluation
run_nemotron_autointerp_vllm.py         # Nemotron-specific
run_autointerp_features_vllm_finbert.py  # FinBERT-specific
```

#### AutoInterp Lite - Deep Dive

**Strengths**:
- Fastest option (2-5 minutes)
- Simple command-line interface
- Good for initial screening
- Optional LLM labeling
- Domain specialization scoring

**Limitations**:
- Less detailed explanations
- No confidence scoring
- Basic metrics only

**Usage**:
```bash
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling
```

## When to Use Which

### Use AutoInterp Lite when:
- ✅ You have thousands of features and need to find 10-50 relevant ones
- ✅ You want quick initial exploration
- ✅ You need domain specialization scores
- ✅ Speed is more important than detailed explanations

**Example**: "I have 1000 features, which 20 are most relevant for finance?"

### Use AutoInterp Full when:
- ✅ You need detailed explanations with confidence scores
- ✅ You want F1, precision, recall metrics
- ✅ You're doing research or validation
- ✅ You need production-ready analysis
- ✅ You want contrastive learning for better quality

**Example**: "I found feature 133 is financial. What exactly does it detect - earnings reports or market volatility?"

### Use AutoInterp SAEEval when:
- ✅ You need a standalone evaluation tool
- ✅ You're comparing different models
- ✅ You want model-agnostic analysis
- ✅ You need domain-agnostic evaluation
- ✅ You prefer script-based configuration

**Example**: "I want to evaluate SAE features across Nemotron, Llama, and Gemma models."

### Use AutoInterp Steer when:
- ✅ You want to understand features through intervention
- ✅ You need to see how features affect generation
- ✅ You're doing controlled experiments
- ✅ You want activation steering analysis

**Example**: "How does feature 133 affect model generation when steered?"

### Use Feature Search when:
- ✅ You need training-free concept discovery
- ✅ You want to find features for specific domains (finance, reasoning, healthcare, etc.)
- ✅ You have domain-specific datasets and token lists
- ✅ You prefer activation-separation over probe-based methods
- ✅ You need fast, transparent feature identification

**Example**: "Which features encode financial reasoning concepts in my SAE?"

### Recommended Workflow

1. **Discovery**: `autointerp_lite/` or `feature_search/` → Find relevant features (2-5 min)
2. **Explanation**: `autointerp_full/` → Get detailed explanations (30-60 min/feature)
3. **Evaluation** (optional): `autointerp_saeeval/` → Compare across models
4. **Intervention** (optional): `autointerp_steer/` → Understand feature effects

## Installation

```bash
# From main directory
pip install -e .

# For visualization support (optional)
pip install -e ".[visualize]"
```

**Requirements**:
- Python 3.10+
- SAE Model files
- Base Model (e.g., `meta-llama/Llama-2-7b-hf`)
- Chat model for explanations (for AutoInterp Full/SAEEval)

## Running Examples

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

### AutoInterp Full
```bash
cd autointerp_full

# FinBERT analysis
./run_finbert.sh

# Nemotron analysis
./run_nemotron.sh

# Llama all features
./run_llama_all.sh

# Custom run
python -m autointerp_full \
    meta-llama/Llama-2-7b-hf \
    /path/to/sae/model \
    --hookpoints layers.16 \
    --n_tokens 50000 \
    --max_latents 20 \
    --explainer_model "openai/gpt-3.5-turbo" \
    --name my_analysis
```

### AutoInterp SAEEval
```bash
cd autointerp_saeeval

# General evaluation
python run_autointerp_features.py

# Nemotron-specific
python run_nemotron_autointerp_vllm.py

# FinBERT-specific
python run_autointerp_features_vllm_finbert.py
```

### AutoInterp Steer
```bash
conda activate sae
cd autointerp_steer
python scripts/run_steering.py --output_folder steering_outputs
```

### Feature Search
```bash
cd feature_search
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path finance_tokens.json \
    --output_dir ./results \
    --score_type fisher \
    --num_features 100 \
    --n_samples 10000
```

## Sample Results

### AutoInterp Lite Output
| Feature | Label | Specialization | Domain Activation | General Activation |
|---------|-------|----------------|-------------------|-------------------|
| 133 | Earnings Reports Rate Changes Announcements | **19.56** | 96.73 | 116.29 |
| 162 | value changes performance indicators | **9.58** | 48.20 | 57.78 |
| 203 | Record performance revenue reports | **8.85** | 40.66 | 49.51 |

**Good features**: Specialization > 3.0, Specialization Confidence > 30.0

### AutoInterp Full Output
| Feature | Label | F1 Score | Precision | Recall | Explanation |
|---------|-------|----------|-----------|--------|-------------|
| 27 | "-ing" forms | 0.745 | 0.82 | 0.68 | Detects sentences containing "-ing" verb forms and gerunds |
| 220 | Conceptual ideas and alternatives | 0.527 | 0.61 | 0.46 | Identifies abstract concepts and alternative possibilities |

**Quality thresholds**: F1 > 0.7 (good), Precision > 0.8 (reliable), Recall > 0.6 (catches cases well)

### AutoInterp SAEEval Output
Similar to AutoInterp Full but with model-agnostic evaluation. Provides interpretability scores and natural-language labels for any LLM/SAE combination.

## Repository Structure

```
autointerp/
├── autointerp_full/          # Production LLM explanations (PRIMARY)
│   ├── run_finbert.sh
│   ├── run_nemotron.sh
│   ├── run_llama_all.sh
│   ├── prompts.yaml
│   └── prompts_finance.yaml
├── autointerp_saeeval/       # Standalone evaluation tool
│   ├── run_autointerp_features.py
│   └── run_nemotron_autointerp_vllm.py
├── autointerp_lite/          # Fast feature discovery
│   └── run_analysis.py
├── autointerp_steer/         # Feature steering analysis
│   └── scripts/run_steering.py
├── feature_search/           # Training-free concept discovery
│   ├── main/                 # Core functionality
│   │   ├── run_feature_search.py  # Unified CLI
│   │   ├── compute_score.py
│   │   └── compute_dashboard.py
│   └── domains/              # Optional domain examples
└── archive/                  # Historical versions (reference only)
```

## Key Configuration Files

- **`autointerp_full/prompts.yaml`** - Domain-agnostic prompts
- **`autointerp_full/prompts_finance.yaml`** - Finance-specific prompts (8-15 words regular, 5-7 contrastive)
- **`.gitignore`** - Configured to allow CSV files and archive folder

## Documentation Links

- [autointerp_full/README.md](autointerp_full/README.md) - Complete AutoInterp Full documentation
- [autointerp_saeeval/README.md](autointerp_saeeval/README.md) - AutoInterp SAEEval documentation
- [autointerp_lite/README.md](autointerp_lite/README.md) - AutoInterp Lite documentation
- [autointerp_steer/README.md](autointerp_steer/README.md) - AutoInterp Steer documentation
- [feature_search/README.md](feature_search/README.md) - Feature Search documentation

---

**Quick Decision Guide**:
- **Need speed?** → AutoInterp Lite or Feature Search
- **Need detailed explanations?** → AutoInterp Full
- **Need model-agnostic evaluation?** → AutoInterp SAEEval
- **Need intervention experiments?** → AutoInterp Steer
- **Need training-free concept discovery?** → Feature Search
