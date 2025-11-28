# Feature Search & AutoInterp: Complete Pipeline

A training-free approach to identify and label sparse features aligned to reasoning, finance, healthcare, science, math, tool-calling, or other domain concepts.

## New Structure

The codebase is organized into three main groups:

1. **`1. search/`** - Feature search functionality
   - Identifies domain-specific features using activation-separation-based scoring
   - See [1. search/README.md](1.%20search/README.md) for details

2. **`2. autointerp_lite/`** - Basic labeling pipeline
   - Collects positive/negative context windows
   - Generates labels using vLLM API
   - See [2. autointerp_lite/README.md](2.%20autointerp_lite/README.md) for details

3. **`3. autointerp_advance/`** - Advanced labeling pipeline
   - Extracts examples using SaeVisRunner
   - Generates labels from extracted examples
   - See [3. autointerp_advance/README.md](3.%20autointerp_advance/README.md) for details

## Quick Start - Complete Pipeline

```bash
# Run all three steps seamlessly
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path 1.search/test_tokens.json \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100
```

Results are saved to `results/`:
- `results/1_search/` - Feature search results
- `results/2_labeling_lite/` - Basic labeling results
- `results/3_labeling_advance/` - Advanced labeling results

## Running Individual Steps

```bash
# Step 1: Search
cd "1. search"
python run_search.py --model_path ... --sae_path ... --output_dir ../results/1_search

# Step 2: Basic Labeling
cd "../2. autointerp_lite"
python run_labeling.py --search_output ../results/1_search

# Step 3: Advanced Labeling
cd "../3. autointerp_advance"
python run_labeling_advanced.py --model_path ... --sae_path ... --search_output ../results/1_search
```

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Scoring Methods](#scoring-methods) ⭐ **Important**
4. [Quick Start](#quick-start)
5. [Core Files & Parameters](#core-files--parameters)
6. [Domain Examples](#domain-examples)
7. [Token Files](#token-files)
8. [Directory Structure](#directory-structure)

---

## Overview

This module implements a **training-free, activation-separation-based** method to discover which SAE (Sparse Autoencoder) features encode specific domain concepts. It directly measures how feature activation distributions shift between examples where a concept is present vs. absent.

**Key Features:**
- ✅ No training required (no probes, no classifiers)
- ✅ Works with 20-100 examples per concept
- ✅ Domain-agnostic (finance, reasoning, healthcare, science, math, tool-calling)
- ✅ Multiple scoring methods (simple, Fisher-style, domain-specific)
- ✅ Compatible with any model (Llama, Gemma, Mistral, DeepSeek, etc.)

---

## How It Works

### Step 1: Identify Concept Presence
- Use token file to find C⁺ (text with domain tokens) and C⁻ (text without)
- `expand_range` includes surrounding context around matched tokens

### Step 2: Extract Activations
- Process dataset through model and SAE
- Track feature activations separately for C⁺ and C⁻

### Step 3: Compute Separation Scores
For each feature, measure activation difference between C⁺ and C⁻ distributions.

### Step 4: Select Top Features
- Choose top N features (`num_features`) or by quantile threshold

---

## Scoring Methods

Three scoring methods measure how strongly each feature encodes your concept:

### 1. Simple Score (`score_type="simple"`)
**Formula**: `Score(i) = |μ⁺ᵢ - μ⁻ᵢ|`

Absolute difference in mean activations. Example: feature activates at 5.0 on financial text and 1.0 on general text → score = |5.0 - 1.0| = 4.0

**Use for**: Quick exploration, simple interpretability

### 2. Fisher Score (`score_type="fisher"`) ⭐ **Recommended**
**Formula**: `Score(i) = (μ⁺ᵢ - μ⁻ᵢ)² / (σ⁺ᵢ + σ⁻ᵢ + ε)`

Variance-normalized separation. Accounts for consistency - features with lower variance score higher. Example: Feature A (μ⁺=5.0, μ⁻=1.0, σ=0.5) scores 16.0, while Feature B (same means, σ=2.0) scores 4.0.

**Use for**: Production analysis, comparing features, most robust results

### 3. Domain Score (`score_type="domain"`)
**Formula**: `Score(i) = (mean_pos / sum_pos) * h_norm^α - (mean_neg / sum_neg)`

Token-based with entropy penalty. Features that activate on many tokens (high entropy) get penalized; specific features (low entropy) score higher.

**Use for**: Token-based matching when specificity matters

**Recommendation**: Use `fisher` for most cases - it's the most stable and widely applicable.

---

## Quick Start

**No domain folders needed!** Just specify any HuggingFace dataset and a token file:

```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path my_tokens.json \
    --output_dir ./results \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 100 \
    --n_samples 10000 \
    --generate_dashboard
```

**What you need:**
1. **Any HuggingFace dataset** (e.g., `jyanimaulik/yahoo_finance_stockmarket_news`, `allenai/sciq`, `lighteval/MATH`)
2. **A token JSON file** with domain keywords (see examples below)
3. **Model and SAE paths**

**Output:**
- `feature_scores.pt` - All feature scores
- `top_features.pt` - Top feature indices
- `feature_list.json` - Feature list with scores and metadata
- `dashboards/features-100.html` - Interactive dashboard (if enabled)

---

## Core Files & Parameters

### Main Files

#### `main/run_feature_search.py`
**Purpose**: Unified command-line interface for complete feature search pipeline

**Key Parameters:**
- `model_path` (required): HuggingFace model path or local path
- `sae_path` (required): SAE path (HuggingFace repo or local directory)
- `dataset_path` (required): HuggingFace dataset path or local path
- `tokens_str_path` (required): JSON file with domain-specific token strings
- `output_dir` (required): Directory to save results
- `sae_id`: SAE identifier (e.g., `"blocks.19.hook_resid_post"`)
- `score_type`: `"simple"`, `"fisher"`, or `"domain"` (default: `"domain"`)
- `num_features`: Number of top features to return (default: 100)
- `selection_method`: `"topk"` or `"quantile"` (default: `"topk"`)
- `n_samples`: Number of dataset samples to process (default: 4096)
- `expand_range`: Tuple `(left, right)` to expand context around matched tokens (e.g., `2,3` for 2 before, 3 after)
- `alpha`: Entropy weight for domain scoring (default: 1.0, only used with `score_type="domain"`)
- `generate_dashboard`: Whether to generate HTML dashboard (default: False)

**Full parameter list**: Run `python main/run_feature_search.py --help`

#### `main/compute_score.py`
**Purpose**: Core scoring implementation

**Key Classes:**
- `RollingMean`: Tracks mean and variance for positive/negative distributions
- `FeatureStatisticsGenerator`: Accumulates statistics across batches
- `SaeSelectionRunner`: Orchestrates feature scoring

#### `main/compute_dashboard.py`
**Purpose**: Generate interactive HTML dashboards for top features

---

## Domain Examples

### 1. Finance

**Dataset**: `jyanimaulik/yahoo_finance_stockmarket_news` (HuggingFace)  
**Token File**: Create `finance_tokens.json`:
```json
[
  "stock", " price", "market", " earnings", "revenue", "profit",
  "dividend", " IPO", "NASDAQ", "NYSE", "trading", " investment",
  "portfolio", " shares", "equity", " valuation"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path finance_tokens.json \
    --output_dir ./results/finance \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 50 \
    --expand_range 2,3 \
    --n_samples 10000
```

### 2. Reasoning

**Dataset**: `allenai/OpenThoughts` (HuggingFace)  
**Token File**: Create `reasoning_tokens.json`:
```json
[
  "therefore", " therefore", "Therefore", " Therefore",
  "however", " however", "However", " However",
  "because", " because", "Because", " Because",
  "since", " since", "Since", " Since",
  "implies", " implies", "Implies", " Implies",
  "follows", " follows", "Follows", " Follows"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path allenai/OpenThoughts \
    --tokens_str_path reasoning_tokens.json \
    --output_dir ./results/reasoning \
    --sae_id blocks.19.hook_resid_post \
    --score_type simple \
    --num_features 100 \
    --expand_range 1,2 \
    --n_samples 10000
```

### 3. Healthcare

**Dataset**: `bigbio/medqa` (HuggingFace)  
**Token File**: Create `healthcare_tokens.json`:
```json
[
  "diagnosis", " treatment", "symptom", " patient",
  "disease", " medication", "therapy", " prognosis",
  "clinical", " medical", "physician", " hospital"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path bigbio/medqa \
    --tokens_str_path healthcare_tokens.json \
    --output_dir ./results/healthcare \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 75 \
    --expand_range 2,3 \
    --n_samples 10000
```

### 4. Science

**Dataset**: `allenai/sciq` (HuggingFace)  
**Token File**: Create `science_tokens.json`:
```json
[
  "hypothesis", " experiment", "observation", " theory",
  "evidence", " methodology", "results", " conclusion",
  "analysis", " data", "research", " study"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path allenai/sciq \
    --tokens_str_path science_tokens.json \
    --output_dir ./results/science \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 80 \
    --expand_range 2,2 \
    --n_samples 10000
```

### 5. Mathematics

**Dataset**: `lighteval/MATH` (HuggingFace)  
**Token File**: Create `math_tokens.json`:
```json
[
  "equation", " solve", "proof", " theorem",
  "calculate", " formula", "variable", " coefficient",
  "derivative", " integral", "matrix", " vector"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path lighteval/MATH \
    --tokens_str_path math_tokens.json \
    --output_dir ./results/math \
    --sae_id blocks.19.hook_resid_post \
    --score_type simple \
    --num_features 100 \
    --expand_range 1,2 \
    --n_samples 10000
```

### 6. Tool-Calling

**Dataset**: `ToolBench/ToolBench_II` (HuggingFace)  
**Token File**: Create `tool_tokens.json`:
```json
[
  "function", " API", "call", " parameter",
  "execute", " result", "error", " response",
  "tool", " method", "endpoint", " request"
]
```

**Command:**
```bash
python main/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path ToolBench/ToolBench_II \
    --tokens_str_path tool_tokens.json \
    --output_dir ./results/tool_calling \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 60 \
    --expand_range 2,3 \
    --n_samples 10000
```

---

## Directory Structure

```
search_autointerp/
├── 1. search/                 # Feature search functionality
│   ├── main/                  # Core search implementation
│   │   ├── run_feature_search.py
│   │   └── compute_score.py
│   ├── domains/               # Domain-specific wrappers
│   │   └── finance/
│   ├── run_search.py          # Unified entry point
│   └── README.md
│
├── 2. autointerp_lite/         # Basic labeling pipeline
│   ├── collect_examples.py    # Collect context windows
│   ├── label_features.py      # Generate labels (vLLM)
│   ├── run_labeling.py        # Unified entry point
│   └── README.md
│
├── 3. autointerp_advance/      # Advanced labeling pipeline
│   ├── extract_examples.py    # Extract using SaeVisRunner
│   ├── generate_labels.py    # Generate labels from examples
│   ├── run_labeling_advanced.py  # Unified entry point
│   └── README.md
│
├── results/                   # All results saved here
│   ├── 1_search/             # Search results
│   ├── 2_labeling_lite/      # Lite labeling results
│   └── 3_labeling_advance/   # Advanced labeling results
│
├── pipeline.py               # Seamless chaining script
├── archive/                  # Archived files
└── README.md                 # This file
```

### About Domain Folders (Optional - Can Be Removed)

The `domains/` folder contains **optional convenience wrappers** that:
- Load settings from `config.json` files
- Call the main `run_feature_search.py` function
- Organize outputs in domain-specific folders

**You don't need them!** The main script (`main/run_feature_search.py`) accepts all parameters directly via command line. Domain folders are just for:
- **Convenience**: Store configs in JSON instead of typing long commands
- **Organization**: Keep domain outputs together

**To use the system, you only need:**
1. **Any HuggingFace dataset** (specify with `--dataset_path`)
2. **A token JSON file** (specify with `--tokens_str_path`)
3. **Run `main/run_feature_search.py`** with your parameters

The system is fully domain-agnostic - no special folders or configs needed!

---

## Token Files

**Token files define which parts of your dataset contain the concept you're looking for.**

### Why Token Files?

To find features that encode a concept (e.g., "financial reasoning"), we need to compare:
- **C⁺ (Positive)**: Token positions in the dataset that match your domain vocabulary
- **C⁻ (Negative)**: All other token positions in the same dataset

**Important**: Both positive and negative positions come from the **same dataset**. The token file acts as a filter to separate them.

### How Token Files Work (ReasonScore-style)

1. System processes your entire dataset
2. **Positions matching tokens in your file** → C⁺ (positive)
3. **All other positions in the dataset** → C⁻ (negative)
4. `expand_range` parameter includes surrounding context (e.g., 1 token before, 2 after)

**Example from ReasonScore** ([source](https://github.com/AIRI-Institute/SAE-Reasoning)):
- Dataset: `OpenThoughts-10k-DeepSeek-R1` (reasoning dataset)
- Token file: `reason_tokens.json` with words like "alternatively", "hmm", "maybe", "therefore", "however"
- Positive: Positions in the dataset where these reasoning words appear
- Negative: All other positions in the same dataset

**Why this works:**
- Domain-specific text contains domain-specific words (e.g., financial text has "stock", "earnings", "revenue")
- General text doesn't contain these terms
- Features that activate more on domain terms likely encode domain concepts

### Token File Format

**Token files can be anywhere on your system.** Create a JSON file with domain-specific keywords:

**Example** (`finance_tokens.json`):
```json
[
  "stock", " price", "market", " earnings", "revenue", "profit",
  "dividend", " IPO", "NASDAQ", "NYSE", "trading", " investment"
]
```

**Tips:**
- Include variations with/without leading spaces and different cases (tokenizers may tokenize them differently)
  - Example: `"stock"`, `" stock"`, `"Stock"`, `" Stock"`
- More tokens = broader coverage, but too many may reduce specificity
- Choose tokens that clearly distinguish your domain from general text
- Token files are simple JSON arrays - no special structure needed

---

## References

- **ReasonScore**: Official implementation from [AIRI-Institute/SAE-Reasoning](https://github.com/AIRI-Institute/SAE-Reasoning) - "I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders" ([arXiv:2503.18878](https://arxiv.org/abs/2503.18878))
- **SAE Lens**: Sparse Autoencoder implementation
- **Autointerp**: Feature interpretation and explanation
