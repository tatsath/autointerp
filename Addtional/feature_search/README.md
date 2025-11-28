# Feature Search: Training-Free Concept Feature Discovery

A training-free approach to identify sparse features aligned to reasoning, finance, healthcare, science, math, or other domain concepts using activation-separation-based scoring.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Folder Structure](#folder-structure)
4. [Creating Your Own Domain](#creating-your-own-domain)
5. [Scoring Methods](#scoring-methods)

---

## Overview

This module implements a **training-free, activation-separation-based** method to discover which SAE (Sparse Autoencoder) features encode specific domain concepts. It directly measures how feature activation distributions shift between examples where a concept is present vs. absent.

**Key Features:**
- ✅ No training required (no probes, no classifiers)
- ✅ Works with 20-100 examples per concept
- ✅ Domain-agnostic (finance, reasoning, healthcare, science, math, tool-calling)
- ✅ Multiple scoring methods (simple, Fisher-style, domain-specific)

---

## Quick Start

### Using Domain-Specific Scripts

**Finance:**
```bash
cd domain_finance
python run_search.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>
```

**Reasoning:**
```bash
cd domain_reasoning
python run_search.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>
```

### Using Common Interface

```bash
python domain_common/run_feature_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path domain_finance/tokens.txt \
    --output_dir domain_finance/results \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 100
```

---

## Folder Structure

```
feature_search/
├── domain_finance/      # Finance domain configuration and scripts
│   ├── config.yaml      # Finance-specific configuration
│   ├── tokens.txt       # Finance domain tokens (one per line)
│   ├── run_search.py    # Finance feature search script
│   └── results/         # Finance results (feature scores, top features, etc.)
│
├── domain_reasoning/    # Reasoning domain configuration and scripts
│   ├── config.yaml      # Reasoning-specific configuration
│   ├── tokens.txt       # Reasoning domain tokens
│   ├── run_search.py    # Reasoning feature search script
│   └── results/         # Reasoning results
│
├── domain_common/       # Core functionality (shared across domains)
│   ├── run_feature_search.py  # Main entry point
│   ├── compute_score.py       # Feature scoring implementation
│   └── compute_dashboard.py   # Dashboard generation
│
└── autointerp/          # Feature interpretation and labeling
    ├── compute_examples_finance.py  # Extract activating examples
    ├── generate_labels_from_examples.py  # Generate feature labels
    └── result/          # Autointerp results
```

---

## Creating Your Own Domain

### Step 1: Create Domain Folder

```bash
mkdir -p domain_yourdomain/results
```

### Step 2: Create Configuration

Create `domain_yourdomain/config.yaml`:

```yaml
dataset_path: "your/dataset/path"
tokens_str_path: "tokens.txt"
expand_range: [1, 2]
alpha: 0.7
score_type: "fisher"
num_features: 100
model_path: "meta-llama/Llama-3.1-8B-Instruct"
sae_path: "/path/to/sae"
sae_id: "blocks.19.hook_resid_post"
output_dir: "results"
```

### Step 3: Create Token File

Create `domain_yourdomain/tokens.txt` with domain-specific keywords (one per line):

```
keyword1
keyword2
keyword3
```

### Step 4: Create Run Script

Copy `domain_finance/run_search.py` to `domain_yourdomain/run_search.py` and update the domain name.

### Step 5: Run

```bash
cd domain_yourdomain
python run_search.py --model_path <model> --sae_path <sae_path> --sae_id <sae_id>
```

Results will be saved in `domain_yourdomain/results/`.

---

## Scoring Methods

1. **Simple Score** (`score_type="simple"`): `|μ⁺ - μ⁻|` - Absolute difference in mean activations
2. **Fisher Score** (`score_type="fisher"`): `(μ⁺ - μ⁻)² / (σ⁺ + σ⁻ + ε)` - Variance-normalized separation ⭐ **Recommended**
3. **Domain Score** (`score_type="domain"`): Token-based with entropy penalty for specificity

---

## References

- **ReasonScore**: [AIRI-Institute/SAE-Reasoning](https://github.com/AIRI-Institute/SAE-Reasoning) - "I Have Covered All the Bases Here: Interpreting Reasoning Features in Large Language Models via Sparse Autoencoders"
