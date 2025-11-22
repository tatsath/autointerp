# AutoInterp - SAE Interpretability System

Two complementary approaches for understanding what features in your SAE model have learned.

## 🎯 What This Does

Four complementary approaches for understanding what features in your SAE model have learned.

**Step 1:** Use AutoInterp Lite to quickly find relevant features and get basic labels. **Step 2:** Use AutoInterp Lite Plus for comprehensive analysis with advanced metrics. **Step 3:** Use AutoInterp Full with the exact feature numbers from Step 1 to get detailed explanations with confidence scores. **Step 4:** Use AutoInterp Steer to analyze features through activation steering and intervention experiments.

### 1. AutoInterp Lite - Feature Discovery
**Find relevant features in minutes, not hours**

- **Why needed**: SAE models have thousands of features. You need to find the 10-50 that matter for your domain (finance, healthcare, legal, etc.).
- **What it does**: Compares feature activations on domain-specific vs general text
- **Key metrics**: Activation strength, specialization score, top activating examples
- **Speed**: 2-5 minutes for 1000+ features
- **Output**: Ranked list of domain-relevant features with activation examples

### 2. AutoInterp Lite Plus - Comprehensive Analysis
**Advanced metrics with clustering and polysemanticity analysis**

- **Why needed**: Get deeper insights into feature quality with comprehensive metrics including clustering, polysemanticity, F1 scores, and robustness testing
- **What it does**: Enhanced version of AutoInterp Lite with advanced clustering analysis, classification metrics, and quality assessment
- **Key metrics**: F1 score, precision, recall, selectivity, clustering, polysemanticity, robustness
- **Speed**: 5-10 minutes for comprehensive analysis
- **Output**: Detailed quality assessment table with feature rankings and recommendations

### 3. AutoInterp Full - Feature Explanation  
**Understand what your features actually do**

- **Why needed**: Knowing a feature is "financial" isn't enough - you need to know if it detects "earnings reports" vs "market volatility"
- **What it does**: Uses LLMs to generate human-readable explanations with confidence scores
- **Key metrics**: F1 score, precision, recall, explanation quality
- **Speed**: 30-60 minutes per feature (due to LLM analysis)
- **Output**: Detailed explanations with confidence scores and validation

### 4. AutoInterp Steer - Feature Steering Analysis
**Analyze features through controlled activation intervention**

- **Why needed**: Understand feature functions by observing how steering affects model generation (implementation of Kuznetsov et al., 2025)
- **What it does**: Implements activation steering (`x' = x + λ·A_max·d_i`) to generate multiple outputs with varying steering strengths
- **Key features**: Maximum activation estimation, steering with varying λ values, text generation with feature intervention
- **Speed**: Depends on number of features and prompts (typically 10-30 minutes per feature)
- **Output**: Generated texts with different steering strengths, saved as JSON for analysis
- **Method**: Based on "Feature-Level Insights into Artificial Text Detection with Sparse Autoencoders" (ACL 2025 Findings)


## 📚 Documentation

- **[autointerp_lite/README.md](autointerp_lite/README.md)** - Feature discovery and ranking system for finding domain-relevant features quickly
- **[autointerp_lite_plus/README.md](autointerp_lite_plus/README.md)** - Enhanced version with comprehensive metrics including clustering, polysemanticity, F1 scores, and quality assessment
- **[autointerp_full/README.md](autointerp_full/README.md)** - Complete LLM-based feature explanation system with confidence scoring and contrastive analysis
- **[autointerp_steer/README.md](autointerp_steer/README.md)** - Feature steering system implementing activation intervention (`x' = x + λ·A_max·d_i`) for analyzing feature functions through controlled generation experiments

## 🚀 Quick Start

### Installation
```bash
# Install everything (from main autointerp/ directory)
pip install -e .

# That's it! Both autointerp_lite and autointerp_full are now available
```

### Files Needed
- **SAE Model**: Path to your trained SAE model files
- **Base Model**: Language model (e.g., `meta-llama/Llama-2-7b-hf`)

### Run Examples
```bash
# Run with API (recommended)
cd autointerp_full
./example_LLM_API.sh

# Run offline
./example_LLM_offline.sh
```

**📋 For detailed parameters and advanced configuration of AutoInterp Full, see:** [autointerp_full/README.md](autointerp_full/README.md)

## 🧠 AutoInterp Full - Advanced Feature Analysis

AutoInterp Full provides comprehensive feature interpretability using LLM-based analysis with contrastive learning and confidence scoring. It generates human-readable explanations with F1 scores, precision, and recall metrics to validate feature quality.

### 🔍 Contrastive Search & FAISS Integration

**Purpose of Contrastive Search:**
AutoInterp Full uses FAISS-based contrastive learning to improve explanation quality by finding semantically similar but non-activating examples. This helps the LLM distinguish between truly relevant features and false positives, leading to more accurate and robust explanations.

**How It Works:**
1. **Embedding Generation**: Uses sentence-transformers to create text embeddings
2. **FAISS Index**: Builds similarity search index of non-activating examples  
3. **Contrastive Prompting**: Shows both activating and non-activating examples to the LLM
4. **Better Explanations**: AI can distinguish between similar-looking content for semantic understanding

### 💬 Chat Model Requirements

AutoInterp Full requires **chat-formatted models only** for proper explanation generation. Supported models include:

| Model Type | Examples | Provider |
|------------|----------|----------|
| **HuggingFace Chat** | `meta-llama/Llama-2-7b-chat-hf`, `Qwen/Qwen2.5-7B-Instruct` | Offline |
| **OpenAI Chat** | `gpt-3.5-turbo`, `gpt-4` | OpenRouter/OpenAI |
| **Other Chat Models** | `google/gemma-2b-it`, `microsoft/DialoGPT-medium` | Various |

**Important**: Base models (like `meta-llama/Llama-2-7b-hf`) are used for SAE analysis, while chat models are used for explanation generation.

### 🎯 Running AutoInterp Full & Viewing Results

**Basic Usage:**
```bash
cd autointerp_full
./example_LLM_API.sh    # API-based (recommended)
./example_LLM_offline.sh # Offline with local models
```

**Key Parameters:**
| Parameter | Purpose | Example |
|-----------|---------|---------|
| `--feature_num` | Specific features to analyze | `27 133 220` |
| `--n_tokens` | Dataset size (affects speed) | `20000` (fast) to `10000000` (thorough) |
| `--explainer_model` | Chat model for explanations | `openai/gpt-3.5-turbo` or `Qwen/Qwen2.5-7B-Instruct` |
| `--non_activating_source` | Contrastive method | `FAISS` (better quality) or `random` (faster) |

**Results Location:**
- **Explanations**: `results/[run_name]/explanations/` - Human-readable feature descriptions
- **Scores**: `results/[run_name]/scores/detection/` - F1, precision, recall metrics  
- **Summary**: `results/[run_name]/results_summary.csv` - Complete results overview

**Quality Metrics:**
- **F1 Score > 0.7**: Good overall accuracy
- **Precision > 0.8**: Reliable when activated
- **Recall > 0.6**: Catches relevant cases well

**Note:** AutoInterp Full includes graceful handling of Chrome dependencies for plotting. If Chrome is not available, plots will be saved as HTML files instead of PDFs.

## 🚀 How to Run

### AutoInterp Lite - Find Relevant Features
```bash
cd autointerp_lite
python run_analysis.py --mode financial
```

### AutoInterp Lite Plus - Comprehensive Analysis
```bash
cd autointerp_lite_plus
python run_analysis.py --base_model "meta-llama/Llama-2-7b-hf" --sae_model "/path/to/sae/model" --domain_data "financial_texts.txt" --general_data "general_texts.txt" --top_n 5 --comprehensive --enable_labeling
```

### AutoInterp Full - Explain Top Features
```bash
cd autointerp_full
./example_LLM_API.sh  # Uses hardcoded features: 27,133,220,17,333
```

### AutoInterp Steer - Feature Steering Analysis
```bash
# Activate sae conda environment first
conda activate sae

# Run steering experiments
cd autointerp_steer
python scripts/run_steering.py --output_folder steering_outputs
```

**Note:** AutoInterp Steer should be run in the `sae` conda environment as it uses SAE-Lens and TransformerLens dependencies.

## 📊 Sample Outputs

### AutoInterp Lite Output
**CSV with ranked features (Real Results):**

| Feature | Label | Specialization | Domain Activation | General Activation | Specialization Conf |
|---------|-------|----------------|-------------------|-------------------|-------------------|
| 133 | Earnings Reports Rate Changes Announcements | **19.56** | 96.73 | 116.29 | 195.60 |
| 162 | value changes performance indicators | **9.58** | 48.20 | 57.78 | 95.76 |
| 203 | Record performance revenue reports | **8.85** | 40.66 | 49.51 | 88.51 |
| 66 | Stock index performance | **4.75** | 19.77 | 24.52 | 47.51 |
| 214 | Inflation indicators labor data | **4.65** | 22.26 | 26.92 | 46.55 |

**Key Metrics:** Domain activation (higher = more active on domain content), specialization score (higher = more domain-specific). Good features: specialization > 3.0, specialization confidence > 30.0.

### AutoInterp Lite Plus Output
**Comprehensive quality assessment table (Real Results):**

| Feature | Label | F1 | F1_Q | Clus | Clus_Q | Poly | Poly_Q | Spec | Spec_Q | Overall |
|---------|-------|----|----|----|----|----|----|----|----|---------|
| 91 | Sector-specific performance investment | 0.917 | **Excellent** | 0 | **Excellent** | 1.000 | **Poor** | 1.203 | **Excellent** | **Excellent** |
| 155 | Cryptocurrency corrections regulatory impacts | 0.935 | **Excellent** | 2 | **Excellent** | 0.286 | **Excellent** | 0.372 | **Good** | **Excellent** |
| 138 | Housing indicators interest rates | 0.904 | **Excellent** | 2 | **Excellent** | 0.143 | **Excellent** | 0.317 | **Good** | **Excellent** |
| 117 | capitalization milestones performance indicators | 0.968 | **Excellent** | 2 | **Excellent** | 0.143 | **Excellent** | 0.229 | **Fair** | **Excellent** |

**Key Metrics:** F1 score (classification accuracy), Clusters (pattern specificity), Polysemanticity (feature coherence), Specialization (domain preference). Quality ranges: Excellent, Good, Fair, Poor.

### AutoInterp Full Output
**Detailed explanations with confidence (Real Results):**

| Feature | Label | F1 Score | Explanation |
|---------|-------|----------|-------------|
| 27 | "-ing" forms | 0.745 | Detects sentences containing "-ing" verb forms and gerunds |
| 220 | Conceptual ideas and alternatives | 0.527 | Identifies abstract concepts and alternative possibilities |
| 133 | Biological taxonomy and species classification | 0.020 | Recognizes biological classification and species terminology |

**Key Metrics:** F1 score (overall accuracy), precision (how often correct when activated), recall (how often it catches relevant cases). Good features: F1 > 0.7, precision > 0.8. Additional metrics available but these are the most important.

## 🎯 When to Use Which

**Use AutoInterp Lite when:** You have thousands of features and need to find the 10-50 that matter for your domain. Perfect for initial exploration and feature screening.

**Use AutoInterp Lite Plus when:** You need comprehensive analysis with advanced metrics including clustering, polysemanticity, F1 scores, and quality assessment. Ideal for research and detailed feature evaluation.

**Use AutoInterp Full when:** You need detailed explanations with confidence scores. Can analyze specific features or all features independently. Essential for research, validation, and detailed analysis.

**Use AutoInterp Steer when:** You want to understand feature functions through controlled intervention experiments. Generates text with varying steering strengths to observe how features affect model behavior. Based on the Kuznetsov et al. (2025) methodology.

**Typical workflow:** Run Lite first to find interesting features, then run Lite Plus for comprehensive analysis, then run Full on the exact feature numbers (e.g., 27,133,220) for detailed explanations, and finally use Steer to analyze how these features affect generation through activation intervention.

---

**Quick Start:**
1. **Find features:** `cd autointerp_lite && python run_analysis.py --mode financial`
2. **Comprehensive analysis:** `cd autointerp_lite_plus && python run_analysis.py --comprehensive --enable_labeling`
3. **Explain features:** `cd autointerp_full && ./example_LLM_API.sh`

## 📁 Repository Structure

This repository contains multiple AutoInterp implementations and related tools. Here's what each folder contains and when to use it:

### 🎯 Main Production Folders

#### `autointerp_full/` - **Production LLM-Based Feature Explanation System**
**Status:** ✅ **Primary production version** - Use this for new projects

**Features:**
- LLM-based feature explanation with confidence scoring
- External prompt configuration via YAML files (`prompts.yaml`, `prompts_finance.yaml`)
- Cache management for efficient re-runs
- FAISS-based contrastive learning for better explanations
- Support for multiple explainer providers (vLLM, OpenRouter, offline)
- CSV generation scripts for results analysis
- Short, concise script names (`run_finbert.sh`, `run_nemotron.sh`, `run_llama_all.sh`, etc.)

**Key Files:**
- `run_finbert.sh` - FinBERT financial news analysis (top 100 features)
- `run_nemotron.sh` - Nemotron financial news analysis (top 100 features)
- `run_nemotron_system.sh` - Nemotron system prompts analysis (top 5 features)
- `run_llama_all.sh` - Llama-3.1-8B all features analysis
- `run_llama_features.sh` - Llama-3.1-8B all features (alternative)
- `run_test_cache.sh` - Cache reuse and prompt override testing
- `prompts.yaml` - Domain-agnostic prompts configuration
- `prompts_finance.yaml` - Finance-specific prompts configuration
- `generate_nemotron_enhanced_csv.py` - Enhanced CSV generation with scorer metrics
- `generate_results_csv.py` - Basic CSV generation

**When to use:** Use this for production feature analysis with LLM-based explanations. Supports both domain-agnostic and finance-specific prompts.

#### `autointerp_lite/` - **Fast Feature Discovery**
**Status:** ✅ **Active** - Use for initial feature screening

**Features:**
- Quick feature ranking by domain specialization
- Compares activations on domain-specific vs general text
- Fast execution (2-5 minutes for 1000+ features)

**When to use:** Use first to find relevant features before running detailed analysis.

#### `autointerp_lite_plus/` - **Comprehensive Feature Analysis**
**Status:** ✅ **Active** - Use for detailed quality assessment

**Features:**
- Advanced metrics: F1 scores, clustering, polysemanticity
- Quality assessment with comprehensive scoring
- Enhanced version of AutoInterp Lite

**When to use:** Use after AutoInterp Lite for comprehensive feature evaluation.

#### `autointerp_steer/` - **Feature Steering Analysis**
**Status:** ✅ **Active** - Use for intervention experiments

**Features:**
- Activation steering experiments (`x' = x + λ·A_max·d_i`)
- Controlled feature intervention
- Text generation with varying steering strengths
- Based on Kuznetsov et al. (2025) methodology

**When to use:** Use to understand how features affect model generation through controlled intervention.

#### `autointerp_saeeval/` - **SAE Evaluation Tools**
**Status:** ⚠️ **Specialized** - Use for SAE-specific evaluation

**Features:**
- SAE model evaluation scripts
- Feature analysis with vLLM integration
- Specialized evaluation workflows

**When to use:** Use for SAE-specific evaluation tasks.

#### `feature_search/` - **Feature Search and Domain Analysis**
**Status:** ⚠️ **Specialized** - Use for domain-specific feature search

**Features:**
- Domain token analysis
- Feature search across domains
- Dashboard and scoring tools

**When to use:** Use for domain-specific feature discovery and analysis.

### 📦 Archive Folders

The `archive/` folder contains historical versions and specialized implementations:

#### `archive/autointerp_full_finance_optimized/` - **Finance-Optimized Version**
**Status:** 📦 **Archived** - Reference for finance-specific optimizations

**Features:**
- Finance-specific prompts (now integrated into `autointerp_full/prompts_finance.yaml`)
- Granular prompts with ENTITY/SECTOR/MACRO/EVENT/STRUCTURAL/LEXICAL classification
- Optimized for financial news analysis
- Legacy scripts and configurations

**When to use:** Reference only - functionality has been integrated into main `autointerp_full/` with external prompt configuration.

#### `archive/autointerp_full_finance/` - **Early Finance Version**
**Status:** 📦 **Archived** - Historical reference

**Features:**
- Early finance-specific implementation
- Legacy scripts and examples

**When to use:** Historical reference only.

#### `archive/autointerp_full_optimized_finbert/` - **FinBERT-Optimized Version**
**Status:** 📦 **Archived** - Reference for FinBERT-specific optimizations

**Features:**
- FinBERT-specific optimizations
- Specialized FinBERT analysis scripts

**When to use:** Reference only - FinBERT support is now in main `autointerp_full/`.

#### `archive/autointerp_full_optimized_toolcall/` - **Tool Call Optimized Version**
**Status:** 📦 **Archived** - Specialized for tool calling features

**Features:**
- Optimized for tool calling feature analysis
- Specialized prompts and configurations

**When to use:** Reference only - for tool calling feature analysis.

#### `archive/autointerp_full_reasoning/` - **Reasoning-Optimized Version**
**Status:** 📦 **Archived** - Specialized for reasoning features

**Features:**
- Optimized for reasoning feature analysis
- Specialized for chain-of-thought patterns

**When to use:** Reference only - for reasoning feature analysis.

#### `archive/autointerp_full_old/` - **Legacy Version**
**Status:** 📦 **Archived** - Historical reference

**Features:**
- Original AutoInterp Full implementation
- Legacy code and examples

**When to use:** Historical reference only.

#### `archive/autointerp_lite_plus/` - **Legacy Lite Plus**
**Status:** 📦 **Archived** - Superseded by main version

**When to use:** Historical reference only.

### 📄 Key Configuration Files

- **`prompts.yaml`** (in `autointerp_full/`) - Domain-agnostic prompts for general use
- **`prompts_finance.yaml`** (in `autointerp_full/`) - Finance-specific prompts with strict length requirements (8-15 words for regular, 5-7 for contrastive)
- **`.gitignore`** - Configured to allow CSV files and archive folder while ignoring logs and large result files

### 🎯 Recommended Workflow

1. **Start with `autointerp_lite/`** to find relevant features quickly
2. **Use `autointerp_lite_plus/`** for comprehensive quality assessment
3. **Run `autointerp_full/`** with specific feature numbers for detailed LLM-based explanations
4. **Use `autointerp_steer/`** for intervention experiments if needed
5. **Reference `archive/`** folders only for historical context or specialized use cases

### 📊 CSV Files

CSV files are tracked in git and provide:
- Feature explanations and labels
- F1 scores, precision, recall metrics
- Quality assessments and rankings
- Summary statistics

All CSV generation scripts are in `autointerp_full/`:
- `generate_nemotron_enhanced_csv.py` - Enhanced CSV with scorer metrics
- `generate_results_csv.py` - Basic CSV generation