# AutoInterp - SAE Interpretability System

Two complementary approaches for understanding what features in your SAE model have learned.

## 🎯 What This Does

Three complementary approaches for understanding what features in your SAE model have learned.

**Step 1:** Use AutoInterp Lite to quickly find relevant features and get basic labels. **Step 2:** Use AutoInterp Lite Plus for comprehensive analysis with advanced metrics. **Step 3:** Use AutoInterp Full with the exact feature numbers from Step 1 to get detailed explanations with confidence scores.

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


## 📚 Documentation

- **[autointerp_lite/README.md](autointerp_lite/README.md)** - Feature discovery and ranking system for finding domain-relevant features quickly
- **[autointerp_lite_plus/README.md](autointerp_lite_plus/README.md)** - Enhanced version with comprehensive metrics including clustering, polysemanticity, F1 scores, and quality assessment
- **[autointerp_full/README.md](autointerp_full/README.md)** - Complete LLM-based feature explanation system with confidence scoring and contrastive analysis

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

**Typical workflow:** Run Lite first to find interesting features, then run Lite Plus for comprehensive analysis, then run Full on the exact feature numbers (e.g., 27,133,220) for detailed explanations.

---

**Quick Start:**
1. **Find features:** `cd autointerp_lite && python run_analysis.py --mode financial`
2. **Comprehensive analysis:** `cd autointerp_lite_plus && python run_analysis.py --comprehensive --enable_labeling`
3. **Explain features:** `cd autointerp_full && ./example_LLM_API.sh`