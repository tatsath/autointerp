# AutoInterp Lite Plus - SAE Feature Analysis with Advanced Metrics

Enhanced tool for analyzing SAE feature activations with domain-specific data. Includes comprehensive metrics: clustering, polysemanticity, F1 scores, and robustness testing.

> ✅ **Enhanced Version**: Comprehensive metrics, clustering analysis, and multi-format output. Tested with financial domain analysis.

## 🎯 What It Does

**Purpose**: Finds SAE features specialized for specific domains by comparing activations on domain-specific vs general text.

**How It Works**: 
- **Domain Data**: Domain-specific sentences (finance, medical, legal, etc.)
- **General Data**: Everyday sentences as baseline
- **Analysis**: Comprehensive metrics including clustering, F1 scores, and robustness

**Output**: Top features with specialization scores and comprehensive metrics.

## 📊 Comprehensive Metrics

- **Clustering**: Number of clusters, silhouette score, polysemanticity
- **Classification**: F1 score, precision, recall, selectivity
- **Robustness**: Performance under text perturbations

## 🚀 Quick Start

### ✅ Tested and Verified
The tool has been successfully tested with:
- **Base Model**: meta-llama/Llama-2-7b-hf
- **SAE Model**: Local trained model (layer 16)
- **Domain Data**: 42 financial texts
- **General Data**: 54 general texts
- **Labeling Model**: Qwen/Qwen2.5-7B-Instruct
- **Results**: Generated meaningful financial domain features with high confidence scores

### Flexible Command-Line Interface (Recommended)
```bash
cd autointerp_lite

# Basic analysis without labeling
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/local/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10

# Comprehensive analysis with all metrics
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data "jyanimaulik/yahoo_finance_stockmarket_news" \
    --general_data "wikitext" \
    --top_n 5 \
    --comprehensive \
    --max_samples 1000

# With LLM labeling
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct"
```

## 🔄 Process

1. **Data Loading**: Load domain and general texts
2. **Feature Discovery**: Extract activations, calculate specialization scores
3. **Comprehensive Analysis**: Clustering, classification metrics, robustness testing, LLM labeling
4. **Output Generation**: Save results with comprehensive metrics

### Quick Examples
```bash
# Run complete analysis with example data (recommended for first-time users)
./example_usage.sh

# This script runs:
# - Top 10 most specialized features analysis
# - LLM labeling with Qwen/Qwen2.5-7B-Instruct
# - Financial vs General text comparison
# - Generates timestamped results in results/ directory
```

### ✅ Verified Example Results
The example script has been successfully tested and produces meaningful results:
```
🏆 Top 5 Features:
1. Feature 133 | Spec: 19.56 | "Earnings Reports Rate Changes Announcements"
2. Feature 162 | Spec: 9.58  | "value changes performance indicators"  
3. Feature 203 | Spec: 8.85  | "Record performance revenue reports"
4. Feature 66  | Spec: 4.75  | "Stock index performance"
5. Feature 214 | Spec: 4.65  | "Inflation indicators labor data"
```

## 🎯 Features

### Flexible Command-Line Interface
- **Configurable Top X Features**: Choose exactly how many features to analyze (e.g., `--top_n 10`)
- **HuggingFace & Local Models**: Support for both HuggingFace models and local model paths
- **Optional LLM Labeling**: Enable/disable intelligent labeling with `--enable_labeling`
- **Multiple Labeling Models**: Support for offline models (Llama, Qwen) and API providers (OpenRouter)
- **Custom Prompts**: Use your own prompt files with `--labeling_prompt custom_prompt.txt`
- **Flexible Configuration**: Device selection, batch size, sequence length control

### Key Parameters
| Parameter | Required | Purpose | Example | Default |
|-----------|----------|---------|---------|---------|
| `--base_model` | ✅ **Required** | Base model (HuggingFace ID or local path) | `"meta-llama/Llama-2-7b-hf"` or `"/path/to/model"` | - |
| `--sae_model` | ✅ **Required** | SAE model (HuggingFace ID or local path) | `"EleutherAI/sae-llama-3-8b-32x"` or `"/path/to/sae"` | - |
| `--domain_data` | ✅ **Required** | Domain-specific data file | `"financial_texts.txt"` | - |
| `--general_data` | ✅ **Required** | General data file | `"general_texts.txt"` | - |
| `--top_n` | ⚪ Optional | Number of top features to analyze | `10`, `50`, `100` | `10` |
| `--layer_idx` | ⚪ Optional | Layer index to analyze | `16`, `22`, `28` | `16` |
| `--device` | ⚪ Optional | Device to use | `auto`, `cuda`, `cpu` | `auto` |
| `--batch_size` | ⚪ Optional | Batch size for processing | `32`, `16`, `64` | `32` |
| `--max_length` | ⚪ Optional | Maximum sequence length | `512`, `1024`, `256` | `512` |
| `--output_dir` | ⚪ Optional | Output directory | `"."`, `"results"` | `"."` |
| `--enable_labeling` | ⚪ Optional | Enable LLM-based labeling | `--enable_labeling` | Disabled |
| `--labeling_model` | ⚪ Optional | Model for labeling (chat models only) | `"Qwen/Qwen2.5-7B-Instruct"` | `"meta-llama/Llama-2-7b-chat-hf"` |
| `--labeling_provider` | ⚪ Optional | Labeling provider | `offline` or `openrouter` | `offline` |
| `--labeling_prompt` | ⚪ Optional | Custom prompt file path | `"core/sample_labeling_prompt.txt"` | Default prompt |

### Supported Labeling Models
| Model Type | Examples | Provider |
|------------|----------|----------|
| **HuggingFace Chat** | `meta-llama/Llama-2-7b-chat-hf`, `Qwen/Qwen2.5-7B-Instruct` | Offline |
| **OpenAI Chat** | `gpt-3.5-turbo`, `gpt-4` | OpenRouter |
| **Other Chat Models** | `google/gemma-2b-it`, `microsoft/DialoGPT-medium` | Various |

## 📋 What You Need

### Required Inputs
1. **Base Model**: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
2. **SAE Model**: Path to SAE model directory with layer-specific folders
3. **Domain Data**: File with domain-specific sentences (one per line)
4. **General Data**: File with general sentences (one per line)

### Domain-Specific Data Requirements
- **Elaborate Dataset**: The more comprehensive your domain texts, the better the analysis
- **Diverse Examples**: Include various aspects of your domain (e.g., for finance: banking, stocks, crypto, real estate)
- **Quality Texts**: Well-formed, meaningful sentences work better than fragments
- **Sufficient Volume**: At least 10-15 sentences per category for meaningful results

## 📝 Custom Prompts

### Creating Custom Prompt Files
You can create custom prompt files to control how the LLM generates labels:

```bash
# Create a custom prompt file
cat > my_prompt.txt << 'EOF'
Analyze this SAE feature and provide a 2-3 word label:

Domain examples (high activation):
{domain_examples}

General examples (low activation):
{general_examples}

Specialization score: {specialization:.2f}

What concept does this feature detect? Label:
EOF

# Use the custom prompt
python run_analysis.py \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --prompt_file my_prompt.txt
```

### Prompt Variables
- `{domain_examples}`: Domain-specific texts where feature activates
- `{general_examples}`: General texts where feature activates weakly  
- `{specialization}`: Numerical specialization score

### Sample Labeling Prompt File
See `core/sample_labeling_prompt.txt` for a well-structured example prompt.

## 📋 Arguments

**Required**: `--base_model`, `--sae_model`, `--domain_data`, `--general_data`

**Optional**: `--top_n` (10), `--layer_idx` (16), `--comprehensive`, `--enable_labeling`, `--labeling_model`, `--max_samples` (1000)

## 📊 What You Get

### Output Files

- **Basic**: `results/analysis_YYYYMMDD_HHMMSS/features_layer{layer}.csv`
- **Comprehensive**: `results/comprehensive_analysis_YYYYMMDD_HHMMSS/comprehensive_results.json`
- `results/analysis_YYYYMMDD_HHMMSS/summary_layer{layer}.json`: Analysis summary and statistics

> 📁 **Note**: Each analysis creates a unique timestamped folder (e.g., `analysis_20250906_232221/`) to prevent overwriting previous results.

### Key Metrics

#### 🎯 **Core Specialization Metrics**
- **Specialization Score**: Domain activation - General activation (higher = more domain-specific)
- **Domain Activation**: Average activation on domain-specific texts
- **General Activation**: Average activation on general texts

#### 📊 **Classification Metrics**
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **Precision**: True positives / (True positives + False positives) - accuracy of positive predictions
- **Recall**: True positives / (True positives + False negatives) - coverage of actual positives
- **Selectivity**: True negatives / (True negatives + False positives) - how well it distinguishes negatives

#### 🔍 **Clustering Metrics**
- **Number of Clusters**: How many distinct groups the positive examples form (lower = more specific)
- **Silhouette Score**: Quality of clustering (-1 to 1, higher = better separated clusters)
- **Polysemanticity**: Noise ratio indicating feature specificity (lower = more specific, 0-1)

#### 🧪 **Robustness Metrics**
- **Positive Drop**: Performance drop when positive examples are perturbed (lower = more robust)
- **Negative Rise**: Performance rise when negative examples are perturbed (lower = more robust)

#### 📈 **How to Interpret Results**

**🎯 Specialization Score Ranges:**
- **Excellent (>1.0)**: Highly domain-specific, very clear separation
- **Good (0.5-1.0)**: Strong domain preference, clear specialization
- **Moderate (0.2-0.5)**: Some domain preference, useful but not dominant
- **Weak (0.1-0.2)**: Minimal domain preference, may be noise
- **Poor (<0.1)**: No clear domain preference, likely not useful

**📊 F1 Score Ranges:**
- **Excellent (0.9-1.0)**: Outstanding classification performance
- **Very Good (0.8-0.9)**: Strong classification performance
- **Good (0.7-0.8)**: Decent classification performance
- **Fair (0.6-0.7)**: Acceptable but could be better
- **Poor (<0.6)**: Weak classification performance

**🔍 Clustering Ranges:**
- **0-1 clusters**: Highly specific, all examples are similar
- **2-3 clusters**: Some variation but still coherent
- **4+ clusters**: Multiple distinct patterns, may be polysemantic

**🧪 Polysemanticity Ranges:**
- **0.0-0.3**: Highly specific, single meaning
- **0.3-0.6**: Some variation but mostly coherent
- **0.6-1.0**: High noise, multiple meanings

**✅ Feature Quality Assessment:**
- **Excellent**: Specialization >0.5, F1 >0.8, Polysemanticity <0.3, Clusters ≤2
- **Good**: Specialization >0.3, F1 >0.7, Polysemanticity <0.5, Clusters ≤3
- **Fair**: Specialization >0.2, F1 >0.6, Polysemanticity <0.7, Clusters ≤4
- **Poor**: Specialization <0.2, F1 <0.6, Polysemanticity >0.7, Clusters >4

### Sample Output (Console Display)
```
================================================================================
AUTOINTERP LITE PLUS - COMPREHENSIVE ANALYSIS
================================================================================
📊 Domain texts: 42
📊 General texts: 54
🎯 Top features: 5
🏷️  Labeling: Enabled
🤖 Labeling model: Qwen/Qwen2.5-7B-Instruct

🔍 Finding top features...
Top 5 features identified
Best feature: 91.0 (specialization: 1.203)
🎯 Top 5 features: [91, 155, 138, 398, 117]

🎯 COMPREHENSIVE FEATURE ANALYSIS - SPECIFIC FEATURES
Analyzing features: [91, 155, 138, 398, 117]
----------------------------------------

--- PROCESSING FEATURE 91 (1/5) ---
🔍 COMPREHENSIVE ANALYSIS FOR FEATURE 91
============================================================
   Positive examples: 7
   Negative examples: 7
🔗 Calculating clustering metrics...
📊 Calculating classification metrics...
🧪 Calculating robustness metrics...
✅ Feature 91 analysis complete!
   Label: Banking insurance sector performance ratios
   F1: 0.917, Selectivity: 0.917
   Clusters: 0, Polysemanticity: 1.000

--- PROCESSING FEATURE 155 (2/5) ---
🔍 COMPREHENSIVE ANALYSIS FOR FEATURE 155
============================================================
   Positive examples: 7
   Negative examples: 7
🔗 Calculating clustering metrics...
📊 Calculating classification metrics...
🧪 Calculating robustness metrics...
✅ Feature 155 analysis complete!
   Label: Cryptocurrency corrections regulatory concerns
   F1: 0.935, Selectivity: 0.935
   Clusters: 2, Polysemanticity: 0.286

📊 COMPREHENSIVE ANALYSIS SUMMARY
================================================================================
Feature ID Label                                              F1     Prec   Rec    Sel    Clus   Silh   Poly   Spec
--------------------------------------------------------------------------------
91         Banking insurance sector performance ratios        0.917  0.917  0.917  0.917  0      0.000  1.000  1.203
155        Cryptocurrency corrections regulatory concerns     0.935  0.935  0.935  0.935  2      0.068  0.286  0.372
138        Housing indicators interest rates                  0.904  0.904  0.904  0.904  2      0.054  0.143  0.317
398        Cryptocurrency corrections regulatory issues       0.889  0.889  0.889  0.889  2      0.033  0.429  0.255
117        capitalization milestones strong performance       0.875  0.875  0.875  0.875  2      0.037  0.143  0.229
--------------------------------------------------------------------------------
📈 METRIC EXPLANATIONS:
  F1/Prec/Rec/Sel: 0-1 (higher is better)
  Clus: Number of clusters (lower is better)
  Silh: Silhouette score (higher is better)
  Poly: Polysemanticity index (lower is better)
  Spec: Specialization score (higher is better)

✅ Comprehensive analysis complete!
📁 Results saved to: results/comprehensive_analysis_20250921_012519/comprehensive_results.json
📊 Processed 5 features successfully
```

#### 📊 **Understanding the Sample Output**

**Feature 91 - "Banking insurance sector performance ratios"**
- **F1: 0.917** - Excellent classification performance (91.7% accuracy)
- **Clusters: 0** - All positive examples are similar (highly specific feature)
- **Polysemanticity: 1.000** - Some noise, but still good specificity
- **Specialization: 1.203** - Very high domain-specific activation

**Feature 155 - "Cryptocurrency corrections regulatory concerns"**
- **F1: 0.935** - Outstanding classification performance (93.5% accuracy)
- **Clusters: 2** - Two distinct patterns in positive examples
- **Polysemanticity: 0.286** - Low noise, highly specific feature
- **Specialization: 0.372** - Good domain-specific activation

**Feature 138 - "Housing indicators interest rates"**
- **F1: 0.904** - Very good classification performance (90.4% accuracy)
- **Clusters: 2** - Two distinct patterns in positive examples
- **Polysemanticity: 0.143** - Very low noise, highly specific feature
- **Specialization: 0.317** - Moderate domain-specific activation

### Sample Output (JSON Format)
```json
[
  {
    "feature_id": 91,
    "label": "Banking insurance sector performance ratios",
    "specialization_score": 1.2025335034503621,
    "domain_activation": 0.4632161557674408,
    "general_activation": 0.02544388361275196,
    "f1": 0.917,
    "precision": 0.917,
    "recall": 0.917,
    "selectivity": 0.917,
    "n_clusters": 0,
    "silhouette_score": 0.0,
    "polysemanticity": 1.0,
    "pos_drop": 0.25,
    "neg_rise": 1.0
  },
  {
    "feature_id": 155,
    "label": "Cryptocurrency corrections regulatory concerns",
    "specialization_score": 0.3719511197233313,
    "domain_activation": 0.18987543880939484,
    "general_activation": 0.06645781546831131,
    "f1": 0.935,
    "precision": 0.935,
    "recall": 0.935,
    "selectivity": 0.935,
    "n_clusters": 2,
    "silhouette_score": 0.06838319450616837,
    "polysemanticity": 0.2857142857142857,
    "pos_drop": 0.25,
    "neg_rise": 1.0
  }
]
```

### Sample Output (Console Display)
```
🏆 Top 5 Features:
--------------------------------------------------------------------------------
 1. Feature 133 | Spec:  19.56 | SpecConf: 195.6 | ActConf: 193.5 | ConsConf: 16.6 | Earnings Reports Rate Changes Announcements
 2. Feature 162 | Spec:   9.58 | SpecConf:  95.8 | ActConf:  96.4 | ConsConf: 16.7 | value changes performance indicators
 3. Feature 203 | Spec:   8.85 | SpecConf:  88.5 | ActConf:  81.3 | ConsConf: 16.4 | Record performance revenue reports
 4. Feature  66 | Spec:   4.75 | SpecConf:  47.5 | ActConf:  39.5 | ConsConf: 16.1 | Stock index performance
 5. Feature 214 | Spec:   4.65 | SpecConf:  46.5 | ActConf:  44.5 | ConsConf: 16.5 | Inflation indicators labor data
```

### 📊 Detailed Results Table

| Rank | Feature | Domain Activation | General Activation | Specialization | Specialization Conf | Activation Conf | Consistency Conf | LLM Label |
|------|---------|-------------------|-------------------|----------------|-------------------|-----------------|------------------|-----------|
| 1 | 133 | 96.73 | 116.29 | **19.56** | 195.60 | 193.46 | 16.64 | Earnings Reports Rate Changes Announcements |
| 2 | 162 | 48.20 | 57.78 | **9.58** | 95.76 | 96.41 | 16.69 | value changes performance indicators |
| 3 | 203 | 40.66 | 49.51 | **8.85** | 88.51 | 81.31 | 16.42 | Record performance revenue reports |
| 4 | 66 | 19.77 | 24.52 | **4.75** | 47.51 | 39.53 | 16.12 | Stock index performance |
| 5 | 214 | 22.26 | 26.92 | **4.65** | 46.55 | 44.52 | 16.54 | Inflation indicators labor data |
| 6 | 340 | 22.77 | 27.11 | **4.33** | 43.33 | 45.54 | 16.80 | Asset class diversification yieldspread dynamics |
| 7 | 105 | 18.53 | 22.62 | **4.09** | 40.93 | 37.05 | 16.38 | Private Equity Venture Capital Funding |
| 8 | 267 | 16.67 | 19.94 | **3.27** | 32.72 | 33.35 | 16.72 | Foreign exchange volatility due to central bank policies Commodity price |
| 9 | 332 | 15.36 | 18.61 | **3.26** | 32.57 | 30.72 | 16.50 | Sophisticated trading performance metrics |
| 10 | 181 | 13.78 | 16.97 | **3.19** | 31.95 | 27.55 | 16.23 | Innovations in sectors |

### 📈 Key Metrics Interpretation

| Metric | Description | Good Values | Interpretation |
|--------|-------------|-------------|----------------|
| **Domain Activation** | Average activation on financial texts | > 15.0 | Higher = more active on domain content |
| **General Activation** | Average activation on general texts | < 30.0 | Lower = more domain-specific |
| **Specialization** | Domain - General activation | > 3.0 | Higher = more domain-specific |
| **Specialization Conf** | Statistical confidence in specialization | > 30.0 | Higher = more reliable measurement |
| **Activation Conf** | Confidence in activation measurements | > 25.0 | Higher = more reliable activations |
| **Consistency Conf** | Feature behavior consistency | > 15.0 | Higher = more consistent behavior |

## 🎯 Use Cases

### Financial Analysis
- Identify features that activate on financial concepts
- Compare activation patterns across different financial domains
- Find features specialized for specific financial tasks

### Medical Analysis
- Discover features that respond to medical terminology
- Analyze activation patterns in healthcare texts
- Identify domain-specific medical concepts

### Legal Analysis
- Find features that activate on legal language
- Analyze patterns in legal documents
- Identify specialized legal concepts

### Scientific Analysis
- Discover features that respond to scientific terminology
- Analyze activation patterns in research papers
- Identify domain-specific scientific concepts

### Educational Analysis
- Find features that activate on educational content
- Analyze patterns in academic texts
- Identify specialized educational concepts

## ⚡ Advantages

- **Fast**: No LLM-based explanations or complex scoring
- **Lightweight**: Minimal dependencies and computational requirements
- **Transparent**: Direct activation analysis with clear metrics
- **Scalable**: Can analyze any number of features quickly
- **Domain-Focused**: Optimized for domain-specific analysis

## 🔧 Configuration

### Layer Selection
- Choose the layer that best represents your domain
- Later layers (16, 22, 28) often have more specialized features
- Earlier layers (4, 10) may have more general patterns

### Feature Count
- Start with top 30-50 features for initial analysis
- Increase for more comprehensive coverage
- Balance between coverage and computational cost

### Text Quality
- Use complete, well-formed sentences
- Include diverse examples from your domain
- Ensure good contrast between domain and general texts

## 🚨 Requirements

```bash
pip install torch transformers pandas numpy safetensors sentence-transformers scikit-learn
```

## 🔍 Troubleshooting

- **CUDA Out of Memory**: Reduce `--max_samples` or use CPU
- **File Not Found**: Check SAE model path and text file paths
- **Poor Results**: Ensure domain texts are comprehensive and diverse
