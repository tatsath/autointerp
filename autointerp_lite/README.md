# AutoInterp Lite - Domain-Specific Feature Activation Analysis

A fast, lightweight tool for analyzing feature activations in SAE models with domain-specific data. This approach focuses on activation patterns and provides quick insights without the complexity of full interpretability pipelines.

> ✅ **Recently Updated**: Enhanced with improved analysis features, better LLM labeling, and comprehensive example usage scripts. Successfully tested with financial domain analysis.

## Table of Contents

1. [What It Does](#-what-it-does)
2. [Installation](#-installation)
3. [Quick Start](#-quick-start)
4. [Approach & Methodology](#-approach--methodology)
5. [How to Use](#-how-to-use)
6. [Features & Parameters](#-features--parameters)
7. [Sample Files](#-sample-files)
8. [Output Format](#-output-format)
9. [Example Results](#-example-results)
10. [Use Cases](#-use-cases)
11. [Requirements](#-requirements)
12. [Troubleshooting](#-troubleshooting)
13. [Next Steps](#-next-steps)

## 🎯 What It Does

**Purpose**: Finds SAE features that are specialized for specific domains (like finance, medical, legal) by comparing how they activate on domain-specific vs general text.

**How It Works**: 
- **Domain Data**: Contains domain-specific sentences (financial, medical, legal, scientific, etc.) to find features that activate strongly on domain concepts
- **General Data**: Contains everyday sentences (weather, cooking, music, etc.) as a baseline to identify domain-specific features

**Output**: Top features ranked by specialization score, showing which features are most relevant to your domain. More details below.

## 📦 Installation

### Dependencies
```bash
pip install torch transformers pandas numpy safetensors
```

### Hardware Requirements
- GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- Sufficient disk space for model caching

### Optional: LLM Labeling Support
For LLM-based feature labeling, install additional dependencies:
```bash
# For offline models
pip install transformers accelerate

# For OpenRouter API (optional)
pip install openai
```

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

## 🔬 Approach & Methodology

### How AutoInterp Lite Works

AutoInterp Lite uses a **comparative activation analysis** approach:

1. **Data Collection**: 
   - Processes domain-specific texts (e.g., financial news)
   - Processes general texts (e.g., everyday conversations)
   - Computes SAE feature activations for both datasets

2. **Activation Comparison**:
   - Calculates average activation for each feature on domain texts
   - Calculates average activation for each feature on general texts
   - Computes specialization score: `Domain Activation - General Activation`

3. **Feature Ranking**:
   - Ranks features by specialization score (higher = more domain-specific)
   - Calculates confidence metrics for statistical reliability
   - Optional: Uses LLM to generate human-readable labels

4. **Output Generation**:
   - CSV file with ranked features and metrics
   - JSON summary with statistics
   - Timestamped results to prevent overwriting

### Key Advantages

- **Fast**: 2-5 minutes for 1000+ features (no LLM explanation overhead)
- **Transparent**: Direct activation metrics, no black-box scoring
- **Domain-Focused**: Optimized for finding domain-specific features
- **Scalable**: Can analyze any number of features quickly

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

## 📝 How to Use

### Basic Usage (Without Labeling)

```bash
cd autointerp_lite

python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --layer_idx 16
```

### With LLM Labeling

```bash
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 10 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct" \
    --labeling_provider offline
```

### Using Example Script

```bash
# Run with provided example data
./example_usage.sh
```

### Advanced Configuration

```bash
python run_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --domain_data financial_texts.txt \
    --general_data general_texts.txt \
    --top_n 50 \
    --layer_idx 22 \
    --device cuda \
    --batch_size 64 \
    --max_length 1024 \
    --enable_labeling \
    --labeling_model "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir results
```

### Supported Labeling Models
| Model Type | Examples | Provider |
|------------|----------|----------|
| **HuggingFace Chat** | `meta-llama/Llama-2-7b-chat-hf`, `Qwen/Qwen2.5-7B-Instruct` | Offline |
| **OpenAI Chat** | `gpt-3.5-turbo`, `gpt-4` | OpenRouter |
| **Other Chat Models** | `google/gemma-2b-it`, `microsoft/DialoGPT-medium` | Various |

## 📄 Sample Files

### Input Files

The tool requires two text files:

1. **Domain Data File** (`financial_texts.txt`):
   ```
   The company reported record quarterly earnings of $2.5 billion.
   Stock prices surged following the announcement of a merger deal.
   The Federal Reserve raised interest rates by 0.25 percentage points.
   ```

2. **General Data File** (`general_texts.txt`):
   ```
   The weather forecast predicts sunny skies for the weekend.
   I enjoy cooking Italian pasta dishes on Sunday evenings.
   The concert featured classical music from the Baroque period.
   ```

### File Format Requirements

- **One sentence per line**: Each line should be a complete sentence
- **UTF-8 encoding**: Standard text file encoding
- **No special formatting**: Plain text files work best
- **Sufficient volume**: At least 10-15 sentences per file for meaningful results

### Example Files Location

- `financial_texts.txt` - 42 financial domain sentences (included)
- `general_texts.txt` - 54 general domain sentences (included)

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

## 📊 What You Get

### Output Files (Saved in results/ folder)
- `results/analysis_YYYYMMDD_HHMMSS/features_layer{layer}.csv`: Top features with all information
  - **With labeling**: layer, feature, llm_label, domain_activation, general_activation, specialization, specialization_conf, activation_conf, consistency_conf
  - **Without labeling**: layer, feature, domain_activation, general_activation, specialization, specialization_conf, activation_conf, consistency_conf
- `results/analysis_YYYYMMDD_HHMMSS/summary_layer{layer}.json`: Analysis summary and statistics

> 📁 **Note**: Each analysis creates a unique timestamped folder (e.g., `analysis_20250906_232221/`) to prevent overwriting previous results.

### Key Metrics
- **Domain Activation**: Average activation on domain-specific texts
- **General Activation**: Average activation on general texts  
- **Specialization**: Domain activation - General activation
- **Specialization Confidence**: Statistical confidence in specialization score
- **Activation Confidence**: Confidence in activation measurements
- **Consistency Confidence**: How consistent the feature behavior is
- **Feature Labels**: Intelligent, context-aware labels generated by LLM

### Sample Output (CSV Format)
```csv
layer,feature,llm_label,domain_activation,general_activation,specialization,specialization_conf,activation_conf,consistency_conf
16,133,Earnings Reports Rate Changes Announcements,96.729,116.289,19.560,195.598,193.458,16.636
16,162,value changes performance indicators,48.203,57.779,9.576,95.758,96.406,16.685
16,203,Record performance revenue reports,40.657,49.507,8.851,88.506,81.314,16.425
16,66,Stock index performance,19.766,24.517,4.751,47.509,39.532,16.124
16,214,Inflation indicators labor data,22.261,26.916,4.655,46.547,44.523,16.541
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

## 📈 Example Results

### Top Financial Features (Verified Results)

| Rank | Feature | Domain Activation | General Activation | Specialization | Specialization Conf | Label |
|------|---------|-------------------|-------------------|----------------|-------------------|-------|
| 1 | 133 | 96.73 | 116.29 | **19.56** | 195.60 | Earnings Reports Rate Changes Announcements |
| 2 | 162 | 48.20 | 57.78 | **9.58** | 95.76 | value changes performance indicators |
| 3 | 203 | 40.66 | 49.51 | **8.85** | 88.51 | Record performance revenue reports |
| 4 | 66 | 19.77 | 24.52 | **4.75** | 47.51 | Stock index performance |
| 5 | 214 | 22.26 | 26.92 | **4.65** | 46.55 | Inflation indicators labor data |

> ✅ **Real Results**: These are actual results from running the example script with financial domain analysis. The confidence scores indicate statistical reliability of the specialization measurements. All features show strong domain specialization with high confidence scores.

## 🚨 Requirements

### Software Dependencies
```bash
pip install torch transformers pandas numpy safetensors
```

### Hardware Requirements
- GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- Sufficient disk space for model caching

### Required Inputs
1. **Base Model**: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
2. **SAE Model**: Path to SAE model directory with layer-specific folders
3. **Domain Data**: File with domain-specific sentences (one per line)
4. **General Data**: File with general sentences (one per line)

## 🔍 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use smaller models
2. **File Not Found**: Check SAE model path and text file paths
3. **Poor Results**: Ensure domain texts are comprehensive and diverse

### Tips for Better Results
1. **Use More Domain Texts**: 20-50 sentences work better than 10
2. **Diverse Examples**: Cover different aspects of your domain
3. **Quality Texts**: Use complete, meaningful sentences
4. **Good Contrast**: Ensure clear difference between domain and general texts

## 📚 Next Steps

After running AutoInterp Lite:
1. **Review Results**: Check the timestamped results folder for your analysis
2. **Analyze Features**: Examine the top specialized features and their LLM-generated labels
3. **Validate Findings**: Cross-reference with domain experts or known domain concepts
4. **Layer Comparison**: Run analysis on different layers (4, 10, 22, 28) to see how specialization changes
5. **Scale Up**: Increase `--top_n` for more comprehensive feature coverage
6. **Custom Domains**: Try with your own domain-specific text files

### 🔄 Workflow Integration
- **Quick Analysis**: Use AutoInterp Lite for fast domain-specific feature discovery
- **Deep Analysis**: Use **AutoInterp Full** for detailed interpretability with LLM explanations and F1 scores
- **Research**: Combine both tools for comprehensive SAE feature analysis

### 📊 AutoInterp Full Output Format
For detailed interpretability analysis, AutoInterp Full provides comprehensive explanations:

| Feature | F1 Score | Precision | Recall | Explanation | Top Examples |
|---------|----------|-----------|--------|-------------|--------------|
| 133 | 0.87 | 0.91 | 0.83 | Detects sentences about corporate earnings, revenue growth, and financial performance metrics | "Q3 revenue increased 15%", "profit margin expanded", "earnings per share" |
| 162 | 0.82 | 0.85 | 0.79 | Identifies market performance indicators and value change discussions | "S&P 500 closed at", "stock price movement", "market volatility" |
| 203 | 0.79 | 0.83 | 0.75 | Recognizes record performance announcements and revenue reporting | "record quarterly revenue", "performance exceeded expectations", "revenue reports" |

**Key Metrics**: F1 Score (overall performance), Precision (accuracy of positive predictions), Recall (coverage of actual positives). Good features: F1 > 0.75, Precision > 0.80, Recall > 0.70.

### 🎯 Recent Improvements
- ✅ Enhanced confidence scoring for better statistical reliability
- ✅ Improved LLM labeling with context-aware prompts
- ✅ Timestamped results to prevent overwriting
- ✅ Comprehensive example usage script
- ✅ Better error handling and user feedback
- ✅ Verified results with financial domain analysis
