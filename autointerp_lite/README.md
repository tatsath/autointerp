# AutoInterp Light - Domain-Specific Feature Activation Analysis

A fast, lightweight tool for analyzing feature activations in SAE models with domain-specific data. This approach focuses on activation patterns and provides quick insights without the complexity of full interpretability pipelines.

## 🎯 What It Does

**Purpose**: Finds SAE features that are specialized for specific domains (like finance, medical, legal) by comparing how they activate on domain-specific vs general text.

**How It Works**: 
- **Financial Data**: Contains financial sentences (earnings, stocks, banking, etc.) to find features that activate strongly on financial concepts
- **General Data**: Contains everyday sentences (weather, cooking, music, etc.) as a baseline to identify domain-specific features

**Output**: Top features ranked by specialization score, showing which features are most relevant to your domain. More details below.

## 🚀 Quick Start

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
# Run all examples
./example_usage.sh
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

## 📊 What You Get

### Output Files (Saved in results/ folder)
- `results/features_layer{layer}.csv`: Top features with all information
  - **With labeling**: layer, feature_number, domain_activation, general_activation, specialization, llm_label
  - **Without labeling**: layer, feature_number, domain_activation, general_activation, specialization
- `results/summary_layer{layer}.json`: Analysis summary and statistics

### Key Metrics
- **Domain Activation**: Average activation on domain-specific texts
- **General Activation**: Average activation on general texts  
- **Specialization**: Domain activation - General activation
- **Feature Labels**: Simple, keyword-based labels for each feature

### Sample Output
```
feature_number,domain_activation,general_activation,specialization,layer,label
27,-85.505,-116.051,30.545,16,Stock market trading
133,-77.501,-106.184,28.683,16,Banking and loans
220,-59.814,-83.790,23.975,16,Cryptocurrency
```

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

### Top Financial Features (Sample)
```
Rank | Feature | Specialization | Label
-----|---------|----------------|------------------
  1  |   27    |     30.545     | Stock market trading
  2  |   133   |     28.683     | Banking and loans  
  3  |   220   |     23.975     | Cryptocurrency
  4  |   17    |     21.603     | Real estate
  5  |   333   |     17.386     | Investment management
```

## 🚨 Requirements

### Dependencies
```bash
pip install torch transformers pandas numpy safetensors
```

### Hardware
- GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- Sufficient disk space for model caching

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

After running AutoInterp Light:
1. Review the top features and their labels
2. Analyze activation patterns across different layers
3. Use results to guide further analysis with AutoInterp Full
4. Validate findings with domain experts

For more detailed interpretability analysis, use **AutoInterp Full** which provides LLM-based explanations, F1 scores, and comprehensive interpretability metrics.
