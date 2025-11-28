# Generic Feature Analysis System

A comprehensive, modular system for analyzing features in any SAE model with any base model. This system provides both direct activation analysis and Delphi-based interpretability, with automatic comparison between the two approaches.

## 🚀 Quick Start

### Complete Pipeline (Recommended)
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --top_n 10 \
    --domain "financial" \
    --output_dir "results"
```

### Individual Components
```bash
# 1. Feature Analysis
python generic_feature_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --top_n 10

# 2. Feature Labeling
python generic_feature_labeling.py \
    --analysis_file "results/top_10_features_analysis.json" \
    --domain "financial"

# 3. Delphi Analysis
python generic_delphi_runner.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --features "163,59,333,208,182" \
    --run_name "my_analysis"

# 4. Comparison
python generic_comparison.py \
    --labeling_file "feature_labels_detailed_financial.csv" \
    --delphi_file "my_analysis_delphi_results.csv"
```

## 📁 System Components

### 1. `generic_feature_analysis.py`
**Purpose**: Find top activating features using direct activation analysis

**Key Features**:
- Works with any base model and SAE
- Configurable number of top features
- Configurable layer for feature extraction
- Automatic sentence extraction for top features
- Saves results in JSON and CSV formats

**Usage**:
```bash
python generic_feature_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --top_n 20 \
    --layer_idx 16 \
    --output_dir "results"
```

**Outputs**:
- `top_N_features_analysis.json`: Complete analysis data
- `top_N_features_analysis.csv`: Feature rankings and metrics

### 2. `generic_feature_labeling.py`
**Purpose**: Generate human-readable labels for features using LLM

**Key Features**:
- Domain-specific labeling (financial, medical, legal, etc.)
- Configurable labeling model
- Granular, specific label generation
- Automatic prompt engineering

**Usage**:
```bash
python generic_feature_labeling.py \
    --analysis_file "results/top_10_features_analysis.json" \
    --labeling_model "meta-llama/Llama-2-7b-chat-hf" \
    --domain "financial" \
    --top_n 10
```

**Outputs**:
- `feature_labels_clean_DOMAIN.csv`: Feature numbers and labels
- `feature_labels_detailed_DOMAIN.csv`: Complete metrics and labels

### 3. `generic_delphi_runner.py`
**Purpose**: Run Delphi interpretability analysis on specific features

**Key Features**:
- Uses Delphi's built-in F1 calculation
- Configurable explainer models
- FAISS integration for contrastive explanations
- Automatic result processing

**Usage**:
```bash
python generic_delphi_runner.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --features "163,59,333,208,182" \
    --explainer_model "openai/gpt-3.5-turbo" \
    --run_name "my_delphi_run"
```

**Outputs**:
- `RUN_NAME_delphi_results.csv`: Delphi explanations and F1 scores
- `runs/RUN_NAME/`: Complete Delphi analysis results

### 4. `generic_comparison.py`
**Purpose**: Compare labels from direct analysis vs Delphi

**Key Features**:
- Semantic similarity calculation
- Statistical analysis and correlations
- Visualization generation
- Comprehensive reporting

**Usage**:
```bash
python generic_comparison.py \
    --labeling_file "feature_labels_detailed_financial.csv" \
    --delphi_file "my_delphi_run_delphi_results.csv" \
    --output_dir "results"
```

**Outputs**:
- `comparison_results.csv`: Detailed comparison data
- `comparison_report.json`: Statistical summary
- `comparison_analysis.png`: Visualization plots

### 5. `generic_master_script.py`
**Purpose**: Orchestrate the complete pipeline

**Key Features**:
- Runs all components in sequence
- Automatic file management
- Error handling and recovery
- Configurable pipeline steps

**Usage**:
```bash
# Complete pipeline
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --top_n 10 \
    --domain "financial"

# Specific steps only
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --steps analysis labeling \
    --top_n 10
```

## 🎯 Use Cases

### Financial Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/financial/sae" \
    --domain "financial" \
    --top_n 15
```

### Medical Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/medical/sae" \
    --domain "medical" \
    --top_n 20
```

### Legal Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/legal/sae" \
    --domain "legal" \
    --top_n 12
```

## 📊 Output Structure

```
results/
├── top_10_features_analysis.json          # Complete analysis data
├── top_10_features_analysis.csv           # Feature rankings
├── feature_labels_clean_financial.csv     # Clean labels
├── feature_labels_detailed_financial.csv  # Detailed labels
├── my_run_delphi_results.csv              # Delphi results
├── comparison_results.csv                 # Comparison data
├── comparison_report.json                 # Statistical report
└── comparison_analysis.png                # Visualization
```

## 🔧 Configuration Options

### Base Models
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `meta-llama/Llama-2-70b-hf`
- Any HuggingFace model

### SAE Models
- Any SAE model with `encoder.pt` and `decoder.pt` files
- Configurable layer indices
- Configurable feature counts

### Domains
- `financial`: Corporate finance, markets, banking
- `medical`: Healthcare, diagnosis, treatment
- `legal`: Law, contracts, regulations
- `scientific`: Research, methodology, data
- `technical`: Engineering, software, systems

### Explainer Models
- `openai/gpt-3.5-turbo`
- `openai/gpt-4o`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`

## 📈 Metrics and Analysis

### Feature Analysis Metrics
- **Financial Activation**: Average activation on financial text
- **General Activation**: Average activation on general text
- **Specialization**: Financial activation - General activation
- **Ranking**: Features sorted by specialization

### Delphi Metrics
- **F1 Score**: Detection performance using Delphi's built-in calculation
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Explanation**: Delphi-generated feature explanation

### Comparison Metrics
- **Semantic Similarity**: Cosine similarity between labels and explanations
- **Correlation Analysis**: Relationships between different metrics
- **Statistical Summary**: Mean, median, distribution analysis

## 🚨 Requirements

### Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn sentence-transformers matplotlib seaborn plotly
```

### Environment Variables
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
export PYTHONPATH="/path/to/sae_autointerp"
```

### Hardware Requirements
- GPU with at least 8GB VRAM (for base model)
- 16GB+ RAM
- Sufficient disk space for model caching

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller models
   - Process fewer features at once

2. **Model Loading Errors**
   - Check model paths
   - Verify HuggingFace authentication
   - Ensure sufficient disk space

3. **Delphi API Errors**
   - Verify OpenRouter API key
   - Check API rate limits
   - Try different explainer models

4. **File Not Found Errors**
   - Check output directory permissions
   - Verify file paths
   - Ensure previous steps completed successfully

### Debug Mode
Add `--verbose` flag to any script for detailed logging:
```bash
python generic_master_script.py --verbose --base_model "..." --sae_model "..."
```

## 📚 Examples

See `example_usage.py` for comprehensive examples:
```bash
python example_usage.py
```

## 🤝 Contributing

To extend the system:
1. Add new domains in `generic_feature_labeling.py`
2. Add new metrics in `generic_comparison.py`
3. Add new visualizations in `generic_comparison.py`
4. Add new analysis methods in `generic_feature_analysis.py`

## 📄 License

This system is part of the SAE interpretability research project.
