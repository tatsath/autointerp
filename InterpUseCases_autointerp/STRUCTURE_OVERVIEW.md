# AutoInterp System Structure Overview

## 🎯 System Architecture

AutoInterp has been restructured into two complementary approaches for SAE interpretability:

### 1. AutoInterp Light - Fast Feature Activation Analysis
**Location**: `autointerp_lite/`
**Purpose**: Quick, lightweight feature analysis focused on activation patterns

**Key Features**:
- Direct activation analysis without LLM-based explanations
- Domain-specific feature identification
- Simple keyword-based labeling
- Fast execution (minutes)
- No API requirements

**Files**:
- `feature_activation_analyzer.py` - Core analysis engine
- `run_analysis.py` - Simple runner script
- `README.md` - Usage documentation

### 2. AutoInterp Full - Detailed Interpretability Analysis
**Location**: `autointerp_full/`
**Purpose**: Comprehensive interpretability with LLM-based explanations and confidence scores

**Key Features**:
- LLM-based feature explanations
- F1 scores and confidence metrics
- FAISS integration for contrastive explanations
- Multi-method comparison and validation
- Publication-quality results

**Files**:
- `sae_autointerp/` - Modified Delphi package with custom features
- `generic_feature_analysis.py` - Direct activation analysis
- `generic_feature_labeling.py` - LLM-based labeling
- `generic_delphi_runner.py` - Delphi interpretability analysis
- `generic_comparison.py` - Method comparison
- `generic_master_script.py` - Complete pipeline orchestrator
- `multi_layer_financial_analysis.py` - Multi-layer analysis system
- `run_financial_analysis.py` - Interactive runner
- `consolidate_labels.py` - Label consolidation utility
- `README.md` - Comprehensive documentation

## 📁 Directory Structure

```
autointerp/
├── README.md                           # Main system overview
├── STRUCTURE_OVERVIEW.md               # This file
├── autointerp_lite/                   # Fast activation analysis
│   ├── feature_activation_analyzer.py  # Core analysis engine
│   ├── run_analysis.py                 # Simple runner
│   └── README.md                       # Light documentation
├── autointerp_full/                    # Detailed interpretability
│   ├── sae_autointerp/                 # Modified Delphi package
│   ├── generic_*.py                    # Core analysis tools
│   ├── multi_layer_*.py                # Multi-layer analysis
│   ├── run_*.py                        # Runner scripts
│   ├── consolidate_labels.py           # Utility scripts
│   └── README.md                       # Full documentation
├── archive/                            # Legacy files
│   ├── example_usage.py
│   ├── GENERIC_SYSTEM_README.md
│   ├── MULTI_LAYER_README.md
│   └── QUICK_START.md
├── complete_financial_analysis/        # Previous analysis results
└── results/                            # General results directory
```

## 🚀 Quick Start Guide

### For Quick Insights (AutoInterp Light)
```bash
cd autointerp_lite
python run_analysis.py --mode financial
```

### For Detailed Analysis (AutoInterp Full)
```bash
cd autointerp_full
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --top_n 10 \
    --domain "financial"
```

## 🎯 When to Use Which Approach

### Use AutoInterp Light When:
- ✅ You want quick insights into feature activations
- ✅ You have domain-specific data and want to find relevant features
- ✅ You need to analyze many features or layers quickly
- ✅ You want to screen features before detailed analysis
- ✅ You don't need LLM-based explanations
- ✅ You want to understand activation patterns

### Use AutoInterp Full When:
- ✅ You need detailed, human-readable explanations
- ✅ You want confidence scores (F1, precision, recall)
- ✅ You're doing research or need publication-quality results
- ✅ You want to compare different interpretability methods
- ✅ You need comprehensive analysis with visualizations
- ✅ You have time for detailed analysis

## 📊 Key Differences

| Feature | AutoInterp Light | AutoInterp Full |
|---------|------------------|-----------------|
| **Speed** | Very Fast (minutes) | Slower (hours) |
| **Explanations** | Simple keyword-based | LLM-generated |
| **Confidence Scores** | Activation metrics | F1, precision, recall |
| **Data Requirements** | Domain + general texts | Domain + general texts |
| **API Requirements** | None | OpenRouter API key |
| **Output Complexity** | Simple CSV | Comprehensive reports |
| **Best For** | Quick screening | Detailed analysis |

## 🔧 Requirements

### Common Requirements
- Python 3.8+
- PyTorch
- Transformers
- Pandas, NumPy
- GPU with 8GB+ VRAM (recommended)

### AutoInterp Light Additional
- Safetensors

### AutoInterp Full Additional
- OpenRouter API key
- Sentence-transformers
- Matplotlib, Seaborn
- Scikit-learn

## 📈 Typical Workflow

1. **Start with AutoInterp Light**: Get quick insights and identify interesting features
2. **Review Results**: Analyze activation patterns and feature rankings
3. **Use AutoInterp Full**: Get detailed explanations for top features
4. **Compare Methods**: Validate findings across different approaches
5. **Iterate**: Refine analysis based on results

## 🎯 Key Benefits

### AutoInterp Light Benefits
- **Fast**: Get results in minutes, not hours
- **Simple**: Easy to use and understand
- **Scalable**: Handle large numbers of features
- **Transparent**: Direct activation analysis
- **No Dependencies**: No API keys or external services

### AutoInterp Full Benefits
- **Comprehensive**: Detailed analysis with multiple methods
- **Confidence**: F1 scores and statistical validation
- **Explanations**: Human-readable feature descriptions
- **Research-Ready**: Publication-quality results
- **Flexible**: Configurable for different domains and models

## 📚 Documentation

- **Main README**: System overview and quick start
- **AutoInterp Light README**: Detailed usage for fast analysis
- **AutoInterp Full README**: Comprehensive documentation for detailed analysis
- **Archive**: Legacy documentation and examples

## 🤝 Contributing

To extend the system:
1. **AutoInterp Light**: Add new labeling heuristics or activation metrics
2. **AutoInterp Full**: Add new explainer models or comparison methods
3. **Both**: Add support for new model architectures or data formats

---

**The restructured AutoInterp system provides both quick insights and detailed analysis, allowing users to choose the right tool for their specific needs.**
