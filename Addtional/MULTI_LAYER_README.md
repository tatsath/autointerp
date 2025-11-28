# Multi-Layer Financial Feature Analysis

A comprehensive system for analyzing financial features across multiple layers of Llama model SAEs. This system identifies the top 30 financial features per layer and provides detailed analysis and comparison.

## 🚀 Quick Start

### Simple Analysis
```bash
cd /home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp

python run_financial_analysis.py
```

### Direct Analysis
```bash
python multi_layer_financial_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --layers 4 10 16 22 28 \
    --top_n 30
```

## 📁 System Components

### 1. `multi_layer_financial_analysis.py`
**Purpose**: Complete multi-layer financial feature analysis system

**Key Features**:
- Analyzes multiple layers in sequence
- Configurable layer selection
- Automatic result consolidation
- Layer comparison analysis
- Top financial feature identification
- Statistical analysis and visualizations
- Comprehensive reporting

**Usage**:
```bash
python multi_layer_financial_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --layers 4 10 16 22 28 \
    --top_n 30 \
    --output_dir "multi_layer_results" \
    --min_specialization 0.1 \
    --top_features_n 50
```

### 2. `run_financial_analysis.py`
**Purpose**: Simple interactive script for easy execution

**Key Features**:
- User-friendly interface
- Multiple analysis options
- Pre-configured settings

**Usage**:
```bash
python run_financial_analysis.py
```

## 🎯 Analysis Modes

### Complete Analysis
- **Layers**: 4, 10, 16, 22, 28
- **Features per layer**: 30
- **Total features**: 150
- **Time**: ~2-3 hours
- **Output**: Complete multi-layer analysis

### Fast Analysis
- **Layers**: 16, 22
- **Features per layer**: 30
- **Total features**: 60
- **Time**: ~1 hour
- **Output**: Focused analysis on key layers

### Single Layer Analysis
- **Layers**: 16
- **Features per layer**: 30
- **Total features**: 30
- **Time**: ~30 minutes
- **Output**: Quick analysis of middle layer

### Custom Analysis
- **Layers**: User-specified
- **Features per layer**: 30
- **Total features**: 30 × number of layers
- **Time**: Variable
- **Output**: Custom analysis

## 📊 Output Structure

```
multi_layer_results/
├── layer_4/
│   ├── top_30_features_analysis.json
│   ├── top_30_features_analysis.csv
│   ├── feature_labels_clean_financial.csv
│   └── feature_labels_detailed_financial.csv
├── layer_10/
│   └── ... (same structure)
├── layer_16/
│   └── ... (same structure)
├── layer_22/
│   └── ... (same structure)
├── layer_28/
│   └── ... (same structure)
├── consolidated_top_features_all_layers.csv
├── layer_comparison_report.csv
├── top_50_financial_features.csv
├── financial_feature_summary.json
└── financial_feature_analysis.png
```

## 📈 Key Metrics

### Feature Analysis Metrics
- **Financial Activation**: Average activation on financial text
- **General Activation**: Average activation on general text
- **Specialization**: Financial activation - General activation
- **Ranking**: Features sorted by specialization

### Layer Performance Metrics
- **Average Specialization**: Mean specialization across all features in layer
- **Maximum Specialization**: Highest specialization in layer
- **Top Feature**: Feature with highest specialization in layer
- **Feature Count**: Number of features analyzed in layer

### Cross-Layer Analysis
- **Layer Ranking**: Layers ranked by average specialization
- **Best Layer**: Layer with highest average specialization
- **Best Feature**: Feature with highest specialization across all layers
- **Distribution Analysis**: Specialization distribution across layers

## 🔧 Configuration Options

### Layer Selection
- **Default layers**: 4, 10, 16, 22, 28 (Llama-2-7B)
- **Custom layers**: Any layer indices
- **Single layer**: Focus on specific layer
- **Layer ranges**: Analyze consecutive layers

### Feature Selection
- **Top N per layer**: Default 30, configurable
- **Minimum specialization**: Filter threshold
- **Top features N**: Number of top features to extract

### Analysis Options
- **Custom domains**: Financial, medical, legal, etc.
- **Output directory**: Custom output location
- **Visualization**: Automatic plot generation

## 📊 Example Results

### Top Financial Features (Sample)
```
Rank | Feature | Layer | Specialization | Financial Act | General Act
-----|---------|-------|----------------|---------------|-------------
  1  |   163   |  16   |     0.847      |     0.923     |    0.076
  2  |   208   |  22   |     0.792      |     0.856     |    0.064
  3  |   234   |  16   |     0.734      |     0.801     |    0.067
  4  |   182   |  22   |     0.689      |     0.745     |    0.056
  5  |   333   |  16   |     0.651      |     0.712     |    0.061
```

### Layer Performance (Sample)
```
Layer | Avg Specialization | Max Specialization | Top Feature
------|-------------------|-------------------|------------
  16  |       0.234       |       0.847       |     163
  22  |       0.198       |       0.792       |     208
  28  |       0.156       |       0.634       |     445
  10  |       0.134       |       0.567       |     278
   4  |       0.089       |       0.423       |     156
```

## 🚨 Requirements

### Dependencies
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn
```

### Hardware Requirements
- GPU with 8GB+ VRAM
- 16GB+ RAM
- Sufficient disk space for model caching

### Environment Variables
```bash
export PYTHONPATH="/home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp"
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce number of layers
   - Use smaller batch sizes
   - Process layers sequentially

2. **Long Execution Time**
   - Use fast mode (fewer layers)
   - Reduce top_n parameter

3. **File Not Found Errors**
   - Check SAE model path
   - Verify output directory permissions
   - Ensure previous steps completed

### Performance Tips

1. **Use Fast Mode**: For quick results, use layers 16, 22 only
2. **Custom Layers**: Focus on specific layers of interest
3. **Batch Processing**: Process layers in smaller batches

## 📚 Usage Examples

### Quick Analysis
```bash
python run_financial_analysis.py
# Select option 2 (fast analysis)
```

### Complete Analysis
```bash
python run_financial_analysis.py
# Select option 1 (complete analysis)
```

### Custom Layers
```bash
python multi_layer_financial_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/sae/model" \
    --layers 10 16 22 \
    --top_n 20
```

## 🎉 Expected Outcomes

After running the analysis, you'll have:

1. **Top Financial Features**: Ranked list of features with highest financial specialization
2. **Layer Performance**: Comparison of which layers contain the most financial features
3. **Statistical Analysis**: Comprehensive metrics and correlations
4. **Visualizations**: Charts and plots showing feature distributions
5. **Detailed Reports**: JSON and CSV files with all results

The system will identify the most financially-specialized features across all layers, helping you understand which parts of the model are most relevant for financial tasks.
