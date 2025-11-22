# AutoInterp - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Navigate to AutoInterp
```bash
cd /home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp
```

### 2. Run Complete Pipeline
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun" \
    --top_n 10 \
    --domain "financial" \
    --output_dir "results"
```

### 3. Check Results
```bash
ls results/
cat results/top_10_features_analysis.csv
cat results/feature_labels_clean_financial.csv
```

## 📋 What You Get

1. **Top Features**: Ranked by specialization (financial vs general activation)
2. **Our Labels**: Generated using LLM based on top activating sentences  
3. **Delphi Explanations**: Generated using Delphi's interpretability pipeline
4. **F1 Scores**: Using Delphi's built-in calculation
5. **Comparison Analysis**: Semantic similarity between methods

## 🎯 Different Domains

### Financial Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --domain "financial" \
    --top_n 10
```

### Medical Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --domain "medical" \
    --top_n 10
```

### Legal Analysis
```bash
python generic_master_script.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --domain "legal" \
    --top_n 10
```

## 🔧 Individual Components

### Feature Analysis Only
```bash
python generic_feature_analysis.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --top_n 10
```

### Feature Labeling Only
```bash
python generic_feature_labeling.py \
    --analysis_file "results/top_10_features_analysis.json" \
    --domain "financial"
```

### Delphi Analysis Only
```bash
python generic_delphi_runner.py \
    --base_model "meta-llama/Llama-2-7b-hf" \
    --sae_model "/path/to/your/sae/model" \
    --features "163,59,333,208,182" \
    --run_name "my_analysis"
```

## 📊 Output Files

- `top_N_features_analysis.csv`: Feature rankings and activations
- `feature_labels_clean_DOMAIN.csv`: Feature numbers and labels
- `RUN_NAME_delphi_results.csv`: Delphi explanations and F1 scores
- `comparison_results.csv`: Comparison between methods
- `comparison_analysis.png`: Visualization plots

## 🆘 Need Help?

- Check `README.md` for detailed documentation
- Check `GENERIC_SYSTEM_README.md` for technical details
- Run `python example_usage.py` for more examples
- Check the `results/` folder for sample outputs

## ✅ Requirements

- GPU with 8GB+ VRAM
- Python 3.8+
- Required packages: `torch`, `transformers`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `matplotlib`, `seaborn`
- OpenRouter API key (for Delphi explanations)
