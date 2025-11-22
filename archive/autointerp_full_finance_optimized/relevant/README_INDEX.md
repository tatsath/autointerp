# Relevant Files Index - Finetuning Impact Analysis

This folder contains all the relevant files for the comprehensive finetuning impact analysis comparing base and finetuned Llama models on financial data.

## 📁 **Folder Structure**

```
relevant/
├── README_INDEX.md                    # This index file
├── README_FINAL.md                   # Final comprehensive README with all layers
├── README_UPDATED.md                 # Updated README (backup)
├── all_layers_comprehensive_data.json # Complete analysis data for all layers
├── scripts/                          # Analysis scripts
│   ├── extract_all_layers_data.py    # Script to extract data from all layers
│   └── generate_updated_readme.py    # Script to generate comprehensive README
├── comparison_files/                 # Original comparison analysis files
│   ├── extract_f1_scores.py         # Original F1 score extraction script
│   ├── extract_individual_feature_metrics.py # Individual feature analysis
│   ├── f1_scores_generic.json       # F1 scores for generic data
│   ├── optimized_results.json       # Optimized analysis results
│   ├── finetuned_model_analysis.log # Analysis log file
│   ├── README_Old.md                # Original README (before all layers)
│   └── README_GENERIC_DATA.md       # Generic data analysis README
└── model_outputs/                    # Model outputs for all layers
    ├── base_model_layer4/           # Base model layer 4 outputs
    ├── finetuned_model_layer4/      # Finetuned model layer 4 outputs
    ├── base_model_layer10/          # Base model layer 10 outputs
    ├── finetuned_model_layer10/     # Finetuned model layer 10 outputs
    ├── base_model_layer16/          # Base model layer 16 outputs
    ├── finetuned_model_layer16/     # Finetuned model layer 16 outputs
    ├── base_model_layer22/          # Base model layer 22 outputs
    ├── finetuned_model_layer22/     # Finetuned model layer 22 outputs
    ├── base_model_layer28/          # Base model layer 28 outputs
    └── finetuned_model_layer28/     # Finetuned model layer 28 outputs
```

## 🎯 **Key Files for Final Comparison**

### **Primary Analysis Files:**
1. **`README_FINAL.md`** - Complete comprehensive analysis with all layers (4, 10, 16, 22, 28)
2. **`all_layers_comprehensive_data.json`** - Raw data containing all feature comparisons, F1 scores, and activation improvements

### **Scripts for Reproduction:**
1. **`scripts/extract_all_layers_data.py`** - Extracts data from all layer directories
2. **`scripts/generate_updated_readme.py`** - Generates the comprehensive README

## 📊 **Files for Initial Activation Difference Analysis**

### **Original Analysis Files:**
1. **`comparison_files/README_Old.md`** - Original analysis (before all layers were included)
2. **`comparison_files/optimized_results.json`** - Optimized analysis results
3. **`comparison_files/f1_scores_generic.json`** - F1 scores for generic data analysis
4. **`comparison_files/extract_individual_feature_metrics.py`** - Individual feature analysis script
5. **`comparison_files/extract_f1_scores.py`** - F1 score extraction script

## 🔍 **Model Output Files (Most Relevant Features)**

### **Layer 4 (Early Layer - Moderate Improvements):**
- **Base Model**: `model_outputs/base_model_layer4/explanations/` - Feature explanations
- **Finetuned Model**: `model_outputs/finetuned_model_layer4/explanations/` - Feature explanations
- **Detection Scores**: `model_outputs/*/scores/detection/` - F1 scores and activation data

### **Layer 10 (Intermediate Layer):**
- Similar structure as Layer 4

### **Layer 16 (Middle Layer - Enhanced Specialization):**
- Similar structure as Layer 4

### **Layer 22 (Advanced Layer - Major Improvements):**
- Similar structure as Layer 4

### **Layer 28 (Peak Layer - Maximum Improvements):**
- Similar structure as Layer 4

## 🏆 **Most Relevant Features by Layer**

### **Layer 4 Top Features:**
- Feature 299: +0.6727 activation improvement (Intellectual achievements → Financial Market Analysis)
- Feature 32: +0.1467 activation improvement (Punctuation → Financial market terminology)
- Feature 347: +0.0950 activation improvement (Investment advice → Date specification)

### **Layer 10 Top Features:**
- Feature 83: +1.3475 activation improvement (Textual references → Financial market trends)
- Feature 162: +0.3599 activation improvement (Economic growth → Financial Market Analysis)
- Feature 91: +0.3233 activation improvement (Transitional phrases → Two-digit year representation)

### **Layer 16 Top Features:**
- Feature 389: +2.1744 activation improvement (Financial numbers → Financial performance indicators)
- Feature 85: +1.9325 activation improvement (Dates and financial numbers → Financial News)
- Feature 385: +0.6014 activation improvement (Financial Market Analysis → Financial market entities)

### **Layer 22 Top Features:**
- Feature 159: +3.4074 activation improvement (Article titles → Financial Performance and Growth)
- Feature 258: +3.1284 activation improvement (Temporal relationships → Financial market indicators)
- Feature 116: +2.3812 activation improvement (Names/Identifiers → Financial themes)

### **Layer 28 Top Features:**
- Feature 116: +5.1236 activation improvement (Prepositional phrases → Company financial performance)
- Feature 375: +3.8403 activation improvement (Punctuation → Financial market terminology)
- Feature 276: +2.2481 activation improvement (Assertion of existence → Temporal relationships)

## 📈 **Summary Statistics**

| Layer | Avg Activation Improvement | Avg F1 Improvement | Best Activation Improvement | Best F1 Improvement |
|-------|---------------------------|-------------------|---------------------------|-------------------|
| 4     | +0.111                    | +0.060            | +0.6727                   | +0.830            |
| 10    | +0.219                    | +0.068            | +1.3475                   | +0.791            |
| 16    | +0.680                    | +0.068            | +2.1744                   | +0.417            |
| 22    | +0.945                    | +0.075            | +3.4074                   | +0.617            |
| 28    | +1.494                    | -0.030            | +5.1236                   | +0.764            |

## 🚀 **Usage Instructions**

1. **To reproduce the analysis**: Run `scripts/extract_all_layers_data.py`
2. **To regenerate README**: Run `scripts/generate_updated_readme.py`
3. **To view results**: Open `README_FINAL.md` for comprehensive analysis
4. **To explore specific features**: Check individual explanation files in `model_outputs/`

## 📝 **File Descriptions**

- **Explanation files** (`.txt`): Contain feature labels and descriptions
- **Detection files** (`.txt`): Contain activation data, F1 scores, and prediction results
- **JSON files**: Contain structured data for analysis
- **Scripts**: Python files for data extraction and analysis
- **README files**: Documentation and analysis reports
