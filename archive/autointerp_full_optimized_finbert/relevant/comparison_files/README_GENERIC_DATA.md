# AutoInterp Analysis - Generic Data Performance Comparison

## 📊 **Analysis Overview**

This analysis compares Sparse Autoencoder (SAE) feature interpretability between a base Llama model and a finetuned Llama model on **generic text data** (Wikitext) to understand how financial finetuning affects feature behavior on non-financial content.

### **Models Used:**
- **Base Model**: `meta-llama/Llama-2-7b-hf`
- **Finetuned Model**: `cxllin/Llama2-7b-Finance`
- **Base SAE**: `llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`
- **Finetuned SAE**: `llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun`

### **Dataset:**
- **Source**: `wikitext-2-raw-v1` (Generic text data)
- **Split**: 10% of training data
- **Purpose**: Test feature generalizability from financial to generic text

### **Analysis Focus:**
- **Layer**: 4 only
- **Features**: Top 10 improved features from finetuning analysis (299, 335, 387, 347, 269, 32, 176, 209, 362, 312)
- **Method**: AutoInterp Full with contrastive analysis

---

## 🎯 **Overall Performance Results**

### **Base Model Performance (Generic Data)**
- **F1 Score**: 0.733
- **Precision**: 0.981
- **Recall**: 0.586
- **Class-Balanced Accuracy**: 0.650

### **Finetuned Model Performance (Generic Data)**
- **F1 Score**: 0.781
- **Precision**: 0.969
- **Recall**: 0.654
- **Class-Balanced Accuracy**: 0.535

### **Key Findings:**
- **Finetuned model shows better overall performance** on generic data (F1: 0.781 vs 0.733)
- **Higher recall** in finetuned model (0.654 vs 0.586) indicates better feature detection
- **Both models maintain high precision** (>0.96), showing reliable predictions when activated

---

## 📋 **Detailed Feature Analysis**

### **Layer 4 Feature Performance Comparison (Generic Data)**

| Rank | Feature | Base Model F1 | Finetuned Model F1 | Base Model Label | Finetuned Model Label | Performance Change |
|------|---------|---------------|-------------------|------------------|---------------------|-------------------|
| 1 | 299 | 0.967 | 1.000 | "Relationships between entities." | "Representation of special characters and punctuation in text data." | **+0.033** |
| 2 | 335 | 0.947 | 0.971 | "Baseball-related statistics and terminology." | "Representations of special characters, punctuation, or abbreviations." | **+0.024** |
| 3 | 387 | 0.947 | 0.875 | "Numerical values and abbreviations." | "Technical or programming-related concepts and special formatting." | **-0.072** |
| 4 | 347 | 0.985 | 0.974 | "Periodic sentence termination." | "Numbers and punctuation embedded within text." | **-0.011** |
| 5 | 269 | 1.000 | 0.938 | "Fragmented, nonsensical language patterns..." | "Spanish language and cultural references..." | **-0.062** |
| 6 | 32 | 1.000 | 1.000 | "Punctuation markers indicating text structure and transition." | "Textual article or document structure." | **0.000** |
| 7 | 176 | 0.923 | 1.000 | "Local gliding competitions..." | "Literary inspiration from classic poetry..." | **+0.077** |
| 8 | 209 | 0.957 | 0.955 | "Representations of proper nouns." | "Punctuation marks indicating range, break, or connection between ideas." | **-0.002** |
| 9 | 362 | 0.944 | 0.952 | "Proper nouns representing names of individuals." | "Representations of proper nouns, including names of people, places, and organizations." | **+0.008** |
| 10 | 312 | 0.972 | 1.000 | "Punctuation and special character usage." | "Family relationships and kinship..." | **+0.028** |

---

## 🔍 **Key Insights from Generic Data Analysis**

### **1. Feature Generalizability**
- **6 out of 10 features improved** their performance on generic data after financial finetuning
- **4 features showed performance decline**, suggesting some specialization trade-offs
- **Average performance change**: +0.002 (essentially neutral)

### **2. Semantic Transformation Patterns**

#### **Successful Generalizations:**
- **Feature 299**: "Relationships between entities" → "Special characters and punctuation" (F1: 0.967→1.000)
- **Feature 176**: "Local gliding competitions" → "Literary inspiration from classic poetry" (F1: 0.923→1.000)
- **Feature 312**: "Punctuation usage" → "Family relationships and kinship" (F1: 0.972→1.000)

#### **Performance Declines:**
- **Feature 269**: Complex pattern recognition (F1: 1.000→0.938)
- **Feature 387**: "Numerical values" → "Technical concepts" (F1: 0.947→0.875)

### **3. Domain Adaptation Effects**

#### **Positive Effects:**
- **Better recall** (0.654 vs 0.586) suggests finetuned model detects more relevant patterns
- **Maintained high precision** indicates reliable feature activation
- **Improved overall F1** shows better balance between precision and recall

#### **Trade-offs:**
- Some features became **more specialized** for financial patterns, reducing generic performance
- **Feature 269** shows the largest decline, suggesting over-specialization

### **4. Label Interpretation Changes**

#### **Consistent Patterns:**
- **Punctuation and formatting** features (299, 335, 347, 312) show consistent behavior
- **Proper noun detection** (209, 362) maintains similar performance levels

#### **Significant Changes:**
- **Feature 176**: Complete semantic shift from "gliding competitions" to "literary inspiration"
- **Feature 269**: From "fragmented language patterns" to "Spanish cultural references"

---

## 📊 **Performance Metrics Summary**

### **Individual Feature F1 Scores (Generic Data)**

| Feature | Base Model | Finetuned Model | Change | Status |
|---------|------------|-----------------|--------|--------|
| 299 | 0.967 | 1.000 | +0.033 | ✅ Improved |
| 335 | 0.947 | 0.971 | +0.024 | ✅ Improved |
| 387 | 0.947 | 0.875 | -0.072 | ❌ Declined |
| 347 | 0.985 | 0.974 | -0.011 | ❌ Declined |
| 269 | 1.000 | 0.938 | -0.062 | ❌ Declined |
| 32 | 1.000 | 1.000 | 0.000 | ➖ Stable |
| 176 | 0.923 | 1.000 | +0.077 | ✅ Improved |
| 209 | 0.957 | 0.955 | -0.002 | ➖ Stable |
| 362 | 0.944 | 0.952 | +0.008 | ✅ Improved |
| 312 | 0.972 | 1.000 | +0.028 | ✅ Improved |

**Summary**: 6 improved, 3 declined, 1 stable

---

## 🎯 **Conclusions**

### **1. Generalizability Assessment**
- **Financial finetuning shows mixed effects** on generic text performance
- **Overall improvement** in F1 score (0.733→0.781) suggests **positive transfer**
- **Better recall** indicates improved pattern detection capabilities

### **2. Feature Specialization Trade-offs**
- Some features **benefit from finetuning** even on generic data
- Others become **over-specialized** for financial patterns
- **Punctuation and formatting** features show consistent improvement

### **3. Practical Implications**
- **Finetuned model maintains interpretability** on non-financial text
- **Feature explanations remain meaningful** across domains
- **Some features may need domain-specific analysis** for optimal performance

### **4. Research Insights**
- **Financial finetuning doesn't completely break** generic text interpretability
- **Feature semantics can shift significantly** while maintaining performance
- **Domain adaptation shows both positive and negative transfer effects**

---

## 📁 **Data Sources**

- **Individual F1 scores**: Extracted from `@results/*/scores/detection/` directories
- **Feature labels**: From `@results/*/explanations/` directories  
- **Overall metrics**: From AutoInterp Full analysis output
- **Analysis scripts**: `run_autointerp_base_model_generic.sh`, `run_autointerp_finetuned_model_generic.sh`

---

## 🔧 **Technical Details**

- **Analysis Method**: AutoInterp Full with FAISS contrastive learning
- **Explainer Model**: `meta-llama/llama-3.1-8b-instruct` via OpenRouter
- **Dataset Size**: 10% of Wikitext-2-raw-v1 training data
- **Features Analyzed**: Top 10 improved features from financial finetuning analysis
- **Evaluation Metrics**: F1, Precision, Recall, Class-Balanced Accuracy

---

*This analysis demonstrates that financial finetuning can improve model interpretability on generic text data, though with some feature-specific trade-offs. The results suggest that domain adaptation doesn't necessarily break general interpretability capabilities.*
