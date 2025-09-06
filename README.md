# AutoInterp - SAE Interpretability System

Two complementary approaches for understanding what features in your SAE model have learned.

## 🎯 Approach

**Step 1:** Use AutoInterp Lite to quickly find relevant features and get basic labels. **Step 2:** Use AutoInterp Full with the exact feature numbers from Step 1 to get detailed explanations with confidence scores. AutoInterp Full can also analyze all features independently.

### 1. AutoInterp Lite - Feature Discovery
**Find relevant features in minutes, not hours**

- **Why needed**: SAE models have thousands of features. You need to find the 10-50 that matter for your domain (finance, healthcare, legal, etc.).
- **What it does**: Compares feature activations on domain-specific vs general text
- **Key metrics**: Activation strength, specialization score, top activating examples
- **Speed**: 2-5 minutes for 1000+ features
- **Output**: Ranked list of domain-relevant features with activation examples

### 2. AutoInterp Full - Feature Explanation  
**Understand what your features actually do**

- **Why needed**: Knowing a feature is "financial" isn't enough - you need to know if it detects "earnings reports" vs "market volatility"
- **What it does**: Uses LLMs to generate human-readable explanations with confidence scores
- **Key metrics**: F1 score, precision, recall, explanation quality
- **Speed**: 30-60 minutes per feature (due to LLM analysis)
- **Output**: Detailed explanations with confidence scores and validation

## 🚀 How to Run

### AutoInterp Lite - Find Relevant Features
```bash
cd autointerp_lite
python run_analysis.py --mode financial
```

### AutoInterp Full - Explain Top Features
```bash
cd autointerp_full
./example_LLM_API.sh  # Uses hardcoded features: 27,133,220,17,333
```

## 📊 Sample Outputs

### AutoInterp Lite Output
**CSV with ranked features:**
```csv
feature_number,activation_strength,specialization_score,label,top_examples
27,8.45,2.3,"Financial earnings","revenue increased by 15%","profit margin"
133,7.82,1.9,"Market data","S&P 500 closed at","stock price"
220,6.91,1.7,"Investment terms","portfolio diversification","risk assessment"
```

**Key Metrics:** Activation strength (higher = more active), specialization score (higher = more domain-specific). Good features: activation > 5.0, specialization > 1.5.

### AutoInterp Full Output
**Detailed explanations with confidence:**
```
Feature 27: "Financial earnings and revenue reporting"
F1 Score: 0.87 | Precision: 0.91 | Recall: 0.83
Explanation: Detects sentences about corporate earnings, revenue growth, and financial performance metrics.
Top Examples: "Q3 revenue increased 15%", "profit margin expanded", "earnings per share"
```

**Key Metrics:** F1 score (overall accuracy), precision (how often correct when activated), recall (how often it catches relevant cases). Good features: F1 > 0.7, precision > 0.8. Additional metrics available but these are the most important.

## 🎯 When to Use Which

**Use AutoInterp Lite when:** You have thousands of features and need to find the 10-50 that matter for your domain. Perfect for initial exploration and feature screening.

**Use AutoInterp Full when:** You need detailed explanations with confidence scores. Can analyze specific features or all features independently. Essential for research, validation, and detailed analysis.

**Typical workflow:** Run Lite first to find interesting features, then run Full on the exact feature numbers (e.g., 27,133,220) for detailed explanations.

---

**Quick Start:**
1. **Find features:** `cd autointerp_lite && python run_analysis.py --mode financial`
2. **Explain features:** `cd autointerp_full && ./example_LLM_API.sh`