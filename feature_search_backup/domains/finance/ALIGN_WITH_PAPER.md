# Aligning FinanceScore with ReasonScore Paper/Code

## Comparison with Actual ReasonScore Implementation

Based on the [ReasonScore code](https://github.com/AIRI-Institute/SAE-Reasoning/blob/main/extraction/scripts/compute_score.sh):

### Current vs Paper Parameters

| Parameter | ReasonScore Paper | Your Current | Status |
|-----------|-------------------|--------------|--------|
| **Formula** | `(mean_pos/sum_pos) * h_norm^alpha - (mean_neg/sum_neg)` | ✅ Same | ✅ CORRECT |
| **expand_range** | `[1, 2]` | `[2, 3]` | ⚠️ Different |
| **n_samples** | `4096` (script) / `10M tokens` (paper) | `100` | ❌ Too small |
| **alpha** | `0.7` | `0.7` | ✅ CORRECT |
| **quantile_threshold** | `0.997` (paper) | `0.95` | ⚠️ Different |
| **encoder.fold_W_dec_norm()** | ✅ Called | ✅ Called (line 361) | ✅ CORRECT |
| **epsilon** | `1e-12` | `1e-12` | ✅ CORRECT |

---

## Step-by-Step Alignment

### ✅ Step 1: Formula (Already Correct)
Your formula matches exactly:
```python
# Your code (line 438-441)
scores = (
    (all_feature_means[:, 0] / sum_pos) * h_norm**alpha
    - (all_feature_means[:, 1] / sum_neg)
)
```
**Status**: ✅ No changes needed

### ✅ Step 2: encoder.fold_W_dec_norm() (Already Called)
Your code calls it at line 361:
```python
encoder.fold_W_dec_norm()
```
**Status**: ✅ No changes needed

### ⚠️ Step 3: Fix expand_range
**ReasonScore uses**: `[1, 2]` (1 token before, 2 after)  
**You use**: `[2, 3]` (2 tokens before, 3 after)

**Action**: Update `config.json`:
```json
"expand_range": [1, 2]
```

**Note**: Your `[2, 3]` is actually MORE context, which might be fine, but to match paper exactly, use `[1, 2]`.

### ❌ Step 4: Fix Sample Size (CRITICAL)
**ReasonScore script uses**: `n_samples: 4096`  
**ReasonScore paper mentions**: `10M tokens` (~10,000 samples with context_size=1024)  
**You use**: `n_samples: 100`

**Action**: Update `config.json`:
```json
"n_samples": 10000
```

**OR** enable target_tokens calculation in `compute_finance_score.py` (uncomment lines 58-65):
```python
if 'target_tokens' in config and config['target_tokens']:
    context_size = config.get('context_size', 1024)
    calculated_samples = max(n_samples, config['target_tokens'] // context_size)
    print(f">>> Target tokens: {config['target_tokens']:,}")
    print(f">>> Context size: {context_size}")
    print(f">>> Calculated n_samples for target: {calculated_samples:,}")
    print(f">>> Expected total tokens: ~{calculated_samples * context_size:,}")
    n_samples = calculated_samples
```

This will use `10,000,000 / 1024 = ~9,766 samples` to match paper's 10M tokens.

### ⚠️ Step 5: Fix Quantile Threshold
**ReasonScore paper uses**: `q = 0.997` (top 0.3%)  
**You use**: `quantile_threshold: 0.95` (top 5%)

**Action**: Update `config.json`:
```json
"quantile_threshold": 0.997,
"expected_top_features": 1  // For 400-feature SAE: 400 * 0.003 = ~1 feature
```

**Note**: With 400 features, 0.997 = ~1 feature. If you want ~20 features, use:
```json
"quantile_threshold": 0.95,  // 400 * 0.05 = 20 features
```

**OR** for paper alignment with larger SAE:
```json
"quantile_threshold": 0.997,  // Matches paper exactly
```

### ✅ Step 6: Vocabulary Format (Already Correct)
Your `finance_tokens.json` now follows the same format as [reason_tokens.json](https://raw.githubusercontent.com/AIRI-Institute/SAE-Reasoning/main/extraction/reason_tokens.json):
- Includes variations: `"stock"`, `" stock"`, `"Stock"`, `" Stock"`
- Includes phrases: `"earnings miss"`, `"guidance cut"`, etc.

**Status**: ✅ Format matches

---

## Recommended Configuration (Paper-Aligned)

Update `config.json` to:

```json
{
    "dataset_path": "jyanimaulik/yahoo_finance_stockmarket_news",
    "tokens_str_path": "finance_tokens.json",
    "expand_range": [1, 2],
    "alpha": 0.7,
    "quantile_threshold": 0.997,
    "n_samples": 10000,
    "context_size": 1024,
    "expected_top_features": 1,
    "ignore_tokens": [128000, 128001],
    "output_dir": "scores",
    "dashboard_output_dir": "dashboards",
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "sae_path": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU",
    "sae_id": "blocks.19.hook_resid_post",
    "target_tokens": 10000000
}
```

**OR** for practical use with 400-feature SAE (to get ~20 features):

```json
{
    "dataset_path": "jyanimaulik/yahoo_finance_stockmarket_news",
    "tokens_str_path": "finance_tokens.json",
    "expand_range": [1, 2],
    "alpha": 0.7,
    "quantile_threshold": 0.95,
    "n_samples": 10000,
    "context_size": 1024,
    "expected_top_features": 20,
    "ignore_tokens": [128000, 128001],
    "output_dir": "scores",
    "dashboard_output_dir": "dashboards",
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "sae_path": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU",
    "sae_id": "blocks.19.hook_resid_post",
    "target_tokens": 10000000
}
```

---

## Summary of Required Changes

1. ✅ **Formula**: Already correct - no change
2. ✅ **encoder.fold_W_dec_norm()**: Already called - no change  
3. ⚠️ **expand_range**: Change `[2, 3]` → `[1, 2]` (to match paper exactly)
4. ❌ **n_samples**: Change `100` → `10000` (or enable target_tokens)
5. ⚠️ **quantile_threshold**: Change `0.95` → `0.997` (if you want paper alignment, but 0.95 is fine for 400-feature SAE)
6. ✅ **Vocabulary**: Format is correct, content is improved

---

## Priority Order

1. **HIGH**: Fix `n_samples` (100 → 10000) - This is critical for reliable results
2. **MEDIUM**: Fix `expand_range` ([2,3] → [1,2]) - To match paper exactly
3. **LOW**: Fix `quantile_threshold` (0.95 → 0.997) - Only if you want exact paper alignment (but 0.95 is fine for your SAE size)

