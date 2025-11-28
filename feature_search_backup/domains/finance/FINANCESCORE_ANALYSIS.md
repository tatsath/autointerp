# FinanceScore Implementation Analysis

## ✅ What Matches the ReasonScore Paper

### 1. **Core Formula** ✓ CORRECT
The implementation uses the **exact ReasonScore formula**:

```python
# From compute_score.py line 438-441
scores = (
    (all_feature_means[:, 0] / sum_pos) * h_norm**alpha
    - (all_feature_means[:, 1] / sum_neg)
)
```

This matches the paper's formula:
- `mean_pos / sum_pos` = normalized activation in finance contexts
- `h_norm^alpha` = entropy penalty (penalizes features that only fire on one token)
- `mean_neg / sum_neg` = normalized activation in non-finance contexts

**Status**: ✅ **CORRECT** - Formula matches paper exactly

### 2. **Context Window Building** ✓ CORRECT
- Uses `expand_range=[2, 3]` (2 tokens before, 3 after)
- Matches paper's approach of expanding around matched tokens
- Implementation in `RollingMean._compute_single_mask()` correctly handles window expansion

**Status**: ✅ **CORRECT**

### 3. **Entropy Penalty** ✓ CORRECT
- Computes entropy: `h = -(probs * log_probs).sum(dim=1)`
- Normalizes: `h_norm = h / math.log(probs.size(1))`
- Applies penalty: `h_norm^alpha` where `alpha=0.7`
- This penalizes features that only activate on one finance token

**Status**: ✅ **CORRECT** - Matches paper methodology

### 4. **Dataset Usage** ✓ CORRECT
- Uses finance dataset: `jyanimaulik/yahoo_finance_stockmarket_news`
- Separates positive (finance token contexts) vs negative (all other positions)
- Both come from same dataset (as in paper)

**Status**: ✅ **CORRECT**

---

## ❌ What Needs Improvement

### 1. **Finance Vocabulary Quality** ❌ **INSUFFICIENT**

**Current tokens** (`finance_tokens.json`):
```json
["stock", " price", "market", " earnings", "revenue", "profit", 
 "dividend", "IPO", "share", "trading", "investment", "portfolio", 
 "NASDAQ", "NYSE", "Dow", "S&P", "index"]
```

**Problems**:
- ❌ Only **single words**, not phrases
- ❌ Too **generic** (appear in many contexts)
- ❌ Missing **finance-specific phrases** that indicate financial reasoning

**What the paper recommends**:
- Phrases like: "earnings miss", "guidance cut", "default risk", "liquidity risk"
- Action phrases: "downgrade", "upgrade", "buyback", "dividend increase"
- Event phrases: "profit warning", "credit rating cut", "share buyback"

**Recommendation**: 
Create a proper `finance_vocab.txt` with:
1. **Finance-specific phrases** (not just words)
2. **Action-oriented terms** (downgrade, upgrade, miss, beat)
3. **Risk indicators** (default risk, liquidity risk, credit risk)
4. **Market events** (earnings announcement, guidance, buyback)

**Example improved vocabulary**:
```
earnings miss
earnings beat
guidance cut
guidance raised
downgrade
upgrade
default risk
liquidity risk
credit rating cut
profit warning
share buyback
dividend increase
revenue growth
profit margin
trading volume
market volatility
interest rate hike
rate cut
IPO pricing
merger announcement
```

### 2. **Sample Size** ⚠️ **TOO SMALL**

**Current**: `n_samples: 100` in config.json
**Paper uses**: 10M tokens (~10,000 samples with context_size=1024)

**Impact**: 
- Small sample size → less reliable statistics
- May miss rare but important finance patterns

**Recommendation**: Increase to at least 1,000-5,000 samples

### 3. **Quantile Threshold** ⚠️ **DIFFERENT FROM PAPER**

**Current**: `quantile_threshold: 0.95` (selects top 5%)
**Paper**: Uses `q = 0.997` (selects top 0.3% = ~200 features from 65k)

**Note**: Your SAE has 400 features, so 0.95 = ~20 features (matches your `expected_top_features: 20`)

**Status**: ⚠️ **ACCEPTABLE** for smaller SAE, but different methodology

---

## 📊 Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **Formula** | ✅ CORRECT | Exact match with ReasonScore paper |
| **Context Windows** | ✅ CORRECT | Properly implements [2,3] expansion |
| **Entropy Penalty** | ✅ CORRECT | Correctly computes and applies |
| **Dataset** | ✅ CORRECT | Uses finance dataset appropriately |
| **Vocabulary** | ❌ **INSUFFICIENT** | Too generic, needs finance-specific phrases |
| **Sample Size** | ⚠️ **TOO SMALL** | 100 vs paper's 10M tokens |
| **Quantile** | ⚠️ **DIFFERENT** | 0.95 vs 0.997 (but acceptable for 400-feature SAE) |

---

## 🔧 Recommended Actions

### Priority 1: Improve Finance Vocabulary
1. Create `finance_vocab.txt` with finance-specific phrases
2. Include action terms (downgrade, upgrade, miss, beat)
3. Include risk indicators (default risk, liquidity risk)
4. Include market events (earnings, guidance, buyback)

### Priority 2: Increase Sample Size
1. Change `n_samples` from 100 to at least 1,000-5,000
2. Or enable `target_tokens: 10,000,000` calculation (currently commented out)

### Priority 3: Verify Results
1. Check if top features actually activate on finance text
2. Compare with manual inspection of activating sentences
3. Consider using Fisher score (`score_type="fisher"`) as alternative validation

---

## ✅ Conclusion

**The implementation correctly follows the ReasonScore methodology**, but the **finance vocabulary is the weakest link**. The current tokens are too generic and will likely identify features that activate on any business/finance text, not specifically financial reasoning patterns.

**The formula and mechanics are correct** - you just need better vocabulary to match the paper's quality.

