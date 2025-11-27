# Comparison: autointerp_saeeval vs autointerp_full

## Why autointerp_saeeval Produces Better Explanations

This document explains why `autointerp_saeeval` avoids the issues described in the problem analysis files and produces more reliable feature explanations.

---

## Key Differences

### 1. **Explicit Example Sorting** ✅

**autointerp_saeeval** (main.py, lines 105-107):
```python
self.examples = sorted(
    self.examples, key=lambda x: max(x.acts), reverse=True
)
```
- **Explicitly sorts** examples by maximum activation before using them
- **Guarantees** highest-activation examples come first
- **No assumptions** - sorting is enforced in code

**autointerp_full** (problem):
- Assumes `record.train` is sorted by max_activation
- No explicit sorting validation
- If data isn't sorted, gets random examples
- **Result**: May use lower-activation examples when higher ones exist

---

### 2. **Prompt Philosophy: "Common" vs "Distinctive"** ✅

**autointerp_saeeval** (main.py, line 389):
```
We will give you a list of documents on which the neuron activates, 
in order from most strongly activating to least strongly activating. 
Look at the parts of the document the neuron activates for and 
summarize in a single sentence what the neuron is activating on. 
Try not to be overly specific in your explanation. 
Your explanation should cover most or all activating words 
(for example, don't give an explanation which is specific to a 
single word if all words in a sentence cause the neuron to activate).
```

**Key instructions:**
- ✅ "in order from most strongly activating to least strongly activating" - tells LLM examples are sorted
- ✅ "Try not to be overly specific" - discourages rare patterns
- ✅ "Your explanation should cover most or all activating words" - requires broad coverage
- ✅ Explicitly warns against single-word explanations

**autointerp_full** (problem):
- Asks for "MOST DISTINCTIVE" pattern
- "Distinctive" = rare, unique, stands out
- Encourages LLM to pick rare phrases like "Smart Beta Equity ETPs globally increased"
- No instruction to find patterns that appear in multiple examples
- **Result**: Explanations based on 1-2 examples instead of common patterns

---

### 3. **Activation Strength Visibility** ✅

**autointerp_saeeval**:
- Examples are numbered 1, 2, 3... in order of activation strength
- Prompt explicitly states: "in order from most strongly activating to least strongly activating"
- LLM understands Example 1 is more important than Example 10
- **Result**: LLM naturally weights higher-activation examples more

**autointerp_full** (problem):
- Examples numbered 1, 2, 3... but no indication of activation strength
- No activation values shown in prompt
- LLM treats all examples equally
- **Result**: May focus on lower-activation examples (like Example 10 with 7.125) when top examples (11.0, 10.125) show different patterns

---

### 4. **Example Selection Strategy** ✅

**autointerp_saeeval** (gather_data method, lines 472-538):
- Uses **top-k examples** (highest activations)
- Uses **importance-weighted sampling** (medium activations, diverse examples)
- Combines both for generation phase
- **Result**: Gets both strong activations AND diverse patterns

**autointerp_full** (problem):
- Only uses top-k examples
- No importance-weighted sampling
- May miss patterns that appear in medium-activation examples
- **Result**: Less diverse example set

---

### 5. **Validation via Scoring Phase** ✅

**autointerp_saeeval** (scoring phase):
- Generates explanation in generation phase
- Tests explanation in scoring phase: "Which sequences will activate?"
- Compares LLM predictions to actual activations
- Calculates AutoInterp score (fraction of correct predictions)
- **Result**: Bad explanations get low scores, can be filtered out

**autointerp_full** (problem):
- No scoring phase
- No validation that explanation matches multiple examples
- No way to detect if explanation is based on 1-2 examples
- **Result**: Bad explanations go undetected

---

### 6. **Importance-Weighted Sampling** ✅

**autointerp_saeeval** (lines 487-501):
```python
# Get importance-weighted examples, using a threshold so they're disjoint from top examples
threshold = top_values[:, self.cfg.buffer].min().item()
acts_thresholded = torch.where(acts[..., i] >= threshold, 0.0, acts[..., i])
iw_indices = get_iw_sample_indices(
    acts_thresholded, k=self.cfg.n_iw_sampled_ex, buffer=self.cfg.buffer
)
```

- Samples examples with **medium activations** (below top-k threshold)
- Ensures diversity beyond just highest activations
- Helps capture patterns that appear in multiple examples
- **Result**: More representative example set

**autointerp_full** (problem):
- Only uses top-k examples
- May miss patterns that appear in medium-activation examples
- **Result**: Less diverse, potentially biased toward edge cases

---

## Why Feature 18529 Failed in autointerp_full

Based on the problem analysis:

1. **The LLM saw 15 examples** (presumably sorted, but not guaranteed)
2. **Example 10** contained "Smart Beta Equity ETPs globally increased" (activation=7.125)
3. **Top examples** (11.0, 10.125, 9.75) showed different patterns
4. **Prompt asked for "MOST DISTINCTIVE"** pattern
5. **LLM focused on the rare phrase** (it's distinctive!)
6. **LLM ignored** that:
   - This phrase appears in only 1-2 examples
   - Top examples show different patterns
   - Common patterns across examples are different

**The LLM did what it was asked**: find the "most distinctive" pattern. But "distinctive" doesn't mean "representative" - it means "rare and unique", which is exactly what happened.

---

## Why autointerp_saeeval Would Have Succeeded

1. **Explicit sorting** ensures top examples (11.0, 10.125, 9.75) come first
2. **Prompt says "in order from most strongly activating"** - LLM knows Example 1 is most important
3. **Prompt says "Try not to be overly specific"** - discourages rare phrases
4. **Prompt says "cover most or all activating words"** - requires broad coverage
5. **Importance-weighted sampling** provides diverse examples beyond top-k
6. **Scoring phase** would catch if explanation only matches 1-2 examples

**Result**: Explanation would focus on common patterns across top examples, not rare phrases from lower-activation examples.

---

## Summary Table

| Aspect | autointerp_saeeval | autointerp_full |
|--------|-------------------|-----------------|
| **Example Sorting** | ✅ Explicitly sorted by max activation | ❌ Assumes sorted, no validation |
| **Prompt Philosophy** | ✅ "Common patterns", "cover most examples" | ❌ "MOST DISTINCTIVE" (encourages rare) |
| **Activation Visibility** | ✅ "in order from most strongly activating" | ❌ No indication of activation strength |
| **Example Selection** | ✅ Top-k + importance-weighted sampling | ❌ Only top-k |
| **Validation** | ✅ Scoring phase tests explanation quality | ❌ No validation |
| **Result** | ✅ Representative explanations | ❌ May be based on 1-2 examples |

---

## Recommendations for autointerp_full

Based on this analysis, to fix autointerp_full:

1. **Explicitly sort examples** by max_activation before using them
2. **Change prompt** from "MOST DISTINCTIVE" to "MOST COMMON pattern across examples"
3. **Add activation values** to prompt: "Example 1 (max_activation=11.0): ..."
4. **Add instruction**: "The explanation must represent AT LEAST 5-10 examples"
5. **Add scoring phase** to validate explanation quality
6. **Use importance-weighted sampling** for diverse examples

These changes would align autointerp_full with the proven approach in autointerp_saeeval.

