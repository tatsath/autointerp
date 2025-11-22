# AutoInterp Optimization: Selective Feature Activation Computation
***********************

4. Minimal “checklist” you can follow

If you just want a short checklist:

FAISS + finance embedding

ConstructorConfig(non_activating_source="FAISS", faiss_embedding_model=...)

ContrastiveExplainer

threshold=0.3, max_examples≈20, max_non_activating≈8.

Explainer system prompt

Paste the finance prompt above: forbid generic “financial news”, require JSON with granularity, focus, trigger_pattern, explanation.

Few-shots

Add ~6–10 examples covering: earnings beats/misses, downgrades, M&A, macro, sector, structural, lexical.

SamplerConfig / RunConfig

n_examples_train=40–60, n_examples_test=30–40.

min_examples_per_prompt=4, max_examples_per_prompt=8.

num_examples_per_explainer_prompt=1, num_examples_per_scorer_prompt=3–5.

Scoring strategy

First: detection + fuzz on all latents.

Then: simulation only on high-F1 latents with reduced n_examples_test.

If you want, you can paste your current YAML/JSON config and I’ll rewrite it line-by-line into a “finance-optimized Delphi config” using exactly these settings.

*******************

## Overview

This optimized version of AutoInterp modifies the activation computation to only compute activations for selected features, rather than computing all features and then filtering. This significantly reduces computation time and memory usage when analyzing a subset of features.

## Changes Made

### 1. Selective SAE Encoding (`autointerp_full/sparse_coders/load_sparsify.py`)

Added `sae_dense_latents_selective()` function that:
- Takes feature indices as input
- For non-TopK SAEs: Slices encoder weights to only compute selected features
- For TopK SAEs: Computes all features but masks to only selected features (TopK requires computing all features first)
- Reduces matrix multiplication from `[batch*seq, d_in] @ [num_latents, d_in].T` to `[batch*seq, d_in] @ [num_selected, d_in].T`

### 2. LatentCache Modification (`autointerp_full/latents/cache.py`)

Modified `LatentCache.__init__()` to:
- Accept `feature_indices` parameter (dict mapping hookpoints to feature indices)
- Wrap encoding functions to use selective encoding when feature indices are provided
- Automatically applies selective encoding during cache generation

### 3. Populate Cache Update (`autointerp_full/__main__.py`)

Modified `populate_cache()` to:
- Accept `feature_indices` parameter
- Pass feature indices to `LatentCache` for selective encoding

Updated main execution to:
- Extract feature indices from `run_cfg.feature_num` or `run_cfg.max_latents`
- Create feature_indices dictionary mapping hookpoints to feature indices
- Pass to `populate_cache()` for optimization

## How It Works

1. **Feature Selection**: When `--feature_num` is specified, the system extracts the feature indices
2. **Selective Encoding**: For each hookpoint, the encoding function is wrapped to only compute activations for selected features
3. **Optimized Computation**: 
   - Non-TopK SAEs: Only compute matrix multiplication for selected features (true optimization)
   - TopK SAEs: Compute all features but mask output (still reduces memory/storage)

## Usage

The optimized version is automatically used when you specify `--feature_num` in the command line:

```bash
python -m autointerp_full \
    model_path \
    sae_path \
    --feature_num 1 2 3 4 5 \
    ...
```

The system will automatically:
- Detect that feature indices are provided
- Use selective encoding during cache generation
- Print a message: "🔧 Using selective encoding: computing only N features per hookpoint"

## Performance Benefits

- **Computation Time**: Reduced by ~(num_selected / num_total) for non-TopK SAEs
- **Memory Usage**: Reduced activation storage during caching
- **Storage**: Only selected features are stored in cache

## Limitations

1. **TopK SAEs**: For TopK SAEs, we still compute all features internally (TopK requires this), but we mask the output. This still reduces memory/storage but not computation time.

2. **SAE Structure**: The optimization requires access to encoder weights. If the SAE structure is different, it falls back to computing all features and masking.

## Files Modified

- `autointerp_full/sparse_coders/load_sparsify.py`: Added selective encoding function
- `autointerp_full/latents/cache.py`: Added feature_indices support
- `autointerp_full/__main__.py`: Pass feature indices through the pipeline

## Testing

To test the optimization, compare:
- Original: `autointerp_full_finance/`
- Optimized: `autointerp_full_finance_optimized/`

Both should produce identical results, but the optimized version should be faster when analyzing a subset of features.

