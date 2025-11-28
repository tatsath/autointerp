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

*************************

1. Overall order of fixes (high level)
In Delphi, for more specific finance labels, do things in this order:


Turn on FAISS + ContrastiveExplainer with a finance embedding model.


Rewrite the explainer system prompt to forbid generic finance labels and force a structured JSON output.


Add 6–10 finance few-shots that demonstrate the level of specificity you want.


Tune explainer & sampler parameters


contrastive: threshold, max_examples, max_non_activating


sampler: min/max examples per prompt, etc.




(Optional but helpful) Use detection + fuzz first; simulation only on a filtered subset.


I’m assuming your SAE is already “good enough”; SAE tuning is separate.

2. Prompts to paste in (explainer side)
2.1 Explainer system prompt (core thing to change)
In delphi/explainers/prompts.py (or wherever you have your system messages), replace the explainer system message with something like this:
You are analysing hidden features of a language model trained on
FINANCIAL TEXT.

You will see text snippets from:
- financial news headlines and articles
- earnings call transcripts
- SEC and other regulatory filings
- broker research notes and credit reports

For ONE hidden feature at a time, you will see multiple examples with an
activation level between 0 and 9 at the [[CURRENT TOKEN]].

Your task is to infer the SINGLE clearest description of the pattern
this feature represents, at a level useful to a finance practitioner.

CRITICAL RULES:

1. AVOID GENERIC LABELS  
   Do NOT use vague descriptions like:
   - "financial news"
   - "finance-related text"
   - "earnings reports"
   - "investment-related content"

   These are TOO VAGUE and should be treated as INCORRECT.

2. BE AS SPECIFIC AS THE DATA ALLOWS  
   Prefer labels such as:
   - "Quarterly earnings results that BEAT analyst expectations, often
      with positive guidance and stock price reaction."
   - "Rating downgrades or negative outlooks by credit rating agencies."
   - "Merger and acquisition announcements where one company buys
      another."
   - "Mentions of a company's stock ticker in parentheses after its name."
   - "Language about covenant breaches, liquidity stress, or default
      risk in credit agreements."

3. PICK A GRANULARITY LEVEL  
   Decide what kind of concept this feature is:
   - ENTITY: specific company, index, ETF, bond, etc.
   - SECTOR: sector or industry cluster (e.g. regional banks, semis).
   - EVENT: discrete event (earnings beat/miss, downgrade, M&A,
     guidance change, dividend cut, etc.).
   - MACRO: macro or policy concepts (e.g. Fed hikes, inflation, recessions).
   - STRUCTURAL: document format (tickers in parentheses, bullet lists,
     section headers, disclaimers, boilerplate).
   - LEXICAL: specific phrase or token (e.g. "EBITDA margin", "GAAP").

4. CONTRAST HIGH vs LOW ACTIVATIONS  
   Focus on what separates HIGH activations (7–9) from LOW activations
   (0–2). If both high and low examples mention finance, that is NOT the
   distinguishing factor.

5. OUTPUT STRICT JSON ONLY

Return exactly this JSON:

{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "If applicable, the entity/sector/event name (e.g. 'Apple Inc.', 'earnings beats', 'Fed rate hikes'); otherwise 'N/A'.",
  "trigger_pattern": "A precise description of what tends to appear near the [[CURRENT TOKEN]] when the feature activates.",
  "explanation": "A single concise sentence that a finance practitioner can read as the feature's label."
}

Do NOT output anything else besides this JSON.


Contrastive explainer uses the same system prompt – it just changes how examples are packaged.


2.2 Few-shot examples for the explainer
In the explainer’s few-shot template, include 4–8 hand-written synthetic latents that look like your dream output.
Example 1 – earnings beat event
[Example latent: f_demo_earnings_beat]

Example 0:
Activation: 9
Text: "Apple shares rally after the company reports Q2 EPS and revenue
ABOVE analyst expectations, driven by strong iPhone and services demand."

Example 1:
Activation: 8
Text: "Tesla jumps in after-hours trading as Q2 results beat Wall Street
estimates and management RAISES full-year delivery guidance."

Example 2:
Activation: 2
Text: "Fed officials reiterate that policy will remain data-dependent."

Example 3:
Activation: 0
Text: "The company operates in the retail sector and competes with other
big-box chains."

Model output:

{
  "granularity": "EVENT",
  "focus": "Earnings beats",
  "trigger_pattern": "Quarterly results where EPS or revenue is reported as ABOVE analyst or Wall Street expectations, often with raised guidance and positive stock reaction.",
  "explanation": "Quarterly earnings announcements that BEAT analyst expectations, usually with raised guidance and a positive move in the stock."
}

Example 2 – rating downgrade
[Example latent: f_demo_rating_downgrade]

Example 0:
Activation: 8
Text: "Moody's downgrades XYZ Corp to Ba2 from Baa3 and maintains a
negative outlook due to rising leverage."

Example 1:
Activation: 7
Text: "S&P cuts the company's credit rating to junk status following a
sharp deterioration in cash flows."

Example 2:
Activation: 1
Text: "The company announces a new share buyback program."

Example 3:
Activation: 0
Text: "US stocks climbed on Monday, led by gains in technology shares."

Model output:

{
  "granularity": "EVENT",
  "focus": "Credit rating downgrades",
  "trigger_pattern": "Mentions of rating agencies (Moody's, S&P, Fitch) lowering a company's credit rating or assigning a negative outlook.",
  "explanation": "News about credit rating downgrades or negative outlooks issued by rating agencies for specific companies."
}

Add similar few-shots for:


ENTITY (e.g. “S&P 500 index level / moves”)


SECTOR (“regional banks selloff”)


MACRO (“Fed rate hikes / tightening”)


STRUCTURAL (“ticker in parentheses after company name”, “safe harbor disclaimer”)


LEXICAL (“EBITDA margin”, “free cash flow”, etc.)


These few-shots are hugely important; Delphi will imitate their specificity.

2.3 (Optional) Detection scorer prompt (keep simple)
You don’t need to change scorer prompts to get more specific labels, but if you want a finance-aware detection prompt:
You are an expert in financial language.

You are given:
- A "latent explanation" that describes a pattern in financial text.
- ONE text example at a time (headline, article snippet, filing
  excerpt, or transcript segment).

Decide if the example CLEARLY exhibits the described pattern.

Return:
- 1 if the text clearly matches the described concept.
- 0 if the concept is absent, only weakly implied, or too generic.

Be strict: if the example is just "financial news" but the explanation
is about a specific event (e.g. earnings beats, downgrades, M&A), return 0.

Answer with a single character: 1 or 0.


3. Config knobs & recommended values (for specificity + speed)
Now, the “minimum number of examples”, “number of prompts”, etc. – here’s how I’d set them for finance autointerp.
3.1 ConstructorConfig – turn on FAISS + change embedding model
In your dataset construction:
from delphi.config import ConstructorConfig

constructor_cfg = ConstructorConfig(
    non_activating_source="FAISS",
    faiss_embedding_model="FinLang/finance-embeddings-investopedia",  # or your finance embedding
    faiss_embedding_cache_enabled=True,
    faiss_embedding_cache_dir=".embedding_cache",
)

Key points:


non_activating_source="FAISS" → uses FAISS for hard negatives and automatically switches to ContrastiveExplainer.


Choose a finance embedding so negatives are semantically close within finance.



3.2 ContrastiveExplainer parameters
When you create the explainer:
from delphi.explainers import ContrastiveExplainer

explainer = ContrastiveExplainer(
    client,
    tokenizer=dataset.tokenizer,
    threshold=0.3,        # keep only clearly high vs low activations
    max_examples=20,      # total positive examples per latent
    max_non_activating=8, # hard negatives per latent
    verbose=True,
)

Suggested ranges:


threshold: 0.3–0.4


Lower → more noisy positives; higher → fewer, but very clear.




max_examples: 15–25


More examples helps the LLM see nuance; 20 is a good middle.




max_non_activating: 5–10


Enough hard negatives so it’s forced to distinguish earnings beat vs generic news.





3.3 SamplerConfig – number of examples per latent
Inside SamplerConfig (or the config you pass into LatentDataset):


For explanation training:


n_examples_train: 40–60


Examples per latent used to generate the explanation prompt.




min_examples_per_prompt: 4


max_examples_per_prompt: 8


→ You want 1–2 explainer prompts per latent, each with a nice pack of 4–8 examples (matching the few-shot pattern).


For scoring:


n_examples_test: 30–40


Examples for detection/fuzz/simulation.




For simulation specifically, you can cut this later to 10–20 if it’s too slow.




If your config doesn’t have exactly these names, look for the equivalent: “number of train/test examples per feature” and “min/max examples per prompt”.

3.4 RunConfig – number of examples per prompt (explainer + scorer)
When you set up the run:


Explainer prompts:


num_examples_per_explainer_prompt: 1
(i.e., one latent per LLM call; Delphi usually packs multiple examples of that latent into one prompt.)




Scorer prompts:


num_examples_per_scorer_prompt: 3–5




Rationale:


Keeping num_examples_per_explainer_prompt=1 avoids cross-latent confusion and keeps explanations sharp.


For scorers, batching 3–5 examples per LLM call gives speed-up without much quality loss.



3.5 Scorers – how to combine detection, fuzz, simulation
For speed + usefulness:


First pass – all latents:
--scorers detection fuzz

or programmatically, a pipe with both RecallScorer and FuzzingScorer as shown in the README.


Filter features by:


detection F1 ≥ 0.7


fuzz F1 ≥ 0.6




Second pass – only on filtered latents:
--scorers simulation
--max_latents 300   # or pass a filtered latent list



And for simulation specifically:


Use smaller n_examples_test (10–20) to keep it tractable.


Keep all_at_once=True if you see that flag in the simulator; it uses logprob mode and is faster.



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

