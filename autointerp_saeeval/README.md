# AutoInterp SAE Evaluation

Tool for generating human-readable explanations for SAE features. Evaluates how well an LLM explains and predicts feature activations. Provides interpretability scores for sparse autoencoder features. Supported via vLLM.

## Table of Contents

- [Overview](#overview)
- [What AutoInterp Eval Returns](#what-autointerp-eval-returns)
- [Usage](#usage)
  - [How to Run Evaluation for Any LLM](#how-to-run-evaluation-for-any-llm)
  - [Example: FinBERT Evaluation](#example-finbert-evaluation)
  - [Example: Nemotron Evaluation](#example-nemotron-evaluation)
- [Example Results](#example-results)
  - [CSV Summary Examples](#csv-summary-examples)
  - [Top Activating Examples](#top-activating-examples)
  - [Where to Find Detailed Results](#where-to-find-detailed-results)
- [Method](#method)
  - [Collecting Activation Evidence](#1-collecting-activation-evidence)
  - [Natural-Language Explanation via LLM](#2-natural-language-explanation-via-llm-the-prompt)
  - [Evaluation and Scoring](#3-evaluation-and-scoring)
- [Files](#files)
- [Dependencies](#dependencies)
- [Dependencies](#dependencies)
- [Relevant Files](#relevant-files)
- [Results Location](#results-location)
- [Why AutoInterp Eval Is Model-Agnostic](#why-autointerp-eval-is-model-agnostic)
- [Comparison With Delphi](#comparison-with-delphi)

## Overview

AutoInterp Eval generates human-readable explanations for SAE features. Each feature represents a direction in the model's hidden-state space. AutoInterp provides a natural-language label describing the pattern.

Key idea: observe where features activate strongly, send examples to an LLM, and let it explain the pattern. Domain-agnostic and model-agnostic. Works with any LLM (Nemotron, Llama, FinBERT, etc.) using hidden activations and text examples. Applicable to finance, sports, legal text, or any domain.

## What AutoInterp Eval Returns

AutoInterp Eval provides exactly four outputs for each feature:

**Label/Explanation**: Short descriptive phrase (parsed from LLM response).

**Score**: AutoInterp score (0 to 1) - fraction of sequences where LLM correctly predicted activation.

**Predictions**: List of sequence indices the LLM predicted would activate.

**Correct Sequences**: List of sequence indices that actually activated.

Note: Confidence is measured by the AutoInterp score. Good explanations lead to higher scores.

## Usage

**Note**: Currently supports vLLM provider only.

### How to Run Evaluation for Any LLM

#### Step 1: Start vLLM Server

Start a vLLM server with your explainer model:

```bash
bash start_vllm_server_72b.sh

# Or manually:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --port 8002
```

Wait until the server is loaded.

#### Step 2: Create or Modify an Evaluation Script

Create a new Python script or modify an existing one. The script should:

1. **Import required modules:**
```python
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file

# Add local autointerp module to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import autointerp.eval_config as autointerp_config
import autointerp.main as autointerp_main
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.topk_sae as topk_sae
```

2. **Configure model and SAE paths:**
```python
MODEL_NAME = "your-model-name"  # e.g., "meta-llama/Llama-3.1-8B-Instruct"
SAE_PATH = Path("/path/to/your/sae")
LAYER = 10  # Layer to evaluate
FEATURES_TO_EVALUATE = [0, 1, 2, 3, 4]  # Features to evaluate
RESULTS_DIR = str(SCRIPT_DIR / "Results")
```

3. **Set evaluation hyperparameters:**
```python
TOTAL_TOKENS = 500_000  # See hyperparameter guide below
CONTEXT_SIZE = 1024  # Match your model's context length
LLM_BATCH_SIZE = 16  # Adjust based on GPU memory
LLM_DTYPE = "bfloat16"  # or "float16", "float32"
TORCH_DTYPE = torch.bfloat16
FORCE_RERUN = True
```

4. **Configure vLLM provider:**
```python
PROVIDER = "vllm"
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Model for generating explanations
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"
API_KEY_PATH = Path(__file__).parent / "openai_api_key.txt"  # Optional
```

5. **Load SAE and run evaluation:**
```python
# Load SAE (implementation depends on SAE format)
sae = load_your_sae(SAE_PATH, MODEL_NAME, device, dtype, LAYER)

# Create config
config = autointerp_config.AutoInterpEvalConfig(
    model_name=MODEL_NAME,
    n_latents=None,
    override_latents=FEATURES_TO_EVALUATE,
    total_tokens=TOTAL_TOKENS,
    llm_context_size=CONTEXT_SIZE,
    llm_batch_size=LLM_BATCH_SIZE,
    llm_dtype=LLM_DTYPE,
    scoring=True,
    dataset_name="your-dataset",  # e.g., "ashraq/financial-news"
    # ... other hyperparameters
)

# Run evaluation
results = autointerp_main.run_eval(
    config=config,
    selected_saes=[(sae_id, sae)],
    device=device,
    api_key=api_key,
    output_path=RESULTS_DIR,
    provider=PROVIDER,
    api_base_url=EXPLAINER_API_BASE_URL,
    explainer_model=EXPLAINER_MODEL,
)
```

#### Step 3: Configuration Parameters

**All parameters are specified INSIDE the code** (not via command-line). Key parameters to modify:

**Location**: Parameters are defined as constants at the top of your evaluation script (e.g., `TOTAL_TOKENS = 500_000`).

**Most Important Parameters:**
- `total_tokens`: Total tokens to sample from dataset (default: 2M, recommended: 100k-1M for testing, 1M-2M for production)
- `n_top_ex_for_generation`: Number of top examples for explanation (default: 10, range: 5-20)
- `n_random_ex_for_scoring`: Random examples for scoring (default: 10, range: 5-20)
- `llm_context_size`: Context window size (default: 128-1024, match your model)
- `llm_batch_size`: Batch size for processing (default: 16-32)

**Additional parameters** can be passed to `AutoInterpEvalConfig`:
- `n_iw_sampled_ex_for_generation` (default: 5): Importance-weighted samples
- `act_threshold_frac` (default: 0.01): Activation threshold fraction
- `max_tokens_in_explanation` (default: 30): Max tokens in explanation
- `scoring` (default: True): Whether to run scoring phase

#### Step 4: Run the Evaluation

```bash
conda run -n sae python your_evaluation_script.py
```

#### Step 5: Monitor Progress and Check Results

The script will output progress information. Results are saved in the `Results/` folder:
- CSV summary: `<model>_layer<num>_features_summary_<timestamp>.csv`
- JSON results: `<sae_id>_eval_results.json`
- Logs: `autointerp_<model>_layer<num>_features<list>_<timestamp>.txt`

### Example: FinBERT Evaluation

Complete example for evaluating FinBERT features using [`run_finbert.py`](run_finbert.py):

**Configuration:**
```python
MODEL_NAME = "ProsusAI/finbert"
SAE_PATH = "/path/to/finbert/sae"
LAYER = 10
FEATURES_TO_EVALUATE = [0, 1, 2, 3, 4]
TOTAL_TOKENS = 500_000
CONTEXT_SIZE = 512  # FinBERT uses 512 context length
LLM_BATCH_SIZE = 32
LLM_DTYPE = "float32"  # FinBERT works better with float32
```

**Run:**
```bash
conda run -n sae python run_finbert.py
```

**Expected Output:**
```
📥 Loading FinBERT SAE...
✅ SAE loaded: 3072 features, K=24, activation_dim=768
Running AutoInterp for 5 features...
Collecting examples: 100%|██████████| 5/5 [00:23<00:00]
Calling API: 100%|██████████| 5/5 [00:02<00:00]
📊 Generating CSV summary...
✅ CSV saved to: Results/finbert_layer10_features_summary_20251122_184907.csv
```

**Results:**
- CSV: `finbert_layer10_features_summary_<timestamp>.csv`
- JSON: `finbert_layer10_features3072_k24_custom_sae_eval_results.json`
- Logs: `autointerp_finbert_layer10_features0_1_2_3_4_<timestamp>.txt`

**Sample Results:**
```csv
feature,label,autointerp_score
0,the end of text marker and company names or ticker symbols,0.5714
1,stock ticker symbols and company names in financial contexts,0.1000
2,end-of-text markers and punctuation,0.4286
3,stock and financial terms often found in investment analysis,0.4000
4,the substring '##' within words,0.0000
```

### Example: Nemotron Evaluation

Complete example for evaluating Nemotron features using [`run_nemotron_top100.py`](run_nemotron_top100.py):

**Configuration:**
```python
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE_CHECKPOINT_PATH = "/path/to/nemotron/sae/ae.pt"
SAE_CONFIG_PATH = "/path/to/nemotron/sae/config.json"
LAYER = 28
TOP_K_FEATURES = 100  # Extract top 100 finance features from summary file
TOTAL_TOKENS = 2_000_000
CONTEXT_SIZE = 128
LLM_BATCH_SIZE = 32
LLM_DTYPE = "bfloat16"
DATASET_NAME = "ashraq/financial-news"  # Financial dataset
```

**Run:**
```bash
conda run -n sae python run_nemotron_top100.py
```

**Expected Output:**
```
📊 Extracting top 100 finance features...
✅ Extracted 100 features
📥 Loading Nemotron SAE...
✅ SAE loaded: 35840 features, K=64, activation_dim=4480
Running AutoInterp for 100 features...
Collecting examples: 100%|██████████| 100/100 [15:23<00:00]
Calling API: 100%|██████████| 100/100 [02:15<00:00]
📊 Generating CSV summary...
✅ CSV saved to: Results/nemotron_layer28_features_summary_20251128_013011.csv
📊 Overall Score: 0.6705 ± 0.1234
```

**Results:**
- CSV: `nemotron_layer28_features_summary_<timestamp>.csv`
- JSON: `nvidia_nemotron_nano_9b_v2_layer28_k64_latents35840_custom_sae_eval_results.json`
- Logs: `autointerp_nvidia_nemotron_nano_9b_v2_layer28_top100_finance_<timestamp>.txt`

**Sample Results:**
```csv
feature,label,autointerp_score
2189,Earnings Reports and Financial Updates,0.7143
2330,CEO Earnings Updates,0.5714
2485,Stock market performance updates,0.7143
10628,Earnings Call Transcripts,0.9286
25313,Daily financial updates and stock movements,0.5714
```

## Example Results

### CSV Summary Examples

The tool automatically generates CSV files with feature explanations and scores. Here are real examples from the Results folder:

**FinBERT Layer 10 Results** (`finbert_layer10_features_summary_20251122_184907.csv`):

```csv
feature,label,autointerp_score
0,the end of text marker and company names or ticker symbols,0.5714
1,stock ticker symbols and company names in financial contexts,0.1000
2,end-of-text markers and punctuation,0.4286
3,stock and financial terms often found in investment analysis,0.4000
4,the substring '##' within words,0.0000
```

**Nemotron Layer 28 Results** (`nemotron_layer28_features_summary_20251128_013011.csv`):

```csv
feature,label,autointerp_score
2189,Earnings Reports and Financial Updates,0.7143
2330,CEO Earnings Updates,0.5714
2485,Stock market performance updates,0.7143
10628,Earnings Call Transcripts,0.9286
25313,Daily financial updates and stock movements,0.5714
```

**Interpreting Scores:**
- **0.85+**: Excellent - The explanation accurately predicts when the feature activates
- **0.70-0.85**: Good - The explanation is mostly reliable
- **0.50-0.70**: Moderate - The explanation has some predictive power but may be incomplete
- **<0.50**: Poor - The explanation doesn't reliably predict activations (may indicate noisy feature or unclear pattern)

### Top Activating Examples

The system collects top-activating text sequences for each feature. These are sent to the LLM to generate explanations. Activating tokens are marked with `<< >>`. These examples help understand why explanations are generated and reveal if they're too generic or specific.

**Example 1: Feature 25313 (Score: 1.0000) - "the number 13 in various contexts including dates and percentages"**

**Top Activating Examples:**
1. **Activation 4.500:** `Pharma's special meeting set for November <<13>> to approve shares`
2. **Activation 4.125:** `Top 2 Trade Alert Ideas October <<13>>: Mast Therapeutics' Potential`
3. **Activation 3.922:** `Vs. Dividends Between <<13>> Top Dividend Aristocrat Survivors`
4. **Activation 3.641:** `shares slump <<13>>% in early trading`
5. **Activation 3.547:** `Daily Round Up 10/<<13>>/16: Ruby Tuesday`

**Analysis:** All examples contain "13" in dates or percentages. This is a good explanation - specific enough to identify the pattern but general enough to cover all contexts.

**Example 2: Feature 18529 (Score: 0.7857) - "words related to energy materials and financial performance"**

**Top Activating Examples:**
1. **Activation 8.875:** `: AGCO, HSBC Holdings plc, The<< Wendy>>'s, Krispy Kreme Doughnuts and McDonald`
2. **Activation 8.750:** `. of Extension of Time for Compliance and That a<< Reverse>> Stock Split<< Would>> be AppropriateFutures, Dow`
3. **Activation 8.250:** `, SpartanNash Company, The Hain Cele<<stial>> Group, DHT, Sportsman's Warehouse and`
4. **Activation 8.062:** `featured highlights include: Cadence Design System, L<<PL>> Financial, CSX and U.S. Physical Therapy`
5. **Activation 7.750:** `Stock Jack in the Box (JACK), The<< Wendy>>'s Company (WEN) or Sonic Corporation (`

**Analysis:** Most examples show company names (AGCO, HSBC, Wendy's, etc.). Only one example mentions "Energy Business Model". The explanation is too generic - most examples are just company names, not specifically about energy.

**Example 3: Feature 6105 (Score: 0.5714) - "words related to business and finance terms"**

**Analysis:** This explanation is too generic. "Business and finance terms" could describe almost any financial text. More specific pattern identification is needed.

**Key Observations:**
- Good explanations are specific enough to identify patterns but general enough to cover most examples
- Generic explanations (like "business terms") don't help distinguish features
- Overly specific explanations (based on 1-2 examples) don't represent the feature well
- The balance: find the MOST COMMON pattern that appears in AT LEAST 5-10 examples

**Debugging Tips:**
To debug why explanations are generic or too specific, check the top activating examples in the JSON file. Look for:
- Patterns that appear in 5+ examples (good - should be in explanation)
- Patterns that appear in only 1-2 examples (too specific - should be ignored)
- Very diverse examples with no common pattern (may indicate noisy feature)
- Examples with very different activation strengths (higher activation = more important)

**Full Example Set for Feature 18529 (for debugging):**

**Top 15 Activating Examples:**
1. **Activation 8.875:** `: AGCO, HSBC Holdings plc, The<< Wendy>>'s, Krispy Kreme Doughnuts and McDonald`
2. **Activation 8.750:** `. of Extension of Time for Compliance and That a<< Reverse>> Stock Split<< Would>> be AppropriateFutures, Dow`
3. **Activation 8.750:** `of Time for Compliance and That a<< Reverse>> Stock Split<< Would>> be AppropriateFutures, Dow Jones Today Edge`
4. **Activation 8.250:** `, SpartanNash Company, The Hain Cele<<stial>> Group, DHT, Sportsman's Warehouse and`
5. **Activation 8.062:** `featured highlights include: Cadence Design System, L<<PL>> Financial, CSX and U.S. Physical Therapy`
6. **Activation 7.750:** `Stock Jack in the Box (JACK), The<< Wendy>>'s Company (WEN) or Sonic Corporation (`
7. **Activation 7.625:** `And<< Oil>> Inventories (Not<< As>> Clear-Cut<< As>> You May Think It Is)California Resources and Ultra`
8. **Activation 7.500:** `Earnings Call TranscriptHollySys Automation Technologies,<< Ltd>>. (H<<OL>>I) CEO Baiqing Sh`
9. **Activation 7.375:** `A PauseEuropean Implosion Sends Panic Through Global Markets<< As>> George Soros Warns 'We May Be Heading For`
10. **Activation 7.375:** `Analyst BlogYour Daily Pharma Scoop: Dynavax<< Achie>>ves Major Milestone,<< GW>> Pharmaceuticals GWP420`
11. **Activation 7.281:** `<< As>> Questions Arise About Europe Approving Latest M<<erg>>ersHow Much Does It Cost To Produce One Barrel`
12. **Activation 5.656:** `ed This Week - That's The Downside Of<< Le>>verage Though There May Be Opportunity HereAnalysts Estimate`
13. **Activation 4.594:** `The Evolving Energy Business Model: A Transformational<< Change>> From 'Drill-Baby-Drill'`
14. **Activation 3.109:** `13: Mast Therapeutics' Potential,<< Johnson>> &<< Johnson>>'s Results, Investor RelationsRadar Signals: RR`
15. **Activation 2.438:** `ights.com Daily Round Up 6/24/<<15>>: Groupon, Allstate, La Jolla`

**Pattern Analysis:** Most examples (1-12, 14-15) show company names. Only example #13 mentions "Energy Business Model". The explanation "energy materials and financial performance" is too generic because most examples are just company names, not specifically about energy.

**Full Example Set for Feature 25313 (for debugging):**

**Top 15 Activating Examples:**
1. **Activation 4.500:** `Pharma's special meeting of shareholders set for November <<13>> to approve additional shares for Depomed bidPremark`
2. **Activation 4.125:** `2Top 2 Trade Alert Ideas October <<13>>: Mast Therapeutics' Potential, Johnson & Johnson`
3. **Activation 3.922:** `Vs. Dividends Smack Down Between <<13>> Top Dividend Aristocrat Survivors Will Wake You`
4. **Activation 3.891:** `New York MellonBerkshire's Revealing <<13>>F: Buffett Didn't Buy The Dip - Didn`
5. **Activation 3.641:** `lung infectionTop 2 Trade Alert Ideas October <<13>>: Mast Therapeutics' Potential, Johnson & Johnson`
6. **Activation 3.641:** `Pharma despite positive early-state data; shares slump <<13>>% in early tradingInsiderInsights.com Daily`
7. **Activation 3.547:** `iderInsights.com Daily Round Up 10/<<13>>/16: Ruby Tuesday, Vishay Precision,`
8. **Activation 3.453:** `iderInsights.com Daily Round Up 1/<<13>>/16: Tuesday Morning, Conn's, Barnes`
9. **Activation 3.344:** `treatment of rare form of epilepsy; shares up <<13>>% premarketBiotech Forum Daily Digest: Another`
10. **Activation 3.312:** `iderInsights.com Daily Round Up 10/<<13>>/16: Ruby Tuesday, Vishay Precision,`
11. **Activation 3.297:** `Affimed: Positive Takeaways From AFM-<<13>> Focused R&D Day; Multiple Catalysts Ahead`
12. **Activation 3.016:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`
13. **Activation 2.891:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`
14. **Activation 2.484:** `orenal diseases in Japan; ARDX up <<13>>% premarketArdelyx prepares to launch`
15. **Activation 2.188:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`

**Pattern Analysis:** ALL examples contain "13" in dates (October 13, 10/13/16, 7/13/15) or percentages (13%). This is a perfect example - the pattern is clear and consistent across all examples, making the explanation accurate and useful.

### Where to Find Detailed Results

All evaluation outputs are saved in the `Results/` folder. Here's where to find different types of information:

**1. CSV Summary** (`*_features_summary_*.csv`):
- **Location**: `Results/<model>_layer<num>_features_summary_<timestamp>.csv`
- **Contains**: Quick overview with feature ID, label, and score
- **Use**: Fast lookup of explanations and scores

**2. JSON Results** (`*_eval_results.json`):
- **Location**: `Results/<sae_id>_eval_results.json`
- **Example**: `Results/nvidia_nemotron_nano_9b_v2_layer28_k64_latents35840_custom_sae_eval_results.json`
- **Contains**: Full logs for ALL features (including "Top act" tables), per-feature details, configuration, aggregate metrics
- **Use**: Extract top activating examples for any feature, analyze detailed results
- **Structure**: The `eval_result_unstructured` field contains `logs` for each feature with generation/scoring details and "Top act" tables.

**3. Log Files** (`autointerp_*.txt`):
- **Location**: `Results/autointerp_<model>_layer<num>_features<list>_<timestamp>.txt`
- **Contains**: Summary table of all features, detailed logs for BEST and WORST features only
- **Use**: Quick inspection of best/worst cases

**4. Artifacts** (`artifacts_*/`):
- **Location**: `Results/artifacts_<model>_layer<num>_<timestamp>/`
- **Contains**: Cached tokenized datasets (`.pt` files)
- **Use**: Faster reruns when `FORCE_RERUN = False`

**Extracting Top Activating Examples from JSON:**

```python
import json

with open('Results/nvidia_nemotron_nano_9b_v2_layer28_k64_latents35840_custom_sae_eval_results.json') as f:
    data = json.load(f)

feature_id = "25313"
logs = data['eval_result_unstructured'][feature_id]['logs']
# Logs contain "Top act" table with activation values and sequences
```

**Note**: Log files (`.txt`) only show best/worst features. For all features' examples, use the JSON file.

## Method

AutoInterp Eval proceeds in three stages:

### 1. Collecting Activation Evidence

For each feature, the system samples text sequences, computes SAE activations, and selects strong (top-k), medium (importance-weighted), and low (random) activation examples. These reveal which patterns cause the feature to fire. Domain knowledge is not assumed; the domain emerges from examples.

### 2. Natural-Language Explanation via LLM: The Prompt

The collected examples are passed to a large language model using a structured, domain-neutral prompt. The exact prompts used in the code are:

**Generation Phase Prompt:**

**System:**
```
We are labeling sparse autoencoder features. Each feature fires on specific words, substrings, or concepts marked with << >>.

You will receive several activating documents in descending activation strength. Identify the SINGLE MOST COMMON activating pattern present in AT LEAST 5–10 examples.

Label requirements:
- Output ONLY a short label (not a sentence), ideally under 10 words.
- Increase specificity when the same type of entity, category, or phrase appears repeatedly across examples.
- If a specific entity/word appears ≥3 times, include it; otherwise keep it category-level.
- Avoid vague labels like "business terms" or "finance-related language."
- Avoid overly specific labels like headlines, article titles, or one-off phrases.
- The label must be balanced: specific enough to reflect the repeated pattern, but general enough to cover MOST examples.
- No punctuation, no lists, no filler words.
```

**User:**
```
Below are the activating documents. Identify the repeated activating pattern following the system rules:

1. <<document with marked activating tokens>>
2. <<document with marked activating tokens>>
...
```

The prompt is domain-adaptive: financial datasets produce financial examples; sports datasets produce sports examples. The LLM infers the domain from examples.

### 3. Evaluation and Scoring

The evaluation uses a two-phase approach:

**Generation Phase**: Collects top-activating sequences and importance-weighted samples from SAE activations, formats tokens with `<<token>>` syntax where features activate, and asks the LLM: "What does this feature activate on?" to get an explanation.

**Scoring Phase**: Uses the following prompt to test how well the explanation predicts activations:

**Scoring Phase Prompt:**

**System:**
```
We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. You will be given a short explanation of what this neuron activates for, and then be shown {n_ex_for_scoring} example sequences in random order. You will have to return a comma-separated list of the examples where you think the neuron should activate at least once, on ANY of the words or substrings in the document. For example, your response might look like "1, 3, 5". Try not to be overly specific in your interpretation of the explanation. If you think there are no examples where the neuron will activate, you should just respond with "None". You should include nothing else in your response other than comma-separated numbers or the word "None" - this is important.
```

**User:**
```
Here is the explanation: this neuron fires on {explanation}.

Here are the examples:

1. <sequence without marked tokens>
2. <sequence without marked tokens>
...
```

The scoring phase creates a shuffled mix of sequences. It asks the LLM: "Which sequences will activate this feature?" Compares predictions to actual activations to calculate the AutoInterp score.

The AutoInterp score is the fraction of sequences where the LLM correctly identified activation. Computed as: `correct_classifications / total_classifications`.

## Files

```
autointerp_saeeval/
├── autointerp/              # AutoInterp evaluation module
│   ├── eval_config.py      # Configuration dataclass
│   ├── eval_output.py      # Output schema and metrics
│   ├── main.py             # Core evaluation logic (prompts, API calls, scoring)
│   ├── sae_encode.py       # SAE encoding utilities
│   └── eval_output_schema_autointerp.json  # JSON schema
├── openai_api_key.txt      # API key (for OpenAI provider, if needed)
├── run_llama.py  # Llama example script
├── run_finbert.py  # FinBERT example with vLLM
├── run_nemotron_top100.py  # Nemotron top 100 finance features
├── run_nemotron_finance.py  # Nemotron finance features (alternative)
├── run_nemotron_vllm.py  # Nemotron with vLLM (alternative)
├── start_vllm_server_72b.sh  # Script to start vLLM server
├── Results/                 # All outputs
│   ├── *.csv               # CSV summaries
│   ├── *.json              # Full results JSON
│   ├── *.txt               # Detailed logs
│   └── artifacts_*/       # Cached tokenized datasets
└── README.md
```

## Dependencies

- Python: `torch`, `safetensors`, `transformer_lens`, `sae_lens`, `requests`
- SAEBench: Only for `TopKSAE` and `CustomSAEConfig` (minimal)
- vLLM server running (no API key needed)

## Relevant Files

### Core Implementation
- [`autointerp/main.py`](autointerp/main.py) - Main evaluation logic, prompts, API calls, scoring
- [`autointerp/eval_config.py`](autointerp/eval_config.py) - Configuration dataclass with all parameters
- [`autointerp/eval_output.py`](autointerp/eval_output.py) - Output schema and metrics definitions

### Example Scripts
- [`run_finbert.py`](run_finbert.py) - Complete example for FinBERT with vLLM, includes CSV generation
- [`run_nemotron_top100.py`](run_nemotron_top100.py) - Complete example for Nemotron top 100 finance features with vLLM
- [`run_llama.py`](run_llama.py) - Basic example script for Llama
- [`run_nemotron_finance.py`](run_nemotron_finance.py) - Nemotron finance features evaluation
- [`run_nemotron_vllm.py`](run_nemotron_vllm.py) - Nemotron with vLLM (alternative version)

### Configuration and Utilities
- [`autointerp/sae_encode.py`](autointerp/sae_encode.py) - SAE encoding utilities
- [`start_vllm_server_72b.sh`](start_vllm_server_72b.sh) - Script to start vLLM server

### Results and Documentation
- [`Results/`](Results/) - All evaluation outputs (CSV summaries, JSON results, logs, artifacts)
- [`Results/LOW_SCORES_EXPLANATION.md`](Results/LOW_SCORES_EXPLANATION.md) - Guide for improving low AutoInterp scores

### Key Code Locations
- **Generation Prompt**: Lines 389-409 in [`autointerp/main.py`](autointerp/main.py)
- **Scoring Prompt**: Lines 405-430 in [`autointerp/main.py`](autointerp/main.py)
- **Score Calculation**: Lines 306-315 in [`autointerp/main.py`](autointerp/main.py)
- **vLLM API Integration**: Lines 317-377 in [`autointerp/main.py`](autointerp/main.py)
- **Data Collection**: Lines 432-550 in [`autointerp/main.py`](autointerp/main.py)

## Results Location

All outputs are saved to the `Results/` folder in the `autointerp_saeeval` directory. The folder structure is:

```
Results/
├── autointerp_<model>_layer<num>_features<list>_<timestamp>.txt  # Detailed logs
├── <sae_id>_eval_results.json                                    # Full results JSON
├── <model>_layer<num>_features_summary_<timestamp>.csv          # CSV summary
└── artifacts_<model>_layer<num>_<timestamp>/
    └── autointerp/
        └── <model>_<tokens>_tokens_<ctx>_ctx.pt                # Cached tokenized dataset
```

**File Types:**

1. **CSV Summary** (`*_features_summary_*.csv`): Quick overview with columns:
   - `feature`: Feature index
   - `label`: Generated explanation
   - `autointerp_score`: Score for this feature

2. **JSON Results** (`*_eval_results.json`): Complete evaluation data including:
   - Aggregate metrics (mean score, std dev)
   - Per-feature details (explanations, scores, predictions)
   - Configuration used
   - **Full logs for ALL features** (including "Top act" tables with activation values and sequences)

3. **Log Files** (`autointerp_*.txt`): Detailed logs showing:
   - Summary table of all features
   - Best and worst scoring features with full prompts and responses
   - Generation and scoring phase details

4. **Artifacts** (`artifacts_*/`): Cached data for faster reruns:
   - Tokenized datasets (saved as `.pt` files)
   - Reused if `FORCE_RERUN = False` and same configuration

**Example File Names:**
- `finbert_layer10_features_summary_20251122_184907.csv`
- `nemotron_layer28_features35840_k64_custom_sae_eval_results.json`
- `autointerp_finbert_layer10_features0_1_2_3_4_20251122_184907.txt`

All files are timestamped to avoid overwriting previous runs.

## Why AutoInterp Eval Is Model-Agnostic

AutoInterp Eval works with any LLM architecture. It only needs a forward pass, access to hidden state vectors at a specific layer, and the ability to feed those into the SAE. Every modern transformer (Nemotron, Llama, FinBERT, etc.) exposes hidden-state vectors. AutoInterp doesn't depend on attention details, routing, head counts, normalization, or tokenizer quirks. It only needs: "Give me the hidden vectors so I can run the SAE on them."

## Comparison With Delphi

Delphi is another LLM-based interpretability technique that also explains features using natural language, but the design philosophy differs.

**AutoInterp Eval (SAEBench Style)**

What it does: Uses only activation examples, produces a simple label, description, and AutoInterp score. Light-weight, fast, easy to run. Minimal assumptions about the model. No extra metrics or verification.

Strengths: Extremely simple, domain-agnostic, architecture-agnostic, ideal for large sweeps across thousands of features.

Limitations: Score depends on LLM's ability to predict activations, not verified against ground truth. No built-in quality metrics beyond the AutoInterp score. May produce unclear labels for noisy features.
