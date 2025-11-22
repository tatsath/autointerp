# AutoInterp SAE Evaluation

Standalone tool for automatically producing human-readable explanations for SAE features using AutoInterp methodology. This tool evaluates how well an LLM can explain and predict feature activations, providing interpretability scores for sparse autoencoder features.

## Table of Contents

- [Overview](#overview)
- [Method](#method)
  - [Collecting Activation Evidence](#1-collecting-activation-evidence)
  - [Natural-Language Explanation via LLM](#2-natural-language-explanation-via-llm-the-prompt)
  - [Evaluation and Scoring](#3-evaluation-and-scoring)
- [What AutoInterp Eval Returns](#what-autointerp-eval-returns)
- [Why AutoInterp Eval Is Model-Agnostic](#why-autointerp-eval-is-model-agnostic)
- [Usage](#usage)
  - [How to Run Evaluation for Any LLM](#how-to-run-evaluation-for-any-llm)
  - [Example: FinBERT Evaluation](#example-finbert-evaluation)
  - [Example: Nemotron Evaluation](#example-nemotron-evaluation)
- [Example Results](#example-results)
  - [CSV Summary Examples](#csv-summary-examples)
  - [JSON Results Structure](#json-results-structure)
- [Files](#files)
- [Dependencies](#dependencies)
- [Relevant Files](#relevant-files)
- [Results Location](#results-location)
- [Comparison With Delphi](#comparison-with-delphi)

## Overview

Auto-Interpretability Evaluation (AutoInterp Eval) is a method for automatically producing human-readable explanations for the latent features learned by a Sparse Autoencoder (SAE). Each SAE feature corresponds to a direction in the model's hidden-state space, and AutoInterp Eval provides a natural-language label describing what semantic or structural pattern that feature represents.

The key idea is simple: observe where a feature activates strongly, send those examples to an LLM, and let the LLM explain the pattern. AutoInterp Eval is domain-agnostic and model-agnostic. It operates purely on hidden activations from any LLM (Nemotron, GPT-OSS, Llama, Gemma, FinBERT, etc.), SAE feature activations, and text examples extracted from the target dataset. This makes the method applicable across finance, sports, legal text, multi-domain corpora, or any domain where SAE features are learned.

## Method

AutoInterp Eval proceeds in three stages:

### 1. Collecting Activation Evidence

For each feature F of the SAE, the system samples thousands of text sequences from a dataset, computes SAE activations for each sequence, and selects strong activation examples (top-k contexts), medium activation examples (importance-weighted samples), and low/near-zero examples (random sequences for contrast). These examples reveal which patterns cause the feature to fire and which do not. AutoInterp does not assume any domain knowledge; the domain naturally emerges from the examples extracted.

### 2. Natural-Language Explanation via LLM: The Prompt

The collected examples are passed to a large language model using a structured, domain-neutral prompt. The exact prompts used in the code are:

**Generation Phase Prompt:**

**System:**
```
We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.
```

If `use_demos_in_explanation=True` (default), the system prompt also includes:
```
Some examples: "This neuron activates on the word 'knows' in rhetorical questions", and "This neuron activates on verbs related to decision-making and preferences", and "This neuron activates on the substring 'Ent' at the start of words", and "This neuron activates on text about government economic policy".
```

Otherwise, it adds:
```
Your response should be in the form "This neuron activates on...".
```

**User:**
```
The activating documents are given below:

1. <<document with marked activating tokens>>
2. <<document with marked activating tokens>>
...
```

This prompt is domain-adaptive because it presents only the activation contexts. If the dataset is financial, the strong examples will contain financial text; if sports, then sports text; the LLM infers the domain without being told.

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

The scoring phase creates a shuffled mix of top sequences, random sequences, and importance-weighted samples. It gives the LLM the explanation and asks: "Which numbered sequences will activate this feature?" Compares LLM predictions to actual activations to calculate the AutoInterp score.

The AutoInterp score represents the fraction of sequences where the LLM correctly identified whether the feature would activate, measuring how well the generated explanation predicts actual feature behavior. The score is computed as: `correct_classifications / total_classifications`, where each sequence is classified as activating or not.

## What AutoInterp Eval Returns

AutoInterp Eval provides exactly four outputs for each feature:

**Label/Explanation**: A short descriptive phrase indicating the feature's interpretation (parsed from the LLM response, typically the part after "activates on").

**Score**: The AutoInterp score (0 to 1) representing the fraction of sequences where the LLM correctly predicted activation. This is computed by comparing LLM predictions to actual activations in the scoring phase.

**Predictions**: The list of sequence indices that the LLM predicted would activate.

**Correct Sequences**: The list of sequence indices that actually activated.

Note: The code does not use a separate "confidence score" or "unknown flag" as described in some paper versions. The confidence is implicitly measured by the AutoInterp score—if the explanation is good, the LLM will correctly predict activations more often, resulting in a higher score.

## Why AutoInterp Eval Is Model-Agnostic

AutoInterp Eval works with any LLM architecture because it only needs a way to run a forward pass, access the hidden state vector at a specific layer, and feed that hidden state into the SAE. Every modern transformer (Nemotron, GPT-NeoX, Llama, Gemma, Mixtral, Falcon, Qwen, FinBERT) exposes a hidden-state vector of size d_model and per-layer outputs. Therefore AutoInterp Eval does not depend on attention mechanism details, routing (MoE), head counts, normalization type, positional embedding type, training data domain, or tokenizer quirks. It only needs: "Give me the hidden vectors so I can run the SAE on them."

## Usage

**Note**: The current implementation only supports the `"vllm"` provider. OpenAI API support is not implemented in the current codebase.

### How to Run Evaluation for Any LLM

This section provides general instructions for running AutoInterp evaluation on any model. The process is model-agnostic and follows the same steps regardless of the architecture.

#### Step 1: Start the vLLM Server

First, start a vLLM server with your chosen explainer model (the LLM that will generate explanations):

```bash
# Example: Start Qwen 2.5 72B
bash start_vllm_server_72b.sh

# Or start your own vLLM server:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --port 8002 \
    --tensor-parallel-size 1
```

The server should be accessible at `http://localhost:8002/v1` (or your configured port). Wait until the server is fully loaded before proceeding.

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

#### Step 3: Understanding and Choosing Hyperparameters

The following hyperparameters control the evaluation quality and speed. Choose them based on your goals:

**Data Collection Hyperparameters:**

- **`total_tokens`** (default: 2M, recommended: 100k-1M for testing, 2M+ for production)
  - More tokens = more diverse examples = better explanations
  - Trade-off: Higher values take longer to process
  - For quick testing: 100k-500k
  - For publication-quality results: 1M-2M

- **`llm_context_size`** (default: 512 or 1024)
  - Should match your model's context length
  - Longer contexts capture more context but use more memory
  - Common values: 512 (BERT), 1024 (many models), 2048+ (larger models)

- **`llm_batch_size`** (default: varies by model)
  - Larger batches = faster processing but more GPU memory
  - Start with 16, increase if you have memory headroom
  - Typical range: 8-32

**Example Selection Hyperparameters:**

- **`n_top_ex_for_generation`** (default: 10, range: 5-20)
  - Number of strongest activation examples shown to LLM for explanation
  - More examples = potentially better explanations but longer prompts
  - Recommended: 10-15 for most cases

- **`n_iw_sampled_ex_for_generation`** (default: 5, range: 3-10)
  - Importance-weighted samples (medium activations) for generation
  - Helps capture patterns beyond just top activations
  - Recommended: 5-7

- **`n_top_ex_for_scoring`** (default: 2, range: 2-5)
  - Top examples included in scoring phase
  - Lower values make scoring faster
  - Recommended: 2-3

- **`n_random_ex_for_scoring`** (default: 10, range: 5-20)
  - Random examples for contrast in scoring phase
  - More examples = more reliable score but slower
  - Recommended: 10-15

- **`n_iw_sampled_ex_for_scoring`** (default: 2, range: 2-5)
  - Importance-weighted examples for scoring
  - Recommended: 2-3

**Activation Threshold Hyperparameters:**

- **`act_threshold_frac`** (default: 0.01, range: 0.001-0.1)
  - Fraction of max activation used as threshold for marking tokens
  - Lower values = more tokens marked as activating (more permissive)
  - Higher values = only strongest activations marked (more strict)
  - Recommended: 0.01 for most cases, 0.001 for sparse features

- **`dead_latent_threshold`** (default: -1.0)
  - Features with sparsity below this are considered "dead" and skipped
  - Negative values allow all features to be evaluated
  - Set to 0.0 or higher to filter truly dead features

**Explanation Quality Hyperparameters:**

- **`max_tokens_in_explanation`** (default: 30, range: 20-100)
  - Maximum tokens allowed in generated explanation
  - Longer explanations can be more detailed but may be less focused
  - Recommended: 30-40 for concise labels, 60-80 for detailed descriptions

- **`use_demos_in_explanation`** (default: True)
  - Whether to include example explanations in the prompt
  - Helps guide LLM to produce better formatted explanations
  - Recommended: True for most cases

**Scoring Hyperparameters:**

- **`scoring`** (default: True)
  - Whether to run the scoring phase (evaluation of explanation quality)
  - Set to False to only generate explanations without scoring
  - Recommended: True for evaluation, False for quick label generation

**Diagnostic Recommendations:**

- **For quick testing (5-10 features):**
  - `total_tokens = 100_000`
  - `n_top_ex_for_generation = 10`
  - `n_random_ex_for_scoring = 10`
  - `max_tokens_in_explanation = 30`

- **For production evaluation (many features):**
  - `total_tokens = 500_000` to `1_000_000`
  - `n_top_ex_for_generation = 15`
  - `n_random_ex_for_scoring = 15`
  - `max_tokens_in_explanation = 40`

- **For noisy or unclear features:**
  - `act_threshold_frac = 0.001` (more permissive)
  - `n_top_ex_for_generation = 20` (more examples)
  - `max_tokens_in_explanation = 50` (allow longer explanations)

- **For very sparse features:**
  - `dead_latent_threshold = -1.0` (evaluate all features)
  - `act_threshold_frac = 0.001` (lower threshold)

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

Here's a complete example for evaluating FinBERT features using [`run_autointerp_features_vllm_finbert.py`](run_autointerp_features_vllm_finbert.py):

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
conda run -n sae python run_autointerp_features_vllm_finbert.py
```

**Expected Output:**
```
🔍 Checking vLLM server at http://localhost:8002/v1...
✅ vLLM server is running
📥 Loading FinBERT SAE...
✅ SAE loaded: 3072 features, K=24, activation_dim=768
Running AutoInterp for 5 features...
Collecting examples for LLM judge: 100%|██████████| 5/5 [00:23<00:00]
Calling API (for gen & scoring): 100%|██████████| 5/5 [00:02<00:00]
📊 Generating CSV summary...
✅ CSV saved to: Results/finbert_layer10_features_summary_20251122_184907.csv
```

**Results:**
- CSV: `finbert_layer10_features_summary_<timestamp>.csv`
- JSON: `finbert_layer10_features3072_k24_custom_sae_eval_results.json`
- Logs: `autointerp_finbert_layer10_features0_1_2_3_4_<timestamp>.txt`

### Example: Nemotron Evaluation

Here's a complete example for evaluating Nemotron features using [`run_nemotron_autointerp_vllm.py`](run_nemotron_autointerp_vllm.py):

**Configuration:**
```python
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE_PATH = Path("/path/to/nemotron/sae")
LAYER = 28
TOTAL_TOKENS = 500_000
CONTEXT_SIZE = 1024
LLM_BATCH_SIZE = 16
LLM_DTYPE = "bfloat16"
# Features are extracted from summary files automatically
```

**Run:**
```bash
conda run -n sae python run_nemotron_autointerp_vllm.py
```

**Expected Output:**
```
🔧 Patching transformer_lens for Nemotron support...
✅ Nemotron patches applied successfully
📊 Extracting features from summary files...
✅ Extracted 5 finance features
📥 Loading Nemotron SAE...
✅ SAE loaded: 35840 features, K=64, activation_dim=4480
Running AutoInterp for finance features: 5 features
  Finance Features Score: 0.6429 ± 0.1750
📊 Generating CSV summary...
✅ CSV saved to: Results/nemotron_layer28_features_summary_20251122_191954.csv
```

**Results:**
- CSV: `nemotron_layer28_features_summary_<timestamp>.csv`
- JSON: `nemotron_layer28_features35840_k64_custom_sae_eval_results.json`
- Logs: `autointerp_nvidia_nemotron_nano_9b_v2_layer28_finance_5features_<timestamp>.txt`

## Example Results

### CSV Summary Examples

The tool automatically generates CSV files with feature explanations and scores. Here are real examples from the Results folder:

**FinBERT Layer 10 Results** (`finbert_layer10_features_summary_20251122_184907.csv`):

```csv
layer,feature,label,autointerp_score
10,0,the end of text marker and company names or ticker symbols,0.5714
10,1,stock ticker symbols and company names in financial contexts,0.1000
10,2,end-of-text markers and punctuation,0.4286
10,3,stock and financial terms often found in investment analysis,0.4000
10,4,the substring '##' within words,0.0000
```

**Nemotron Layer 28 Results** (`nemotron_layer28_features_summary_20251122_191954.csv`):

```csv
layer,feature,label,autointerp_score
28,2216,words indicating financial market movements and corporate actions,0.7143
28,6105,financial and business-related terms and symbols,0.5000
28,8982,the word 'healthcare' and similar terms like 'Cosmetics' in business contexts,0.8571
28,18529,company names and stock-related terms,0.4286
28,19903,words indicating action or decision-making such as selling ordering building and going,0.7143
```

**Interpreting Scores:**
- **0.85+**: Excellent - The explanation accurately predicts when the feature activates
- **0.70-0.85**: Good - The explanation is mostly reliable
- **0.50-0.70**: Moderate - The explanation has some predictive power but may be incomplete
- **<0.50**: Poor - The explanation doesn't reliably predict activations (may indicate noisy feature or unclear pattern)

### JSON Results Structure

The full JSON results file contains detailed information for each feature:

```json
{
  "eval_result_metrics": {
    "autointerp": {
      "autointerp_score": 0.6429,
      "autointerp_std_dev": 0.1750
    }
  },
  "eval_result_unstructured": {
    "18529": {
      "explanation": "company names and stock-related terms",
      "score": 0.4286,
      "predictions": [2, 5, 8],
      "correct seqs": [2, 5, 9]
    },
    "6105": {
      "explanation": "financial and business-related terms and symbols",
      "score": 0.5000,
      "predictions": [1, 3, 7],
      "correct seqs": [1, 3, 7]
    }
  },
  "eval_config": {
    "model_name": "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
    "override_latents": [18529, 6105, 8982, 2216, 19903],
    "total_tokens": 500000,
    "llm_context_size": 1024
  }
}
```

**Key Fields:**
- `autointerp_score`: Mean score across all evaluated features
- `autointerp_std_dev`: Standard deviation of scores
- `explanation`: The generated label for the feature
- `score`: Individual feature's AutoInterp score
- `predictions`: Sequence indices the LLM predicted would activate
- `correct seqs`: Sequence indices that actually activated

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
├── run_autointerp_features.py  # Main script (OpenAI)
├── run_autointerp_features_vllm_finbert.py  # FinBERT example with vLLM
├── run_nemotron_autointerp_vllm.py  # Nemotron example with vLLM
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
- For vLLM provider: vLLM server running (no API key needed, but can be provided optionally)

## Relevant Files

### Core Implementation
- [`autointerp/main.py`](autointerp/main.py) - Main evaluation logic, prompts, API calls, scoring
- [`autointerp/eval_config.py`](autointerp/eval_config.py) - Configuration dataclass with all parameters
- [`autointerp/eval_output.py`](autointerp/eval_output.py) - Output schema and metrics definitions

### Example Scripts
- [`run_autointerp_features_vllm_finbert.py`](run_autointerp_features_vllm_finbert.py) - Complete example for FinBERT with vLLM, includes CSV generation
- [`run_nemotron_autointerp_vllm.py`](run_nemotron_autointerp_vllm.py) - Complete example for Nemotron with vLLM, includes CSV generation
- [`run_autointerp_features.py`](run_autointerp_features.py) - Basic example script

### Configuration and Utilities
- [`autointerp/sae_encode.py`](autointerp/sae_encode.py) - SAE encoding utilities
- [`start_vllm_server_72b.sh`](start_vllm_server_72b.sh) - Script to start vLLM server

### Results and Documentation
- [`Results/`](Results/) - All evaluation outputs (CSV summaries, JSON results, logs, artifacts)
- [`Results/LOW_SCORES_EXPLANATION.md`](Results/LOW_SCORES_EXPLANATION.md) - Guide for improving low AutoInterp scores

### Key Code Locations
- **Generation Prompt**: Lines 389-403 in [`autointerp/main.py`](autointerp/main.py)
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
   - `layer`: Model layer number
   - `feature`: Feature index
   - `label`: Generated explanation
   - `autointerp_score`: Score for this feature

2. **JSON Results** (`*_eval_results.json`): Complete evaluation data including:
   - Aggregate metrics (mean score, std dev)
   - Per-feature details (explanations, scores, predictions)
   - Configuration used

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

## Comparison With Delphi

Delphi is another LLM-based interpretability technique that also explains features using natural language, but the design philosophy differs.

**AutoInterp Eval (SAEBench Style)**

What it does: Uses only activation examples, produces a simple label, description, and AutoInterp score. Light-weight, fast, easy to run. Minimal assumptions about the model. No extra metrics or verification.

Strengths: Extremely simple, domain-agnostic, architecture-agnostic, ideal for large sweeps across thousands of features.

Limitations: Score depends on LLM's ability to predict activations, not verified against ground truth. No built-in quality metrics beyond the AutoInterp score. May produce unclear labels for noisy features.
