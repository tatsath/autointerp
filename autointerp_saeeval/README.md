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
  - [Using OpenAI API](#using-openai-api)
  - [Using vLLM (Faster Inference)](#using-vllm-faster-inference)
- [Results Location](#results-location)
- [CSV Summary](#csv-summary)
- [Files](#files)
- [Comparison With Delphi](#comparison-with-delphi)
- [Dependencies](#dependencies)
- [Relevant Files](#relevant-files)

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

### Using vLLM (Recommended)

For faster inference with local models, you can use vLLM instead of OpenAI API:

1. Start the vLLM server (e.g., Qwen 72B):
   ```bash
   bash start_vllm_server_72b.sh
   ```
   This typically starts a server at `http://localhost:8002/v1`.

2. Edit the run script:
   - `PROVIDER`: Set to `"vllm"` (this is the only provider currently supported in the code)
   - `EXPLAINER_MODEL`: Model name (e.g., `"Qwen/Qwen2.5-72B-Instruct"`)
   - `EXPLAINER_API_BASE_URL`: vLLM server URL (e.g., `"http://localhost:8002/v1"`)
   - `API_KEY_PATH`: Optional (vLLM doesn't require authentication by default)

3. Run:
   ```bash
   # For FinBERT example
   conda run -n sae python run_autointerp_features_vllm_finbert.py
   
   # For Nemotron example
   conda run -n sae python run_nemotron_autointerp_vllm.py
   ```

The vLLM provider offers faster inference for large models and doesn't require API keys or rate limits. It's ideal for running evaluations on many features or with large explainer models. The code uses vLLM's OpenAI-compatible API endpoint, making it easy to switch between different local models.

**Configuration Options:**

All scripts support the following key configuration parameters (see [`autointerp/eval_config.py`](autointerp/eval_config.py) for full list):

- `total_tokens`: Number of tokens to sample from dataset (default: 2M, but scripts often use 100k-500k for testing)
- `llm_batch_size`: Batch size for forward passes (default: varies by model)
- `llm_context_size`: Context length (default: 512 or 1024 depending on model)
- `n_top_ex_for_generation`: Number of top-activating examples for generation (default: 10)
- `n_iw_sampled_ex_for_generation`: Number of importance-weighted examples for generation (default: 5)
- `n_top_ex_for_scoring`: Number of top examples for scoring (default: 2)
- `n_random_ex_for_scoring`: Number of random examples for scoring (default: 10)
- `n_iw_sampled_ex_for_scoring`: Number of importance-weighted examples for scoring (default: 2)
- `act_threshold_frac`: Fraction of max activation to use as threshold (default: 0.01)
- `max_tokens_in_explanation`: Max tokens in LLM explanation (default: 30)
- `use_demos_in_explanation`: Whether to include example explanations in prompt (default: True)
- `dead_latent_threshold`: Threshold below which features are considered dead (default: -1.0)
- `scoring`: Whether to run scoring phase (default: True)

**Note**: Set `FORCE_RERUN = True` to regenerate artifacts even if results exist.

## Results Location

All outputs saved to `Results/` folder:

```
Results/
├── autointerp_<model>_layer<num>_features<list>_<timestamp>.txt  # Logs
├── <sae_id>_eval_results.json                                    # Results JSON
├── <model>_layer<num>_features_summary_<timestamp>.csv          # CSV summary
└── artifacts_<model>_layer<num>_<timestamp>/
    └── autointerp/
        └── <model>_<tokens>_tokens_<ctx>_ctx.pt                # Tokenized dataset
```

**Example Result:**
```json
{
  "eval_result_metrics": {
    "autointerp": {
      "autointerp_score": 0.7857,
      "autointerp_std_dev": 0.1750
    }
  },
  "eval_result_unstructured": {
    "0": {
      "explanation": "Financial market terminology and trading concepts",
      "score": 0.8234,
      "predictions": [1, 3, 5, 7],
      "correct seqs": [1, 3, 5, 8]
    }
  },
  "eval_config": {
    "model_name": "ProsusAI/finbert",
    "override_latents": [0, 1, 2, 3, 4],
    "total_tokens": 500000
  }
}
```

The AutoInterp score (0.7857) means the LLM correctly identified activating sequences 78.57% of the time. Higher scores indicate better interpretability—the explanation more accurately predicts when the feature activates.

## CSV Summary

The tool automatically generates a CSV summary file with columns: `layer`, `feature`, `label` (explanation), and `autointerp_score`. This makes it easy to review and analyze results across many features. See example outputs in [`Results/`](Results/) folder.

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

## Comparison With Delphi

Delphi is another LLM-based interpretability technique that also explains features using natural language, but the design philosophy differs.

**AutoInterp Eval (SAEBench Style)**

What it does: Uses only activation examples, produces a simple label, description, and AutoInterp score. Light-weight, fast, easy to run. Minimal assumptions about the model. No extra metrics or verification.

Strengths: Extremely simple, domain-agnostic, architecture-agnostic, ideal for large sweeps across thousands of features.

Limitations: Score depends on LLM's ability to predict activations, not verified against ground truth. No built-in quality metrics beyond the AutoInterp score. May produce unclear labels for noisy features.

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
