# AutoInterp Full - SAE Feature Interpretability

AutoInterp Full provides detailed explanations and confidence scores for SAE features using LLM-based analysis. It uses LLMs to generate human-readable explanations with F1 scores, precision, and recall metrics to validate feature quality.

## 📊 What Makes a Good Feature?

**High Quality Features:**
- **F1 Score > 0.7**: Overall accuracy of explanation
- **Precision > 0.8**: How often correct when activated  
- **Recall > 0.6**: How often it catches relevant cases
- **Clear Semantic Focus**: Explains concepts, not grammar

**Key Metrics Explained:**
- **F1 Score**: Harmonic mean of precision and recall (0.0-1.0)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Activation Strength**: How strongly feature activates on relevant text
- **Specialization**: Domain-specific vs general activation difference

## 🎯 Core Parameters

| Parameter | Default | Purpose | Speed Impact |
|-----------|---------|---------|--------------|
| `--n_tokens` | 10000000 | Number of tokens to process | **HIGH** - Lower = Much Faster |
| `--max_latents` | None | Number of features to analyze | **HIGH** - Lower = Faster |
| `--hookpoints` | [] | Model layers to analyze | **HIGH** - Fewer = Faster |
| `--scorers` | detection | Quality metrics (F1-based) | **MEDIUM** - Fewer = Faster |
| `--explainer_model` | gpt-3.5-turbo | AI model for explanations | **HIGH** - Smaller = Faster |
| `--explainer_provider` | openrouter | API provider | None |
| `--n_non_activating` | 50 | Negative examples for contrast | **MEDIUM** - Lower = Faster |
| `--non_activating_source` | random | Method for finding negatives | **HIGH** - FAISS = Slower |

## 🧠 FAISS Contrastive Learning

**How FAISS Works:**
1. **Embedding Generation**: Uses sentence-transformers to create text embeddings
2. **Similarity Search**: Builds FAISS index of non-activating examples  
3. **Hard Negative Selection**: Finds semantically similar but non-activating examples
4. **Contrastive Prompting**: Shows both activating and non-activating examples to AI

**Value Add:**
- **Better Explanations**: AI can distinguish between similar-looking content
- **Semantic Understanding**: Focuses on meaning, not just surface patterns
- **Robust Features**: Reduces false positives and improves accuracy

## 📝 Prompt Engineering

**Automatic Prompt Selection:**
- **DEFAULT**: Standard analysis prompt (when `--non_activating_source` not set)
- **FAISS CONTRASTIVE**: Contrastive prompt (when `--non_activating_source FAISS`)
- **CHAIN OF THOUGHT**: Detailed analysis prompt (optional)

**Best Practices:**
- **Don't modify prompts** - they're optimized for SAE interpretability
- **Use FAISS for better quality** - semantic similarity improves explanations
- **Keep explanations concise** - single phrases work better than long descriptions

**Custom Prompt Modification:**
If you need to modify prompts, locate them in the explainer classes:
- **File**: `autointerp_full/explainers/default.py`
- **Function**: `DefaultExplainer.generate_explanation()`
- **Key Variables**: `prompt_template`, `contrastive_prompt_template`
- **Warning**: Custom prompts may reduce explanation quality and F1 scores

## ⚡ Speed Optimization

**Order of Impact (Lower These First):**
1. `--n_tokens` (cache less data) - **HIGHEST IMPACT**
2. `--max_latents` (analyze fewer features) - **HIGH IMPACT**  
3. Explainer model size/quantization - **HIGH IMPACT**
4. Examples per feature - **MEDIUM IMPACT**
5. Disable FAISS (use `random`) - **MEDIUM IMPACT**

**Sample Commands:**

```bash
# Ultra-Fast Development (2-5 minutes)
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 50000 \
  --max_latents 20 \
  --hookpoints layers.16 \
  --scorers detection \
  --filter_bos \
  --name ultra-fast-dev

# Fast Production (15-30 minutes)  
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 2000000 \
  --max_latents 200 \
  --hookpoints layers.16 \
  --scorers detection recall \
  --filter_bos \
  --name fast-production

# Full Quality Research (3-6 hours)
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 10000000 \
  --max_latents 1000 \
  --hookpoints layers.16 \
  --scorers detection recall fuzz simulation \
  --filter_bos \
  --name full-quality-research
```

## 📋 Complete Parameter Reference

### 🎯 Core Model Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--model` | meta-llama/Meta-Llama-3-8B | Base LLM to analyze | None |
| `--sparse_model` | EleutherAI/sae-llama-3-8b-32x | SAE/Transcoder model path | None |
| `--hookpoints` | [] | Model layers where SAE is attached | **HIGH** - Fewer = Faster |
| `--max_latents` | None | Maximum features to analyze | **HIGH** - Lower = Faster |

### 🧠 Explainer Model Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--explainer_model` | hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 | LLM used to generate explanations | **HIGH** - Smaller = Faster |
| `--explainer_model_max_len` | 5120 | Maximum context length for explainer | **MEDIUM** - Lower = Faster |
| `--explainer_provider` | offline | How to run explainer | None |
| `--explainer` | default | Explanation strategy | None |

### 📊 Scoring Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--scorers` | ['fuzz', 'detection'] | Quality metrics to evaluate | **HIGH** - Fewer = Faster |
| `--num_examples_per_scorer_prompt` | 5 | Examples per prompt for scoring | **MEDIUM** - Lower = Faster |

### 🗃️ Dataset & Caching Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--dataset_repo` | EleutherAI/SmolLM2-135M-10B | Dataset source for generating activations | **MEDIUM** - Smaller = Faster |
| `--dataset_split` | train[:1%] | Dataset portion to use | **HIGH** - Smaller = Much Faster |
| `--dataset_name` | `` | Custom dataset name | None |
| `--dataset_column` | text | Column containing text data | None |
| `--n_tokens` | 10000000 | Total tokens to process | **HIGH** - Lower = Much Faster |
| `--batch_size` | 32 | Sequences per batch | **MEDIUM** - Optimize for GPU |
| `--cache_ctx_len` | 256 | Context length for each sequence | **MEDIUM** - Lower = Faster |
| `--n_splits` | 5 | Number of safetensors files | **LOW** - Fewer = Slightly Faster |

### 🔍 Example Construction Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--min_examples` | 200 | Minimum examples needed per feature | **MEDIUM** - Lower = Faster |
| `--n_examples_train` | 40 | Training examples for explanation | **MEDIUM** - Lower = Faster |
| `--n_examples_test` | 50 | Testing examples for validation | **MEDIUM** - Lower = Faster |
| `--n_non_activating` | 50 | Negative examples to contrast | **MEDIUM** - Lower = Faster |
| `--example_ctx_len` | 32 | Length of each example sequence | **MEDIUM** - Lower = Faster |
| `--center_examples` | True | Center examples on activation point | None |
| `--non_activating_source` | random | Source of negative examples | **HIGH** - FAISS = Slower |
| `--neighbours_type` | co-occurrence | Type of neighbor search | **LOW** - Different types have minimal impact |

### 🔧 Technical Parameters
| Parameter | Default | What It Controls | Speed Impact |
|-----------|---------|------------------|--------------|
| `--pipeline_num_proc` | 120 | CPU processes for data processing | **LOW** - Optimize for your CPU |
| `--num_gpus` | 8 | GPU count for model inference | **MEDIUM** - Fewer = Less overhead |
| `--seed` | 22 | Random seed for reproducibility | None |
| `--verbose` | True | Detailed logging output | None |
| `--filter_bos` | False | Filter beginning-of-sequence tokens | None |
| `--log_probs` | False | Gather log probabilities | **MEDIUM** - Disable = Faster |
| `--load_in_8bit` | False | 8-bit model loading for memory efficiency | **MEDIUM** - Enable = Faster |
| `--hf_token` | None | HuggingFace API token | None |
| `--overwrite` | [] | What to overwrite | None |

## 📁 Output Files

Results are saved in the `results/` directory:

### Key Output Files:
- **`explanations/`**: Human-readable feature explanations
- **`scores/detection/`**: F1 scores, precision, recall metrics
- **`latents/`**: Cached model activations for analysis
- **`run_config.json`**: Complete configuration used for the run

## 🔧 Advanced Usage

### Custom Dataset Example
```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 1000000 \
  --max_latents 100 \
  --hookpoints layers.16 \
  --dataset_repo "jyanimaulik/yahoo_finance_stockmarket_news" \
  --dataset_split "train[:1000]" \
  --scorers detection recall \
  --filter_bos \
  --name custom-finance-dataset
```

### Programmatic Usage
```python
from autointerp_full.latents import LatentCache
from autointerp_full.explainers import DefaultExplainer
from autointerp_full.clients import OpenRouter

# Cache activations
cache = LatentCache(model, submodule_dict, batch_size=8)
cache.run(n_tokens=10_000_000, tokens=tokens)

# Generate explanations
client = OpenRouter("gpt-3.5-turbo", api_key=key)
explainer = DefaultExplainer(client, tokenizer=tokenizer)
```

## 🧪 Available Scorers

| Scorer | Purpose | Best For |
|--------|---------|----------|
| `detection` | F1-based accuracy scoring | General feature validation |
| `recall` | Recall-focused scoring | High-sensitivity applications |
| `fuzz` | Fuzzing-based robustness | Adversarial testing |
| `simulation` | OpenAI neuron simulation | Research validation |
| `surprisal` | Loss-based scoring | Language modeling tasks |
| `embedding` | Semantic similarity scoring | Content-based features |

## 🚀 Pro Tips

**Start Small:**
- Use 50K tokens and 20 features first
- Test with `--scorers detection` only
- Use `train[:1000]` dataset slices

**Optimize for Speed:**
- Disable FAISS: `--non_activating_source random`
- Use quantized models: `--explainer_model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ"`
- Lower context: `--explainer_model_max_len 1024`

**Quality Improvements:**
- Enable FAISS for better explanations
- Use more tokens for better coverage
- Analyze multiple layers for comprehensive understanding


## 📄 License

Copyright 2024 the EleutherAI Institute

Licensed under the Apache License, Version 2.0
