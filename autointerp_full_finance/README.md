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
| `--explainer_api_base_url` | None | API base URL for API-based providers | None |
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

## 🚀 vLLM Provider Support

AutoInterp Full now supports **vLLM** as an explainer provider for faster inference through OpenAI-compatible API endpoints.

### What is vLLM?

vLLM is a high-throughput and memory-efficient inference engine for LLMs that provides:
- **Faster Inference**: Optimized for high-throughput serving
- **Memory Efficiency**: Better GPU memory utilization
- **OpenAI Compatibility**: Drop-in replacement for OpenAI API
- **Multi-GPU Support**: Automatic tensor parallelism

### Setting Up vLLM Server

**1. Install vLLM:**
```bash
pip install vllm
```

**2. Start vLLM Server:**
```bash
# Basic setup
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0

# Advanced setup with multiple GPUs
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --trust-remote-code
```

**3. Verify Server:**
```bash
curl http://localhost:8002/v1/models
```

### Using vLLM with AutoInterp

**Basic vLLM Usage:**
```bash
python -m autointerp_full \
  meta-llama/Llama-3.1-8B-Instruct \
  /path/to/sae/model \
  --n_tokens 200000 \
  --feature_num 0 1 2 3 4 \
  --hookpoints layers.19 \
  --explainer_provider vllm \
  --explainer_model Qwen/Qwen2.5-7B-Instruct \
  --explainer_api_base_url http://localhost:8002/v1 \
  --scorers detection \
  --name vllm_run
```

**Advanced vLLM Configuration:**
```bash
python -m autointerp_full \
  meta-llama/Llama-3.1-8B-Instruct \
  /path/to/sae/model \
  --n_tokens 500000 \
  --feature_num 0 1 2 3 4 5 6 7 8 9 \
  --hookpoints layers.19 \
  --explainer_provider vllm \
  --explainer_model meta-llama/Llama-3.1-8B-Instruct \
  --explainer_api_base_url http://localhost:8002/v1 \
  --explainer_model_max_len 4096 \
  --num_examples_per_scorer_prompt 10 \
  --n_non_activating 20 \
  --min_examples 1 \
  --non_activating_source FAISS \
  --scorers detection \
  --verbose \
  --name vllm_advanced_run
```

### vLLM Example Script

Create `example_vllm.sh`:

```bash
#!/bin/bash

echo "🚀 AutoInterp Analysis with vLLM Provider"
echo "=========================================="

# Configuration
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
SAE_MODEL="/path/to/your/sae/model"
EXPLAINER_MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_URL="http://localhost:8002/v1"
N_TOKENS=200000
LAYER=19

# Check vLLM server status
echo "🔍 Checking vLLM server status..."
if curl --output /dev/null --silent --head --fail "$VLLM_URL/models"; then
    echo "✅ vLLM server is running at $VLLM_URL"
else
    echo "❌ vLLM server is NOT running at $VLLM_URL"
    echo "Please start vLLM server first:"
    echo "python -m vllm.entrypoints.openai.api_server --model $EXPLAINER_MODEL --port 8002"
    exit 1
fi

# Run AutoInterp analysis
echo "🔍 Running AutoInterp Analysis with vLLM Provider..."

python -m autointerp_full \
    "$BASE_MODEL" \
    "$SAE_MODEL" \
    --n_tokens "$N_TOKENS" \
    --feature_num 0 1 2 3 4 \
    --hookpoints "layers.$LAYER" \
    --scorers detection \
    --explainer_model "$EXPLAINER_MODEL" \
    --explainer_provider vllm \
    --explainer_api_base_url "$VLLM_URL" \
    --explainer_model_max_len 4096 \
    --num_examples_per_scorer_prompt 10 \
    --n_non_activating 20 \
    --min_examples 1 \
    --non_activating_source FAISS \
    --verbose \
    --name vllm_run

echo "✅ Analysis completed! Check results in: results/vllm_run/"
```

### Supported Providers

| Provider | Description | Use Case |
|----------|-------------|----------|
| `offline` | Local HuggingFace models | Development, privacy |
| `openrouter` | OpenRouter API | Production, multiple models |
| `vllm` | vLLM server | High-throughput, custom deployments |

### Performance Comparison

| Provider | Speed | Memory | Scalability | Setup Complexity |
|----------|-------|--------|-------------|------------------|
| **offline** | Medium | High | Low | Low |
| **openrouter** | Fast | Low | High | Low |
| **vllm** | **Very Fast** | Medium | **Very High** | Medium |

### Troubleshooting vLLM

**Common Issues:**

1. **Server Not Running:**
   ```bash
   curl http://localhost:8002/v1/models
   # Should return model list, not connection error
   ```

2. **CUDA Out of Memory:**
   ```bash
   # Reduce GPU memory utilization
   --gpu-memory-utilization 0.5
   ```

3. **Model Loading Errors:**
   ```bash
   # Add trust-remote-code for custom models
   --trust-remote-code
   ```

4. **Timeout Issues:**
   ```bash
   # Increase timeout in AutoInterp
   --explainer_model_max_len 2048  # Reduce context length
   ```

## 🚀 Pro Tips

**Start Small:**
- Use 50K tokens and 20 features first
- Test with `--scorers detection` only
- Use `train[:1000]` dataset slices

**Optimize for Speed:**
- **Use vLLM provider** for fastest inference
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
