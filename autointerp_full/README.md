# AutoInterp Full

AutoInterp Full provides automated interpretability analysis for Sparse Autoencoder (SAE) features in large language models. It uses LLM-based analysis to generate human-readable explanations and validate feature quality through automated scoring.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Cache Management](#cache-management)
5. [Prompt Customization](#prompt-customization)
6. [Configuration Parameters](#configuration-parameters)
7. [Output Structure](#output-structure)
8. [Advanced Features](#advanced-features)
9. [Examples](#examples)
10. [vLLM Server Setup](#vllm-server-setup)

## Quick Start

### Basic Command

```bash
python -m autointerp_full \
  <model_name> \
  <sparse_model_path> \
  --hookpoints <layer_name> \
  --n_tokens <number> \
  --max_latents <number>
```

### Minimal Example

```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae/model \
  --hookpoints layers.16 \
  --n_tokens 50000 \
  --max_latents 20 \
  --name my_analysis
```

### Recommended Workflow

1. **Start with a small test run** to verify setup:
   ```bash
   python -m autointerp_full \
     <model> <sae_path> \
     --hookpoints layers.16 \
     --n_tokens 50000 \
     --max_latents 10 \
     --name test_run
   ```

2. **Scale up for production analysis**:
   ```bash
   python -m autointerp_full \
     <model> <sae_path> \
     --hookpoints layers.16 \
     --n_tokens 10000000 \
     --max_latents 500 \
     --name production_run
   ```

## Installation

```bash
pip install -e .
```

For visualization support (optional):
```bash
pip install -e ".[visualize]"
```

## Basic Usage

### Required Arguments

- `model`: Base model identifier (e.g., `meta-llama/Llama-2-7b-hf`)
- `sparse_model`: Path to SAE/transcoder model directory
- `--hookpoints`: Model layers to analyze (e.g., `layers.16`)

### Essential Parameters

- `--n_tokens`: Number of tokens to process (default: 10000000)
- `--max_latents`: Maximum number of features to analyze (default: None, analyzes all)
- `--name`: Run identifier for organizing results

### Common Options

- `--scorers`: Scoring methods to use (default: `detection`)
- `--explainer_model`: Model for generating explanations
- `--explainer_provider`: Provider type (`offline`, `openrouter`, `vllm`)
- `--filter_bos`: Filter beginning-of-sequence tokens

## Cache Management

Activation caches store computed model activations and can be reused across multiple runs, significantly reducing computation time. The cache is automatically generated during the first run and stored in `results/<run_name>/latents/`.

### How Cache Reuse Works

1. **First run** generates the cache:
   ```bash
   python -m autointerp_full <model> <sae> --hookpoints layers.16 --n_tokens 10000000 --name base_run
   ```
   This creates activation data for the specified hookpoints and stores it in `results/base_run/latents/`.

2. **Subsequent runs** automatically reuse the cache when:
   - Same model identifier
   - Same hookpoints
   - Same dataset configuration (`dataset_repo`, `dataset_split`, `dataset_column`)
   - Same cache parameters (`n_tokens`, `cache_ctx_len`, `batch_size`)

3. **Force regeneration** if needed:
   ```bash
   python -m autointerp_full <model> <sae> --overwrite cache --name fresh_run
   ```

### Cache Structure

Caches are stored in `results/<run_name>/latents/` and include:
- Activation data in safetensors format (split across multiple files)
- Token sequences and metadata
- Configuration metadata (`config.json`)
- Each hookpoint requires its own cache directory

### Best Practices

- **Generate large caches once**: Create a comprehensive cache with high `n_tokens` value, then reuse it for multiple feature analyses
- **Consistent parameters**: Use the same `n_tokens`, `cache_ctx_len`, and `batch_size` across runs to maximize cache reuse
- **Separate caches per dataset**: Different datasets require separate caches
- **Cache size**: Larger caches provide better feature coverage but require more storage space

### Example Workflow

```bash
# Step 1: Generate comprehensive cache (one-time, takes time)
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --hookpoints layers.16 \
  --n_tokens 10000000 \
  --name base_cache

# Step 2: Reuse cache for different feature analyses (fast)
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --hookpoints layers.16 \
  --n_tokens 10000000 \
  --max_latents 100 \
  --name analysis_run_1

python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --hookpoints layers.16 \
  --n_tokens 10000000 \
  --max_latents 200 \
  --name analysis_run_2
```

## Prompt Customization

AutoInterp supports external prompt configuration through YAML files, allowing you to customize all prompts without modifying source code. This is useful for domain-specific adaptations or fine-tuning explanation quality.

### Enabling Prompt Override

To use custom prompts, enable prompt override and specify the YAML file:

```bash
python -m autointerp_full \
  <model> <sae_path> \
  --prompt_override \
  --prompt_config_file /path/to/prompts.yaml \
  [other arguments...]
```

If `--prompt_config_file` is not specified, the system automatically looks for `prompts.yaml` in the project root directory.

### YAML Configuration Structure

Create a `prompts.yaml` file with this structure:

```yaml
explainers:
  default:
    system: |
      Your custom system prompt here...
      {prompt}
    
    system_single_token: |
      Your custom single token prompt...
      {prompt}
    
    system_contrastive: |
      Your custom contrastive prompt...

  np_max_act:
    system_concise: |
      Your custom np_max_act prompt...

scorers:
  detection:
    system: |
      Your custom detection scorer prompt...
  
  fuzz:
    system: |
      Your custom fuzz scorer prompt...
  
  intruder:
    system: |
      Your custom intruder scorer prompt...
```

### Key Points

- **Placeholder replacement**: The `{prompt}` placeholder is automatically replaced with example data during execution
- **Partial override**: You can override only specific prompts - others will use defaults from the code
- **Default behavior**: If `--prompt_override` is not specified, default prompts from the code are used
- **File location**: Place `prompts.yaml` in the project root, or specify the path with `--prompt_config_file`

### Available Prompt Types

**Explainer Prompts** (`explainers.default`):
- `system`: Main system prompt for feature explanation
- `system_single_token`: Prompt for single-token analysis
- `system_contrastive`: Prompt for contrastive analysis
- `cot`: Chain-of-thought reasoning prompt (optional)

**NP Max Act Explainer** (`explainers.np_max_act`):
- `system_concise`: Concise labeling prompt for max-activation approach

**Scorer Prompts** (`scorers`):
- `detection.system`: Detection scorer prompt
- `fuzz.system`: Fuzzing scorer prompt
- `intruder.system`: Intruder detection scorer prompt

### Example: Domain-Specific Prompts

For financial domain analysis:

```yaml
explainers:
  default:
    system: |
      You are a financial AI researcher analyzing neural network activations 
      in a model trained on financial news and market data. Your task is to 
      provide specific, precise financial explanations.
      
      Focus on specific financial metrics, market sectors, and economic indicators.
      {prompt}
```

### Environment Variable Alternative

You can also set the prompt config file path using an environment variable:

```bash
export PROMPT_CONFIG_FILE=/path/to/your/prompts.yaml
python -m autointerp_full --prompt_override [other arguments...]
```

For detailed documentation, see `PROMPT_CONFIG_README.md`.

## Configuration Parameters

### Core Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | meta-llama/Meta-Llama-3-8B | Base LLM to analyze |
| `--sparse_model` | EleutherAI/sae-llama-3-8b-32x | SAE model path |
| `--hookpoints` | [] | Model layers where SAE is attached |
| `--max_latents` | None | Maximum features to analyze |
| `--feature_num` | None | Specific feature indices to analyze |

### Explainer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--explainer_model` | hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 | LLM for explanations |
| `--explainer_model_max_len` | 5120 | Maximum context length |
| `--explainer_provider` | offline | Provider type (offline, openrouter, vllm) |
| `--explainer_api_base_url` | None | API base URL for API-based providers |
| `--explainer` | default | Explanation strategy |

### Prompt Customization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt_override` | False | Enable external prompt configuration |
| `--prompt_config_file` | None | Path to YAML file with custom prompts |

See [Prompt Customization](#prompt-customization) section for details.

### Scoring Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--scorers` | ['fuzz', 'detection'] | Quality metrics to evaluate |
| `--num_examples_per_scorer_prompt` | 5 | Examples per prompt for scoring |

### Dataset and Caching Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_repo` | EleutherAI/SmolLM2-135M-10B | Dataset source |
| `--dataset_split` | train[:1%] | Dataset portion to use |
| `--dataset_column` | text | Column containing text data |
| `--n_tokens` | 10000000 | Total tokens to process |
| `--batch_size` | 32 | Sequences per batch |
| `--cache_ctx_len` | 256 | Context length for each sequence |
| `--n_splits` | 5 | Number of safetensors files |

### Example Construction Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min_examples` | 200 | Minimum examples needed per feature |
| `--n_examples_train` | 40 | Training examples for explanation |
| `--n_examples_test` | 50 | Testing examples for validation |
| `--n_non_activating` | 50 | Negative examples for contrast |
| `--example_ctx_len` | 32 | Length of each example sequence |
| `--center_examples` | True | Center examples on activation point |
| `--non_activating_source` | random | Source of negative examples (random, neighbours, FAISS) |
| `--neighbours_type` | co-occurrence | Type of neighbor search |

### Technical Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pipeline_num_proc` | cpu_count()//2 | CPU processes for data processing |
| `--num_gpus` | torch.cuda.device_count() | GPU count for model inference |
| `--seed` | 22 | Random seed for reproducibility |
| `--verbose` | True | Detailed logging output |
| `--filter_bos` | False | Filter beginning-of-sequence tokens |
| `--log_probs` | False | Gather log probabilities |
| `--load_in_8bit` | False | 8-bit model loading |
| `--hf_token` | None | HuggingFace API token |
| `--overwrite` | [] | Components to recompute (cache, neighbours, scores) |
| `--enable_visualization` | False | Generate visualization plots (requires plotly) |


## Output Structure

Results are saved in `results/<run_name>/`:

```
results/
└── <run_name>/
    ├── run_config.json          # Complete configuration
    ├── latents/                  # Cached activations
    │   ├── activations_*.safetensors
    │   └── config.json
    ├── explanations/             # Feature explanations
    │   └── <hookpoint>_latent_<id>.txt
    └── scores/                  # Scoring results
        └── <scorer_type>/
            └── <hookpoint>_latent_<id>.txt
```

### Key Output Files

- **explanations/**: Human-readable feature explanations
- **scores/**: Quality metrics and validation results
- **latents/**: Cached model activations (reusable)
- **run_config.json**: Complete configuration used for the run

## Advanced Features

### FAISS Contrastive Learning

FAISS uses semantic similarity search to find hard negative examples, improving explanation quality:

```bash
python -m autointerp_full \
  <model> <sae_path> \
  --non_activating_source FAISS \
  [other arguments...]
```

**Benefits:**
- Better distinction between similar content
- Improved explanation accuracy
- More robust feature validation

**Trade-offs:**
- Slower than random sampling
- Requires embedding model computation

### Available Scorers

| Scorer | Purpose | Use Case |
|--------|---------|----------|
| `detection` | F1-based accuracy scoring | General feature validation |
| `fuzz` | Fuzzing-based robustness | Adversarial testing |
| `simulation` | OpenAI neuron simulation | Research validation |
| `surprisal` | Loss-based scoring | Language modeling tasks |
| `embedding` | Semantic similarity scoring | Content-based features |

### Provider Options

| Provider | Description | Use Case |
|----------|-------------|----------|
| `offline` | Local HuggingFace models | Development, privacy |
| `openrouter` | OpenRouter API | Production, multiple models |
| `vllm` | vLLM server | High-throughput deployments |

## Examples

### Minimal Development Run

```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 50000 \
  --max_latents 20 \
  --hookpoints layers.16 \
  --scorers detection \
  --filter_bos \
  --name dev_run
```

### Production Analysis

```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 10000000 \
  --max_latents 500 \
  --hookpoints layers.16 \
  --scorers detection fuzz \
  --non_activating_source FAISS \
  --filter_bos \
  --name production_run
```

### Custom Dataset

```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --n_tokens 1000000 \
  --max_latents 100 \
  --hookpoints layers.16 \
  --dataset_repo "custom/dataset" \
  --dataset_split "train[:1000]" \
  --scorers detection \
  --name custom_dataset_run
```

### Using vLLM Provider

```bash
python -m autointerp_full \
  meta-llama/Llama-3.1-8B-Instruct \
  /path/to/sae \
  --n_tokens 500000 \
  --feature_num 0 1 2 3 4 \
  --hookpoints layers.19 \
  --explainer_provider vllm \
  --explainer_model Qwen/Qwen2.5-7B-Instruct \
  --explainer_api_base_url http://localhost:8002/v1 \
  --scorers detection \
  --name vllm_run
```

### With Prompt Override

```bash
python -m autointerp_full \
  meta-llama/Llama-2-7b-hf \
  /path/to/sae \
  --hookpoints layers.16 \
  --n_tokens 1000000 \
  --max_latents 100 \
  --prompt_override \
  --prompt_config_file /path/to/custom_prompts.yaml \
  --name custom_prompts_run
```

## vLLM Server Setup

vLLM provides high-throughput inference for explainer models. This section covers setting up and using vLLM with AutoInterp.

### Installation

```bash
pip install vllm
```

### Starting vLLM Server

**Basic Setup:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 4096 \
  --tensor-parallel-size 4 \
  --host 0.0.0.0
```

**Advanced Multi-GPU Setup:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8002 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --tensor-parallel-size 8 \
  --host 0.0.0.0 \
  --trust-remote-code
```

### Verifying Server

```bash
curl http://localhost:8002/v1/models
```

### Using vLLM with AutoInterp

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

### Troubleshooting

**Server Not Running:**
- Verify with: `curl http://localhost:8002/v1/models`
- Check port availability and firewall settings

**CUDA Out of Memory:**
- Reduce `--gpu-memory-utilization` (e.g., 0.5)
- Use smaller models or reduce `--tensor-parallel-size`

**Model Loading Errors:**
- Add `--trust-remote-code` for custom models
- Verify model path and HuggingFace access

**Timeout Issues:**
- Reduce `--explainer_model_max_len`
- Check network connectivity for API calls

## Performance Optimization

### Speed Optimization Tips

1. **Reduce token count**: Lower `--n_tokens` for faster runs
2. **Limit features**: Use `--max_latents` to analyze fewer features
3. **Fewer hookpoints**: Analyze fewer layers
4. **Optimize explainer**: Use smaller or quantized models
5. **Disable FAISS**: Use `--non_activating_source random` instead
6. **Reduce examples**: Lower `--n_examples_train` and `--n_examples_test`

### Quality Optimization Tips

1. **More tokens**: Increase `--n_tokens` for better coverage
2. **Enable FAISS**: Use `--non_activating_source FAISS` for better explanations
3. **Multiple layers**: Analyze multiple hookpoints for comprehensive understanding
4. **Reuse cache**: Leverage existing caches to save computation time

## License

[Add license information here]
