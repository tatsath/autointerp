# Neuronpedia Max-Activation Explainer

This implementation adds a Neuronpedia-style max-activation explainer to the AutoInterp framework. It uses cached activations from Delphi without modifying the Delphi codebase.

## Overview

The Neuronpedia explainer uses a concise, max-activation approach to generate precise labels for latent features:

- **Top K max-activation examples**: Uses the top 24 examples where the feature activates most strongly
- **Context windows**: Shows ±12 tokens around each max-activation token
- **Concise labels**: Generates ≤18 word descriptions focused on specific financial concepts
- **No code changes**: Works with existing cached activations from Delphi

## Files Created

### 1. `autointerp_full/explainers/np_max_act_explainer.py`

The main explainer class that implements the Neuronpedia max-activation approach:

- `NPMaxActExplainer`: Extends the base `Explainer` class
- Uses top K max-activation examples with surrounding context
- Generates JSON output with granularity, focus, label, and say_token fields
- Parses responses to extract concise labels

### 2. `run_np_max_act_explainer.py`

Standalone Python script to run the explainer on cached activations:

- Loads cached activations using `LatentDataset`
- Processes specified features/hookpoints
- Generates labels and saves them to the results folder
- Can be run independently or integrated into pipelines

### 3. `run_np_max_act_labels.sh`

Shell script to run the Neuronpedia explainer (similar to `run_nemotron_finance_news_autointerp_system.sh`):

- Extracts top features from summary file
- Uses cached activations from previous Delphi runs
- Generates labels using the max-activation explainer
- Creates CSV summaries of results

## Usage

### Prerequisites

1. **Cached activations**: You need to have cached activations from a previous Delphi run. The script looks for them in:
   ```
   results/nemotron_finance_news_run_system/latents/
   ```

2. **vLLM server**: The explainer requires a running vLLM server (default: `http://localhost:8002/v1`)

### Running the Script

```bash
# Make sure you're in the autointerp_full_finance_optimized directory
cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full_finance_optimized

# Run the Neuronpedia explainer
bash run_np_max_act_labels.sh
```

### Manual Usage

You can also run the Python script directly:

```bash
python run_np_max_act_explainer.py \
    --latents_path results/nemotron_finance_news_run_system/latents \
    --explanations_path results/nemotron_finance_news_run_np_max_act/explanations \
    --hookpoints backbone.layers.28 \
    --feature_num 2216 6105 8982 18529 19903 \
    --explainer_model Qwen/Qwen2.5-72B-Instruct \
    --explainer_api_base_url http://localhost:8002/v1 \
    --model nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
    --k_max_act 24 \
    --window 12 \
    --verbose
```

## Configuration

### Key Parameters

- `--k_max_act`: Number of top max-activation examples to use (default: 24)
- `--window`: Context window size around max-act token (default: 12, meaning ±12 tokens)
- `--feature_num`: List of feature indices to process (default: all)
- `--hookpoints`: Hookpoints to process (e.g., `backbone.layers.28`)

### Prompt Style

The explainer uses a concise, finance-focused prompt:

```
You are labeling ONE hidden feature from a language model trained on financial text.
You will see the top-activating tokens and short surrounding text spans.
Infer the single clearest description of what this feature detects.

Rules:
- Be SPECIFIC and CONCISE (≤ 18 words). No filler.
- Prefer concrete finance concepts
- Output strict JSON only
```

## Output Format

The explainer generates JSON with the following structure:

```json
{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "Entity/Sector/Event name or 'N/A'",
  "label": "≤18 words, precise financial description",
  "say_token": "TOKEN if applicable else 'N/A'"
}
```

The final label (stored in `.txt` files) is extracted from the `label` field.

## Integration with Delphi

This implementation:

✅ **Uses cached activations** from Delphi without modification  
✅ **No code changes** to Delphi's core functionality  
✅ **Compatible** with existing scoring pipelines  
✅ **Standalone** - can be run independently or integrated  

## Results

Results are saved to:
```
results/nemotron_finance_news_run_np_max_act/
├── explanations/
│   ├── backbone.layers.28_latent2216.txt
│   ├── backbone.layers.28_latent6105.txt
│   └── ...
├── nemotron_finance_results_summary_enhanced.csv
└── run.log
```

## Comparison with Default Explainer

| Feature | Default Explainer | Neuronpedia Explainer |
|---------|------------------|----------------------|
| Approach | Full context windows | Max-activation tokens only |
| Examples | All training examples | Top K max-activation examples |
| Context | Full sequence | ±12 tokens around max-act |
| Label length | 8-15 words | ≤18 words (more concise) |
| Prompt style | Detailed with examples | Concise, direct |
| Say-token detection | No | Yes (optional) |

## References

- [Neuronpedia AutoInterp](https://github.com/hijohnnylin/neuronpedia/tree/main/apps/autointerp)
- Based on the "np_max-act" approach from Neuronpedia's documentation

## Notes

- The explainer works best when you have high-quality cached activations
- For best results, use the same dataset and token count as your original Delphi run
- The `say_token` field requires additional logit computation (currently disabled by default)
- Labels are optimized for finance-specific concepts and may need adjustment for other domains






