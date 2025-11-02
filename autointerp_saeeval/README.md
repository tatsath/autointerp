# AutoInterp SAE Evaluation

Standalone tool for evaluating SAE feature interpretability using AutoInterp methodology.

## What Was Done

Extracted AutoInterp evaluation logic from SAEBench into a self-contained module. The tool uses an LLM judge (OpenAI API) to automatically generate explanations for SAE features and evaluate how well those explanations predict feature activations.

## How It Works

**Two-Phase Evaluation:**

1. **Generation Phase**: 
   - Collects top-activating sequences and importance-weighted samples from SAE activations
   - Formats tokens with `<<token>>` syntax where features activate
   - Asks LLM: "What does this feature activate on?" → Gets explanation

2. **Scoring Phase**:
   - Creates shuffled mix of top sequences, random sequences, and importance-weighted samples
   - Gives LLM the explanation and asks: "Which numbered sequences will activate this feature?"
   - Compares LLM predictions to actual activations → Calculates score

**Prompts Used:**
- **Generation**: Shows activating tokens (`<<token>>`) and asks what the feature activates on
- **Scoring**: Shows explanation + numbered sequences, asks for comma-separated indices of activating sequences

## Results Location

All outputs saved to `Results/` folder:

```
Results/
├── autointerp_<model>_layer<num>_features<list>_<timestamp>.txt  # Logs
├── <sae_id>_eval_results.json                                    # Results JSON
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
      "autointerp_std_dev": 0.0
    }
  },
  "eval_config": {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "override_latents": [3, 6],
    "total_tokens": 100000
  }
}
```
Score: **0.7857** = LLM correctly identified activating sequences 78.57% of the time.

## Files

```
autointerp_saeeval/
├── autointerp/              # AutoInterp evaluation module
│   ├── eval_config.py      # Configuration
│   ├── eval_output.py      # Output schema
│   ├── main.py             # Core evaluation logic
│   ├── sae_encode.py       # SAE encoding utilities
│   └── eval_output_schema_autointerp.json
├── openai_api_key.txt      # API key (required)
├── run_autointerp_features.py  # Main script
├── Results/                 # All outputs
└── README.md
```

## Usage

1. Edit `run_autointerp_features.py`:
   - `MODEL_NAME`: Model to evaluate
   - `SAE_PATH`: Path to SAE
   - `FEATURES_TO_EVALUATE`: Features to evaluate (e.g., `[3, 6]`)
   - `TOTAL_TOKENS`: Tokens for evaluation (default: 100,000 for testing)

2. Run:
   ```bash
   conda run -n sae python run_autointerp_features.py
   ```

**Note**: Set `FORCE_RERUN = True` to regenerate artifacts even if results exist.

## Dependencies

- Python: `torch`, `safetensors`, `transformer_lens`, `sae_lens`, `openai`
- SAEBench: Only for `TopKSAE` and `CustomSAEConfig` (minimal)
- OpenAI API key: Required in `openai_api_key.txt`
