# Tool-Call Feature Analysis with AutoInterp

This directory contains a customized version of AutoInterp for analyzing tool-call features in agent models.

## Overview

This setup is designed to analyze SAE features that are related to tool usage in agent models (like Llama 3.1 8B). The explainer uses specialized prompts that focus on tool behavior rather than generic topics.

## Key Features

1. **Tool-Call Focused Prompts**: The explainer analyzes features in the context of agent tool usage, identifying:
   - Which tools are triggered (web_search, code_exec, etc.)
   - Call tendencies (strong_trigger, weak_trigger, preference, suppression, unrelated)
   - Preconditions for tool activation
   - Tool argument patterns

2. **JSON Output Format**: Explanations are provided in JSON format with structured fields:
   - `tool_match`: Name of tool or 'multi_tool' or 'none'
   - `call_tendency`: strong_trigger | weak_trigger | preference | suppression | unrelated
   - `preconditions`: Description of question/context patterns
   - `tool_arguments_pattern`: How arguments are structured
   - `explanation`: Concise sentence describing tool influence

## Files

- `autointerp_full/explainers/default/prompts_toolcall.py`: Tool-call specific prompts with few-shot examples
- `autointerp_full/explainers/contrastive_explainer.py`: Modified to use tool-call prompts
- `run_toolcall_features_autointerp.sh`: Script to run analysis on features from tool_features.json
- `generate_toolcall_enhanced_csv.py`: Script to generate CSV summary with metrics

## Usage

### Running Tool-Call Feature Analysis

1. Ensure vLLM server is running:
   ```bash
   bash start_vllm_server_72b.sh
   ```

2. Run the analysis script:
   ```bash
   cd /home/nvidia/Documents/Hariom/autointerp/autointerp_full_optimized_toolcall
   bash run_toolcall_features_autointerp.sh
   ```

The script will:
- Extract features from `tool_features.json` (both market_features and news_features)
- Run AutoInterp analysis using tool-call prompts
- Generate explanations for each feature
- Create CSV summaries with F1 scores

### Input Format

The script expects `tool_features.json` with the following structure:
```json
{
  "market_features": [
    {"feature_id": 186, "mu_tool": 3.29, "mu_other": 0.45, "ftc_confidence": 0.76},
    ...
  ],
  "news_features": [
    {"feature_id": 197, "mu_tool": 4.53, "mu_other": 1.67, "ftc_confidence": 0.46},
    ...
  ]
}
```

### Output

Results are saved in a timestamped directory:
- `toolcall_features_autointerp_results_YYYYMMDD_HHMMSS/`
  - `toolcall_features_run/`: Full AutoInterp results
    - `explanations/`: Feature explanations (JSON parsed to text)
    - `scores/`: Detection and fuzz scorer results
  - `toolcall_features_autointerp_results_summary_enhanced.csv`: Enhanced CSV with metrics
  - `toolcall_features_autointerp_results_summary.csv`: Basic CSV

## Configuration

Key configuration in `run_toolcall_features_autointerp.sh`:
- `BASE_MODEL`: "meta-llama/Llama-3.1-8B-Instruct"
- `SAE_MODEL_DIR`: Path to SAE model (layer 19, K=32, 400 features)
- `LAYER`: 19
- `N_TOKENS`: 500000 (500K tokens for analysis)
- `EXPLAINER_MODEL`: "Qwen/Qwen2.5-72B-Instruct"
- `DATASET`: lmsys/lmsys-chat-1m (contains agent conversations)

## Prompt Structure

The tool-call prompts include:
1. System prompt explaining the task (analyzing tool-use features)
2. Critical requirements (focus on tool behavior, not generic topics)
3. Decision framework (strong_trigger, weak_trigger, etc.)
4. JSON output format specification
5. Few-shot examples:
   - Search tool example
   - Code tool example
   - Tool suppression example

## Notes

- The explainer extracts the `explanation` field from JSON responses
- Results are saved in timestamped directories to preserve history
- CSV generation includes F1 score thresholds (>= 0.70 for detection, <= 0.30 for fuzz)

