# Usage Guide - Reorganized Structure

## Quick Start

### Complete Pipeline (All Steps)
```bash
# Activate conda environment
conda activate sae

# Run complete pipeline
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path "1. search/test_tokens.json" \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100
```

Results will be saved to:
- `results/1_search/` - Feature search results
- `results/2_labeling_lite/` - Basic labeling results  
- `results/3_labeling_advance/` - Advanced labeling results

## Individual Steps

### Step 1: Feature Search
```bash
cd "1. search"
conda activate sae
python run_search.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path test_tokens.json \
    --output_dir ../results/1_search \
    --score_type fisher \
    --num_features 20 \
    --n_samples 100
```

**Output**: `results/1_search/feature_list.json`

### Step 2: Basic Labeling (Lite)
```bash
cd "../2. autointerp_lite"
conda activate sae
python run_labeling.py \
    --search_output ../results/1_search \
    --output_dir ../results/2_labeling_lite
```

**Output**: 
- `results/2_labeling_lite/activating_sentences.json`
- `results/2_labeling_lite/feature_labels.json`

### Step 3: Advanced Labeling
```bash
cd "../3. autointerp_advance"
conda activate sae
python run_labeling_advanced.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --sae_id blocks.19.hook_resid_post \
    --search_output ../results/1_search \
    --output_dir ../results/3_labeling_advance \
    --n_samples 5000
```

**Output**:
- `results/3_labeling_advance/feature_examples.jsonl`
- `results/3_labeling_advance/feature_labels.json`

## Results Structure

All results are organized in `results/` with numbered subfolders:

```
results/
├── 1_search/                    # From Step 1
│   ├── feature_scores.pt
│   ├── feature_list.json         # ⭐ Used by Step 2 & 3
│   └── top_features.pt
│
├── 2_labeling_lite/              # From Step 2
│   ├── activating_sentences.json
│   └── feature_labels.json
│
└── 3_labeling_advance/           # From Step 3
    ├── feature_examples.jsonl
    └── feature_labels.json
```

## Notes

- **Conda Environment**: Use `conda activate sae` for all scripts
- **Seamless Flow**: Each step automatically reads from previous step's output
- **Reusable Files**: All original files are reused, just reorganized
- **Minimal Changes**: Only path updates, no logic changes

