# Reorganization Summary

## New Structure

The codebase has been reorganized into three main groups:

### 1. search/ - Feature Search
- **Purpose**: Identify domain-specific features using activation-separation-based scoring
- **Main Files**:
  - `main/run_feature_search.py` - Core search implementation
  - `main/compute_score.py` - Scoring algorithm
  - `domains/finance/` - Domain-specific wrappers
  - `run_search.py` - Unified entry point
- **Output**: `results/1_search/`
  - `feature_scores.pt`
  - `feature_list.json`
  - `top_features.pt`

### 2. autointerp_lite/ - Basic Labeling
- **Purpose**: Collect context windows and generate labels using vLLM API
- **Main Files**:
  - `collect_examples.py` - Collects positive/negative context windows
  - `label_features.py` - Generates labels using vLLM API
  - `run_labeling.py` - Unified entry point (chains both steps)
- **Output**: `results/2_labeling_lite/`
  - `activating_sentences.json`
  - `feature_labels.json`

### 3. autointerp_advance/ - Advanced Labeling
- **Purpose**: Extract examples using SaeVisRunner and generate labels
- **Main Files**:
  - `extract_examples.py` - Extracts examples using SaeVisRunner
  - `generate_labels.py` - Generates labels from extracted examples
  - `run_labeling_advanced.py` - Unified entry point (chains both steps)
- **Output**: `results/3_labeling_advance/`
  - `feature_examples.jsonl`
  - `feature_labels.json`

## Results Folder Structure

All results are saved to `results/` with numbered subfolders:
```
results/
├── 1_search/              # Search results
├── 2_labeling_lite/        # Basic labeling results
└── 3_labeling_advance/      # Advanced labeling results
```

## Usage

### Complete Pipeline
```bash
python pipeline.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --sae_path /path/to/sae \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --tokens_str_path "1. search/test_tokens.json" \
    --sae_id blocks.19.hook_resid_post \
    --score_type fisher \
    --num_features 20
```

### Individual Steps
```bash
# Step 1: Search
cd "1. search"
python run_search.py --model_path ... --sae_path ... --output_dir ../results/1_search

# Step 2: Basic Labeling
cd "../2. autointerp_lite"
python run_labeling.py --search_output ../results/1_search

# Step 3: Advanced Labeling
cd "../3. autointerp_advance"
python run_labeling_advanced.py --model_path ... --sae_path ... --search_output ../results/1_search
```

## Key Changes

1. **All files reused** - No duplication, files moved to appropriate groups
2. **Paths updated** - All relative paths updated to work with new structure
3. **Results centralized** - All results in `results/` with numbered subfolders
4. **Seamless flow** - Each step can read from previous step's output
5. **Minimal code changes** - Only path updates, no logic changes

## File Reuse

- ✅ `main/` files → `1. search/main/` (reused)
- ✅ `collect_activating_sentences.py` → `2. autointerp_lite/collect_examples.py` (reused, renamed)
- ✅ `label_features.py` → `2. autointerp_lite/label_features.py` (reused)
- ✅ `extraction/compute_examples_finance.py` → `3. autointerp_advance/extract_examples.py` (reused, renamed)
- ✅ `extraction/generate_labels_from_examples.py` → `3. autointerp_advance/generate_labels.py` (reused, renamed)
- ✅ `main/compute_dashboard.py` → `3. autointerp_advance/compute_dashboard.py` (reused)

## Notes

- Domain-specific files (like `domains/finance/add_labels.py`) remain in their domain folders
- All scripts maintain backward compatibility where possible
- Paths are resolved relative to BASE_DIR for flexibility



