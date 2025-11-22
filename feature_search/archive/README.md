# Archive

This folder contains files that are not needed for the main feature search functionality but are kept for reference.

## Archived Files

- **`prepare_openthoughts_subset.py`**: Helper script for preparing OpenThoughts dataset subsets. Not needed for core feature search.

- **`start_vllm_server.sh`**: Script to start vLLM server for feature labeling. Not needed for core feature search functionality.

- **`domain_tokens.json`**: Example token file. Users should use domain-specific token files in their domain folders.

- **`scripts/`**: Old individual shell scripts that have been replaced by the unified `run_feature_search.py` interface:
  - `compute_score.sh`: Replaced by `run_feature_search.py`
  - `compute_dashboard.sh`: Replaced by `run_feature_search.py`

## Main Entry Point

Use `run_feature_search.py` in the parent directory for all feature search operations.

