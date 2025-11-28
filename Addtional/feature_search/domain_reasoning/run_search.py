#!/usr/bin/env python3
"""
Reasoning domain feature search script.
Loads configuration from config.yaml and runs feature search.
"""

import os
import sys
import yaml
import json
import fire

# Add domain_common directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
common_dir = os.path.join(os.path.dirname(script_dir), "domain_common")
if common_dir not in sys.path:
    sys.path.insert(0, common_dir)

from run_feature_search import run_feature_search


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve relative paths
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    # Convert tokens.txt to JSON format for run_feature_search
    tokens_path = config.get('tokens_str_path', 'tokens.txt')
    if not os.path.isabs(tokens_path):
        tokens_path = os.path.join(config_dir, tokens_path)
    
    # Read tokens.txt and convert to JSON format
    if os.path.exists(tokens_path):
        with open(tokens_path, 'r') as f:
            tokens = [line.strip() for line in f if line.strip()]
        # Save as temporary JSON file
        tokens_json_path = tokens_path.replace('.txt', '.json')
        with open(tokens_json_path, 'w') as f:
            json.dump(tokens, f, indent=2)
        config['tokens_str_path'] = tokens_json_path
    
    # Resolve output_dir relative to domain folder
    if 'output_dir' in config and not os.path.isabs(config['output_dir']):
        config['output_dir'] = os.path.join(config_dir, config['output_dir'])
        os.makedirs(config['output_dir'], exist_ok=True)
    
    return config


def run_reasoning_search(
    model_path: str = None,
    sae_path: str = None,
    sae_id: str = None,
    config_path: str = None,
    **kwargs
):
    """
    Run reasoning feature search with configuration from config.yaml.
    
    Args:
        model_path: Override model path from config
        sae_path: Override SAE path from config
        sae_id: Override SAE ID from config
        config_path: Path to config.yaml (default: reasoning/config.yaml)
        **kwargs: Additional arguments passed to run_feature_search
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override with command-line arguments if provided
    if model_path is None:
        model_path = config['model_path']
    if sae_path is None:
        sae_path = config.get('sae_path')
        if not sae_path:
            raise ValueError("sae_path must be provided either in config or as argument")
    if sae_id is None:
        sae_id = config.get('sae_id')
    
    # Prepare arguments for run_feature_search
    search_args = {
        'model_path': model_path,
        'sae_path': sae_path,
        'dataset_path': config['dataset_path'],
        'tokens_str_path': config['tokens_str_path'],
        'output_dir': config['output_dir'],
        'sae_id': sae_id,
        'expand_range': tuple(config.get('expand_range', [1, 2])),
        'ignore_tokens': config.get('ignore_tokens'),
        'n_samples': config.get('n_samples', 4096),
        'alpha': config.get('alpha', 1.0),
        'score_type': config.get('score_type', 'simple'),
        'num_features': config.get('num_features', 100),
        'selection_method': config.get('selection_method', 'topk'),
        'quantile_threshold': config.get('quantile_threshold', 0.95),
    }
    
    # Override with kwargs
    search_args.update(kwargs)
    
    # Run feature search
    return run_feature_search(**search_args)


if __name__ == "__main__":
    fire.Fire(run_reasoning_search)

