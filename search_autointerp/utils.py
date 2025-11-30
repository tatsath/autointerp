#!/usr/bin/env python3
"""Utility functions for output directory naming."""

import os
import re
from datetime import datetime


def get_output_dir(base_results_dir: str, step: str, model_path: str, 
                   sae_id: str = None, dataset_path: str = None, 
                   tokens_str_path: str = None, timestamp: str = None):
    """
    Generate descriptive output directory name.
    
    Format: {step}/{model}_{domain}_l{layer}_{timestamp}
    
    Args:
        base_results_dir: Base results directory (e.g., "results")
        step: Step name (e.g., "1_search", "2_labeling_lite", "3_labeling_advance")
        model_path: Model path (e.g., "nvidia/NVIDIA-Nemotron-Nano-9B-v2")
        sae_id: SAE ID (e.g., "blocks.28.hook_resid_post")
        dataset_path: Dataset path (e.g., "jyanimaulik/yahoo_finance_stockmarket_news")
        tokens_str_path: Path to tokens file (e.g., "finance_tokens.json")
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Full output directory path
    """
    # Extract model name
    model_name = model_path.split('/')[-1].lower().replace('-', '_').replace('.', '_')
    
    # Extract domain from dataset or tokens file
    domain = "unknown"
    if tokens_str_path:
        # Try to extract from tokens file path (e.g., "finance_tokens.json" -> "finance")
        tokens_name = os.path.basename(tokens_str_path).replace('_tokens.json', '').replace('.json', '')
        if tokens_name:
            domain = tokens_name
    elif dataset_path:
        # Try to extract from dataset path (e.g., "jyanimaulik/yahoo_finance_stockmarket_news" -> "finance")
        dataset_name = dataset_path.split('/')[-1].lower()
        if 'finance' in dataset_name:
            domain = "finance"
        elif 'code' in dataset_name:
            domain = "code"
        elif 'math' in dataset_name:
            domain = "math"
        # Add more domain detection as needed
    
    # Extract layer from sae_id
    layer = "unknown"
    if sae_id:
        match = re.search(r'blocks\.(\d+)', sae_id)
        if match:
            layer = f"l{match.group(1)}"
    
    # Generate timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory name
    dir_name = f"{model_name}_{domain}_{layer}_{timestamp}"
    output_dir = os.path.join(base_results_dir, step, dir_name)
    
    return output_dir



