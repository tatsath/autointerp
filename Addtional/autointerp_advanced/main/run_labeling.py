#!/usr/bin/env python3
"""
Unified runner for the complete feature labeling pipeline.
Extracts examples and generates labels in one simple function call.
"""

import os
import sys
from pathlib import Path

# Handle imports - works both as package and standalone
try:
    from .extract_examples import extract_examples
    from .generate_labels import generate_labels
except ImportError:
    # If running as standalone script, add parent to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from main.extract_examples import extract_examples
    from main.generate_labels import generate_labels


def run_labeling(
    feature_indices,
    model_path: str,
    sae_path: str,
    dataset_path: str,
    sae_id: str,
    output_dir: str = "results",
    n_samples: int = 1000,
    max_examples_per_feature: int = 20,
    device: str = None,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    label_model_path: str = None,
    max_examples_for_labeling: int = 10,
    feature_scores: list = None
):
    """
    Simple function to run complete labeling pipeline.
    
    Args:
        feature_indices: List of feature indices (integers) or path to JSON file with feature_indices
        model_path: Path to the base model (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        sae_path: Path to SAE model directory
        dataset_path: Path to dataset (e.g., "jyanimaulik/yahoo_finance_stockmarket_news")
        sae_id: SAE hook point identifier (e.g., "blocks.19.hook_resid_post")
        output_dir: Directory to save outputs (default: "results")
        n_samples: Number of dataset samples to process (default: 1000)
        max_examples_per_feature: Max examples to extract per feature (default: 20)
        device: Device to use (default: auto-detect)
        column_name: Dataset column name (default: "text")
        minibatch_size_features: Batch size for features (default: 256)
        minibatch_size_tokens: Batch size for tokens (default: 64)
        label_model_path: Model to use for label generation (default: same as model_path)
        max_examples_for_labeling: Max examples to use per feature in label prompt (default: 10)
        feature_scores: Optional list of scores corresponding to feature_indices (default: None, uses 0.0 for all)
    
    Returns:
        dict: Paths to output files
    """
    import json
    import tempfile
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set default label model path
    if label_model_path is None:
        label_model_path = model_path
    
    # Handle feature_indices - can be a list or a file path
    if isinstance(feature_indices, str) and Path(feature_indices).exists():
        # It's a file path, load it
        with open(feature_indices, 'r') as f:
            feature_data = json.load(f)
        feature_indices = feature_data['feature_indices']
        feature_scores = feature_data.get('scores', [0.0] * len(feature_indices))
    elif isinstance(feature_indices, (list, tuple)):
        # It's already a list
        if feature_scores is None:
            feature_scores = [0.0] * len(feature_indices)
    else:
        raise ValueError("feature_indices must be a list of integers or a path to a JSON file")
    
    # Ensure feature_scores matches feature_indices length
    if len(feature_scores) != len(feature_indices):
        feature_scores = [0.0] * len(feature_indices)
    
    # Create temporary feature list JSON file
    feature_list_data = {
        "feature_indices": feature_indices,
        "scores": feature_scores
    }
    
    # Use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(feature_list_data, f)
        temp_feature_list_path = f.name
    
    try:
        # Step 1: Extract examples
        examples_output_path = output_dir / "feature_examples.jsonl"
        print("=" * 80)
        print("STEP 1: Extracting examples")
        print("=" * 80)
        print(f"Processing {len(feature_indices)} features: {feature_indices[:10]}{'...' if len(feature_indices) > 10 else ''}")
        
        extract_examples(
            model_path=model_path,
            sae_path=sae_path,
            dataset_path=dataset_path,
            feature_list_path=temp_feature_list_path,
            output_path=str(examples_output_path),
            sae_id=sae_id,
            column_name=column_name,
            minibatch_size_features=minibatch_size_features,
            minibatch_size_tokens=minibatch_size_tokens,
            n_samples=n_samples,
            max_examples_per_feature=max_examples_per_feature,
            device=device
        )
        
        # Step 2: Generate labels
        labels_output_path = output_dir / "feature_labels.json"
        print("\n" + "=" * 80)
        print("STEP 2: Generating labels")
        print("=" * 80)
        
        generate_labels(
            examples_jsonl_path=str(examples_output_path),
            output_path=str(labels_output_path),
            model_path=label_model_path,
            max_examples_per_feature=max_examples_for_labeling,
            use_same_model=True
        )
        
        print("\n" + "=" * 80)
        print("Pipeline complete!")
        print("=" * 80)
        print(f"Examples saved to: {examples_output_path}")
        print(f"Labels saved to: {labels_output_path}")
        
        return {
            'examples_path': str(examples_output_path),
            'labels_path': str(labels_output_path)
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_feature_list_path):
            os.unlink(temp_feature_list_path)

