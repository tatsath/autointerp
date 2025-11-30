#!/usr/bin/env python3
"""
Unified entry point for advanced labeling pipeline.
Chains: extract_examples.py → generate_labels.py
"""

import os
import sys
import argparse
import subprocess
import fire
import json

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_output_dir

def run_labeling_advanced(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    sae_id: str = None,
    search_output: str = "../results/1_search",
    output_dir: str = "../results/3_labeling_advance",
    n_samples: int = 5000,
    max_examples_per_feature: int = 20,
    minibatch_size_features: int = 64,
    minibatch_size_tokens: int = 32,
    skip_extract: bool = False,
    skip_label: bool = False
):
    """
    Run advanced labeling pipeline.
    
    Args:
        model_path: Path to model
        sae_path: Path to SAE
        dataset_path: Path to dataset
        sae_id: SAE identifier
        search_output: Path to search output directory
        output_dir: Path to output directory
        n_samples: Number of samples to process
        max_examples_per_feature: Max examples per feature
        skip_extract: Skip extraction step
        skip_label: Skip labeling step
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    base_results_dir = os.path.join(base_dir, "results")
    
    # Auto-generate descriptive output directory if generic path detected
    if output_dir.endswith("3_labeling_advance") or output_dir == os.path.join(base_results_dir, "3_labeling_advance"):
        # Try to extract info from search output
        config_path = os.path.join(search_output, "config.json")
        model_path_from_config = model_path
        sae_path_from_config = None
        sae_id_from_config = sae_id
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_path_from_config = config.get("model_path", model_path)
                sae_path_from_config = config.get("sae_path", None)
                sae_id_from_config = config.get("sae_id", sae_id)
        
        output_dir = get_output_dir(base_results_dir, "3_labeling_advance", model_path_from_config, sae_id_from_config, dataset_path, None)
        print(f">>> Dynamic output directory created: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    feature_list_path = os.path.join(search_output, "feature_list.json")
    if not os.path.exists(feature_list_path):
        print(f"❌ Error: Feature list not found at {feature_list_path}")
        print("   Please run search first (1. search/run_search.py)")
        return 1
    
    examples_output = os.path.join(output_dir, "feature_examples.jsonl")
    labels_output = os.path.join(output_dir, "feature_labels.json")
    
    # Step 1: Extract examples
    if not skip_extract:
        print("=" * 80)
        print("Step 1: Extracting Examples using SaeVisRunner")
        print("=" * 80)
        print()
        
        extract_script = os.path.join(script_dir, "extract_examples.py")
        cmd = [
            sys.executable, extract_script,
            "--model_path", model_path,
            "--sae_path", sae_path,
            "--dataset_path", dataset_path,
            "--feature_list_path", feature_list_path,
            "--output_path", examples_output,
            "--n_samples", str(n_samples),
            "--max_examples_per_feature", str(max_examples_per_feature),
            "--minibatch_size_features", str(minibatch_size_features),
            "--minibatch_size_tokens", str(minibatch_size_tokens)
        ]
        
        if sae_id:
            cmd.extend(["--sae_id", sae_id])
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print("❌ Failed to extract examples")
            return 1
        
        print()
        print("✅ Examples extracted successfully!")
        print()
    
    # Step 2: Generate labels
    if not skip_label:
        print("=" * 80)
        print("Step 2: Generating Labels from Examples")
        print("=" * 80)
        print()
        
        if not os.path.exists(examples_output):
            print(f"❌ Error: Examples file not found at {examples_output}")
            return 1
        
        generate_script = os.path.join(script_dir, "generate_labels.py")
        cmd = [
            sys.executable, generate_script,
            "--examples_jsonl_path", examples_output,
            "--output_path", labels_output,
            "--model_path", model_path,
            "--max_examples_per_feature", str(max_examples_per_feature)
        ]
        
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print("❌ Failed to generate labels")
            return 1
        
        print()
        print("✅ Labels generated successfully!")
        print()
    
    print("=" * 80)
    print("Advanced Labeling Pipeline Complete!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print()
    
    return 0

if __name__ == "__main__":
    fire.Fire(run_labeling_advanced)

