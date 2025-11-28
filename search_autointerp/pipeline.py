#!/usr/bin/env python3
"""
Seamless pipeline script that chains all three steps:
1. Search → 2. Labeling (Lite) → 3. Labeling (Advanced)
"""

import os
import sys
import argparse
import subprocess
import json

def run_pipeline(config_path=None, **kwargs):
    """
    Run complete pipeline: search → labeling_lite → labeling_advance
    
    Args:
        config_path: Path to JSON config file (optional)
        **kwargs: Command-line arguments override config
    """
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Override with kwargs
    config.update(kwargs)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "1_search"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "2_labeling_lite"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "3_labeling_advance"), exist_ok=True)
    
    print("=" * 80)
    print("Feature Search & Labeling Pipeline")
    print("=" * 80)
    print()
    
    # Step 1: Search
    print("🔍 Step 1: Feature Search")
    print("-" * 80)
    search_dir = os.path.join(base_dir, "1. search")
    search_output = os.path.join(results_dir, "1_search")
    
    if not config.get("skip_search", False):
        search_script = os.path.join(search_dir, "main", "run_feature_search.py")
        
        # Build command
        cmd = [sys.executable, search_script]
        required_params = ["model_path", "sae_path", "dataset_path"]
        
        for param in required_params:
            if param not in config:
                print(f"❌ Error: Missing required parameter: {param}")
                return 1
            cmd.extend([f"--{param}", config[param]])
        
        # Optional parameters
        optional_params = {
            "tokens_str_path": "--tokens_str_path",
            "sae_id": "--sae_id",
            "score_type": "--score_type",
            "num_features": "--num_features",
            "n_samples": "--n_samples",
            "expand_range": "--expand_range"
        }
        
        for key, flag in optional_params.items():
            if key in config:
                cmd.extend([flag, str(config[key])])
        
        cmd.extend(["--output_dir", search_output])
        
        result = subprocess.run(cmd, cwd=search_dir)
        
        if result.returncode != 0:
            print("❌ Search failed")
            return 1
        
        print("✅ Search completed!")
        print()
    else:
        print("⏭️  Skipping search (skip_search=True)")
        print()
    
    # Step 2: Labeling Lite
    if not config.get("skip_labeling_lite", False):
        print("🏷️  Step 2: Basic Labeling (Lite)")
        print("-" * 80)
        
        labeling_dir = os.path.join(base_dir, "2. autointerp_lite")
        labeling_output = os.path.join(results_dir, "2_labeling_lite")
        
        labeling_script = os.path.join(labeling_dir, "run_labeling.py")
        cmd = [
            sys.executable, labeling_script,
            "--search_output", search_output,
            "--output_dir", labeling_output
        ]
        
        result = subprocess.run(cmd, cwd=labeling_dir)
        
        if result.returncode != 0:
            print("❌ Labeling (Lite) failed")
            return 1
        
        print("✅ Labeling (Lite) completed!")
        print()
    else:
        print("⏭️  Skipping labeling lite (skip_labeling_lite=True)")
        print()
    
    # Step 3: Labeling Advanced
    if not config.get("skip_labeling_advance", False):
        print("🏷️  Step 3: Advanced Labeling")
        print("-" * 80)
        
        advance_dir = os.path.join(base_dir, "3. autointerp_advance")
        advance_output = os.path.join(results_dir, "3_labeling_advance")
        
        advance_script = os.path.join(advance_dir, "run_labeling_advanced.py")
        
        cmd = [
            sys.executable, advance_script,
            "--model_path", config.get("model_path", ""),
            "--sae_path", config.get("sae_path", ""),
            "--dataset_path", config.get("dataset_path", ""),
            "--search_output", search_output,
            "--output_dir", advance_output,
            "--n_samples", str(config.get("n_samples", 5000)),
            "--max_examples_per_feature", str(config.get("max_examples_per_feature", 20))
        ]
        
        if "sae_id" in config:
            cmd.extend(["--sae_id", config["sae_id"]])
        
        result = subprocess.run(cmd, cwd=advance_dir)
        
        if result.returncode != 0:
            print("❌ Labeling (Advanced) failed")
            return 1
        
        print("✅ Labeling (Advanced) completed!")
        print()
    else:
        print("⏭️  Skipping labeling advance (skip_labeling_advance=True)")
        print()
    
    print("=" * 80)
    print("🎉 Pipeline Complete!")
    print("=" * 80)
    print(f"Results saved to: {results_dir}")
    print(f"  • Search: {os.path.join(results_dir, '1_search')}")
    print(f"  • Labeling Lite: {os.path.join(results_dir, '2_labeling_lite')}")
    print(f"  • Labeling Advanced: {os.path.join(results_dir, '3_labeling_advance')}")
    print()
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete feature search and labeling pipeline")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--sae_path", type=str, help="SAE path")
    parser.add_argument("--dataset_path", type=str, help="Dataset path")
    parser.add_argument("--sae_id", type=str, help="SAE ID")
    parser.add_argument("--tokens_str_path", type=str, help="Tokens file path")
    parser.add_argument("--score_type", type=str, default="fisher", help="Score type")
    parser.add_argument("--num_features", type=int, default=20, help="Number of features")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--skip_search", action="store_true", help="Skip search step")
    parser.add_argument("--skip_labeling_lite", action="store_true", help="Skip labeling lite")
    parser.add_argument("--skip_labeling_advance", action="store_true", help="Skip labeling advance")
    
    args = parser.parse_args()
    
    # Convert args to dict, filtering out None values
    config = {k: v for k, v in vars(args).items() if v is not None}
    
    sys.exit(run_pipeline(**config))

