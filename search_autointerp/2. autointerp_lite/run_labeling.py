#!/usr/bin/env python3
"""
Unified entry point for basic labeling pipeline.
Chains: collect_examples.py → label_features.py
"""

import os
import sys
import argparse
import subprocess
import json
import re
from datetime import datetime

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_output_dir

def main():
    parser = argparse.ArgumentParser(description="Run basic labeling pipeline")
    parser.add_argument("--search_output", type=str, 
                       default="../results/1_search",
                       help="Path to search output directory (default: ../results/1_search)")
    parser.add_argument("--output_dir", type=str,
                       default="../results/2_labeling_lite",
                       help="Path to output directory (default: ../results/2_labeling_lite)")
    parser.add_argument("--skip_collect", action="store_true",
                       help="Skip collecting examples (use existing activating_sentences.json)")
    parser.add_argument("--skip_label", action="store_true",
                       help="Skip labeling (only collect examples)")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    base_results_dir = os.path.join(base_dir, "results")
    
    # Auto-generate descriptive output directory if generic path detected
    if args.output_dir.endswith("2_labeling_lite") or args.output_dir == os.path.join(base_results_dir, "2_labeling_lite"):
        # Try to extract info from search output or feature_list.json
        feature_list_path = os.path.join(args.search_output, "feature_list.json")
        model_path = "unknown"
        sae_id = None
        dataset_path = None
        tokens_str_path = None
        
        # Try to read config from search output if available
        config_path = os.path.join(args.search_output, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                model_path = config.get("model_path", model_path)
                sae_id = config.get("sae_id", sae_id)
                dataset_path = config.get("dataset_path", dataset_path)
                tokens_str_path = config.get("tokens_str_path", tokens_str_path)
        
        args.output_dir = get_output_dir(base_results_dir, "2_labeling_lite", model_path, sae_id, dataset_path, tokens_str_path)
        print(f">>> Dynamic output directory created: {args.output_dir}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect examples
    if not args.skip_collect:
        print("=" * 80)
        print("Step 1: Collecting Activating Examples")
        print("=" * 80)
        print()
        
        # Update environment to point to correct search output and output directory
        env = os.environ.copy()
        env['SEARCH_OUTPUT_DIR'] = args.search_output
        env['OUTPUT_DIR'] = args.output_dir
        
        collect_script = os.path.join(script_dir, "collect_examples.py")
        result = subprocess.run([sys.executable, collect_script], env=env)
        
        if result.returncode != 0:
            print("❌ Failed to collect examples")
            return 1
        
        print()
        print("✅ Examples collected successfully!")
        print()
    
    # Step 2: Generate labels
    if not args.skip_label:
        print("=" * 80)
        print("Step 2: Generating Labels")
        print("=" * 80)
        print()
        
        # Update environment to point to correct input/output directories
        env = os.environ.copy()
        env['FEATURE_LIST_JSON'] = os.path.join(args.search_output, "feature_list.json")
        env['ACTIVATING_CONTEXTS_JSON'] = os.path.join(args.output_dir, "activating_sentences.json")
        env['OUTPUT_JSON'] = os.path.join(args.output_dir, "feature_labels.json")
        
        label_script = os.path.join(script_dir, "label_features.py")
        result = subprocess.run([sys.executable, label_script], cwd=script_dir, env=env)
        
        if result.returncode != 0:
            print("❌ Failed to generate labels")
            return 1
        
        print()
        print("✅ Labels generated successfully!")
        print()
    
    print("=" * 80)
    print("Labeling Pipeline Complete!")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

