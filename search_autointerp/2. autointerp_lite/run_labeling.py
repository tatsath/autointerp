#!/usr/bin/env python3
"""
Unified entry point for basic labeling pipeline.
Chains: collect_examples.py → label_features.py
"""

import os
import sys
import argparse
import subprocess

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
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Collect examples
    if not args.skip_collect:
        print("=" * 80)
        print("Step 1: Collecting Activating Examples")
        print("=" * 80)
        print()
        
        # Update environment to point to correct search output
        env = os.environ.copy()
        env['SEARCH_OUTPUT_DIR'] = args.search_output
        
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
        
        label_script = os.path.join(script_dir, "label_features.py")
        result = subprocess.run([sys.executable, label_script], cwd=script_dir)
        
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

