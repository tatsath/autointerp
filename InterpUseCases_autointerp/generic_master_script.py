#!/usr/bin/env python3
"""
Generic Master Script
Orchestrates the complete feature analysis pipeline for any model and SAE
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json

class GenericMasterScript:
    def __init__(self, base_model_name, sae_model_path, output_dir="results"):
        """
        Initialize the master script
        
        Args:
            base_model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            sae_model_path: Path to the SAE model directory
            output_dir: Directory to save results
        """
        self.base_model_name = base_model_name
        self.sae_model_path = sae_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract model name for run naming
        self.model_name = base_model_name.split('/')[-1].replace('-', '_')
        self.sae_name = Path(sae_model_path).name
    
    def run_feature_analysis(self, top_n=10, layer_idx=16):
        """Run feature analysis to find top features"""
        print("="*80)
        print("STEP 1: FEATURE ANALYSIS")
        print("="*80)
        
        cmd = [
            sys.executable, "generic_feature_analysis.py",
            "--base_model", self.base_model_name,
            "--sae_model", self.sae_model_path,
            "--top_n", str(top_n),
            "--layer_idx", str(layer_idx),
            "--output_dir", str(self.output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Feature analysis completed successfully!")
            return True
        else:
            print("❌ Feature analysis failed!")
            print("STDERR:", result.stderr)
            return False
    
    def run_feature_labeling(self, domain="financial", labeling_model="meta-llama/Llama-2-7b-chat-hf"):
        """Run feature labeling"""
        print("="*80)
        print("STEP 2: FEATURE LABELING")
        print("="*80)
        
        # Find the analysis file
        analysis_files = list(self.output_dir.glob("top_*_features_analysis.json"))
        if not analysis_files:
            print("❌ No analysis file found!")
            return False
        
        analysis_file = analysis_files[0]  # Take the first one
        
        cmd = [
            sys.executable, "generic_feature_labeling.py",
            "--analysis_file", str(analysis_file),
            "--labeling_model", labeling_model,
            "--domain", domain,
            "--output_dir", str(self.output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Feature labeling completed successfully!")
            return True
        else:
            print("❌ Feature labeling failed!")
            print("STDERR:", result.stderr)
            return False
    
    def run_delphi_analysis(self, top_n=10, explainer_model="openai/gpt-3.5-turbo"):
        """Run Delphi analysis"""
        print("="*80)
        print("STEP 3: DELPHI ANALYSIS")
        print("="*80)
        
        # Load feature numbers from analysis
        analysis_files = list(self.output_dir.glob("top_*_features_analysis.json"))
        if not analysis_files:
            print("❌ No analysis file found!")
            return False
        
        with open(analysis_files[0], 'r') as f:
            analysis_data = json.load(f)
        
        feature_numbers = [f['feature_number'] for f in analysis_data['top_features'][:top_n]]
        features_str = ','.join(map(str, feature_numbers))
        
        run_name = f"{self.model_name}_{self.sae_name}_top{top_n}"
        
        cmd = [
            sys.executable, "generic_delphi_runner.py",
            "--base_model", self.base_model_name,
            "--sae_model", self.sae_model_path,
            "--features", features_str,
            "--run_name", run_name,
            "--explainer_model", explainer_model,
            "--output_dir", str(self.output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Delphi analysis completed successfully!")
            return True
        else:
            print("❌ Delphi analysis failed!")
            print("STDERR:", result.stderr)
            return False
    
    def run_comparison(self, domain="financial"):
        """Run comparison analysis"""
        print("="*80)
        print("STEP 4: COMPARISON ANALYSIS")
        print("="*80)
        
        # Find the files
        labeling_file = f"feature_labels_detailed_{domain}.csv"
        delphi_file = f"{self.model_name}_{self.sae_name}_top10_delphi_results.csv"
        
        # Check if files exist
        if not (self.output_dir / labeling_file).exists():
            print(f"❌ Labeling file not found: {labeling_file}")
            return False
        
        if not (self.output_dir / delphi_file).exists():
            print(f"❌ Delphi file not found: {delphi_file}")
            return False
        
        cmd = [
            sys.executable, "generic_comparison.py",
            "--labeling_file", labeling_file,
            "--delphi_file", delphi_file,
            "--output_dir", str(self.output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Comparison analysis completed successfully!")
            return True
        else:
            print("❌ Comparison analysis failed!")
            print("STDERR:", result.stderr)
            return False
    
    def run_complete_pipeline(self, top_n=10, layer_idx=16, domain="financial", 
                            labeling_model="meta-llama/Llama-2-7b-chat-hf",
                            explainer_model="openai/gpt-3.5-turbo"):
        """Run the complete pipeline"""
        print("="*80)
        print("GENERIC MASTER SCRIPT - COMPLETE PIPELINE")
        print("="*80)
        print(f"Base Model: {self.base_model_name}")
        print(f"SAE Model: {self.sae_model_path}")
        print(f"Top Features: {top_n}")
        print(f"Domain: {domain}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80)
        
        # Step 1: Feature Analysis
        if not self.run_feature_analysis(top_n, layer_idx):
            print("❌ Pipeline failed at feature analysis step")
            return False
        
        # Step 2: Feature Labeling
        if not self.run_feature_labeling(domain, labeling_model):
            print("❌ Pipeline failed at feature labeling step")
            return False
        
        # Step 3: Delphi Analysis
        if not self.run_delphi_analysis(top_n, explainer_model):
            print("❌ Pipeline failed at Delphi analysis step")
            return False
        
        # Step 4: Comparison
        if not self.run_comparison(domain):
            print("❌ Pipeline failed at comparison step")
            return False
        
        print("="*80)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("="*80)
        print(f"Results saved in: {self.output_dir}")
        print("\nGenerated files:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Generic Master Script for Feature Analysis")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--sae_model", required=True, help="Path to SAE model directory")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top features to analyze")
    parser.add_argument("--layer_idx", type=int, default=16, help="Layer index for feature extraction")
    parser.add_argument("--domain", default="financial", help="Domain for analysis (financial, medical, legal, etc.)")
    parser.add_argument("--labeling_model", default="meta-llama/Llama-2-7b-chat-hf", help="Model for labeling")
    parser.add_argument("--explainer_model", default="openai/gpt-3.5-turbo", help="Model for Delphi explanations")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--steps", nargs='+', choices=['analysis', 'labeling', 'delphi', 'comparison'], 
                       help="Specific steps to run (default: all)")
    
    args = parser.parse_args()
    
    # Create master script
    master = GenericMasterScript(
        base_model_name=args.base_model,
        sae_model_path=args.sae_model,
        output_dir=args.output_dir
    )
    
    # Run specific steps or complete pipeline
    if args.steps:
        success = True
        if 'analysis' in args.steps:
            success &= master.run_feature_analysis(args.top_n, args.layer_idx)
        if 'labeling' in args.steps and success:
            success &= master.run_feature_labeling(args.domain, args.labeling_model)
        if 'delphi' in args.steps and success:
            success &= master.run_delphi_analysis(args.top_n, args.explainer_model)
        if 'comparison' in args.steps and success:
            success &= master.run_comparison(args.domain)
        
        if success:
            print("🎉 Selected steps completed successfully!")
        else:
            print("❌ Some steps failed!")
    else:
        # Run complete pipeline
        master.run_complete_pipeline(
            top_n=args.top_n,
            layer_idx=args.layer_idx,
            domain=args.domain,
            labeling_model=args.labeling_model,
            explainer_model=args.explainer_model
        )

if __name__ == "__main__":
    main()
