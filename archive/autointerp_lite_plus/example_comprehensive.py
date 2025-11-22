#!/usr/bin/env python3
"""
Example script demonstrating AutoInterp Lite Plus comprehensive analysis
"""

import subprocess
import sys
from pathlib import Path

def run_comprehensive_analysis():
    """Run comprehensive analysis with example data"""
    
    print("🚀 AutoInterp Lite Plus - Comprehensive Analysis Example")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("financial_texts.txt").exists():
        print("❌ Please run this script from the autointerp_lite_plus directory")
        return
    
    # Run comprehensive analysis
    cmd = [
        "python", "run_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_finance_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--domain_data", "financial_texts.txt",
        "--general_data", "general_texts.txt",
        "--top_n", "5",
        "--comprehensive"
    ]
    
    print("Running comprehensive analysis...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Comprehensive analysis completed successfully!")
        print("\n📊 Key Features of AutoInterp Lite Plus:")
        print("  • F1, Precision, Recall metrics")
        print("  • Clustering and Polysemanticity analysis")
        print("  • Robust conceptual labeling")
        print("  • Selectivity and robustness metrics")
        print("  • Comprehensive results saved to results/ directory")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(run_comprehensive_analysis())
