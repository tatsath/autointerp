#!/usr/bin/env python3
"""
Simple script to run multi-layer financial feature analysis
"""

import subprocess
import sys

def run_complete_analysis():
    """Run complete analysis for all layers"""
    print("="*80)
    print("RUNNING COMPLETE MULTI-LAYER FINANCIAL ANALYSIS")
    print("="*80)
    
    cmd = [
        sys.executable, "multi_layer_financial_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--layers", "4", "10", "16", "22", "28",
        "--top_n", "30",
        "--output_dir", "complete_financial_analysis",
        "--min_specialization", "0.1",
        "--top_features_n", "50"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_fast_analysis():
    """Run fast analysis for key layers only"""
    print("="*80)
    print("RUNNING FAST MULTI-LAYER FINANCIAL ANALYSIS")
    print("="*80)
    
    cmd = [
        sys.executable, "multi_layer_financial_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--layers", "16", "22",
        "--top_n", "30",
        "--output_dir", "fast_financial_analysis",
        "--min_specialization", "0.1",
        "--top_features_n", "30"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_single_layer_analysis():
    """Run analysis for single layer"""
    print("="*80)
    print("RUNNING SINGLE LAYER FINANCIAL ANALYSIS")
    print("="*80)
    
    cmd = [
        sys.executable, "multi_layer_financial_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--layers", "16",
        "--top_n", "30",
        "--output_dir", "single_layer_analysis",
        "--min_specialization", "0.1",
        "--top_features_n", "30"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_custom_analysis():
    """Run custom analysis with user-specified layers"""
    print("="*80)
    print("RUNNING CUSTOM MULTI-LAYER FINANCIAL ANALYSIS")
    print("="*80)
    
    layers = input("Enter layer numbers (space-separated, e.g., 4 10 16): ").strip()
    layer_list = layers.split()
    
    cmd = [
        sys.executable, "multi_layer_financial_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--layers"] + layer_list + [
        "--top_n", "30",
        "--output_dir", "custom_financial_analysis",
        "--min_specialization", "0.1",
        "--top_features_n", "50"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    print("="*80)
    print("MULTI-LAYER FINANCIAL FEATURE ANALYSIS")
    print("="*80)
    print("This script analyzes top 30 features across multiple layers")
    print("of the Llama-2-7B SAE model to identify financial features.")
    print("="*80)
    
    print("\nAvailable options:")
    print("1. Complete analysis (all layers: 4, 10, 16, 22, 28)")
    print("2. Fast analysis (layers 16, 22 only)")
    print("3. Single layer analysis (layer 16 only)")
    print("4. Custom analysis (you specify layers)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_complete_analysis()
    elif choice == "2":
        run_fast_analysis()
    elif choice == "3":
        run_single_layer_analysis()
    elif choice == "4":
        run_custom_analysis()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()