#!/usr/bin/env python3
"""
AutoInterp Light - Simple runner script
"""

import argparse
import subprocess
import sys
from pathlib import Path

def create_sample_texts():
    """Create sample financial and general texts"""
    
    financial_texts = [
        "The company reported quarterly earnings of $2.5 billion, beating analyst expectations.",
        "Stock prices surged 15% following the merger announcement between the two tech giants.",
        "The Federal Reserve raised interest rates by 0.25% to combat inflation.",
        "Bank of America's loan loss provisions increased by $500 million this quarter.",
        "Tesla's market capitalization reached $800 billion after strong delivery numbers.",
        "The cryptocurrency market experienced a 20% correction amid regulatory concerns.",
        "Goldman Sachs reported record trading revenue of $3.2 billion for Q3.",
        "The housing market showed signs of cooling with mortgage rates at 7.5%.",
        "Apple's dividend yield increased to 2.1% following strong cash flow generation.",
        "The S&P 500 index closed at 4,200 points, up 2.3% for the day.",
        "JPMorgan Chase's net interest margin expanded to 2.8% in the current quarter.",
        "Bitcoin's price volatility increased as institutional adoption accelerated.",
        "The unemployment rate dropped to 3.5%, indicating a strong labor market.",
        "Real estate investment trusts (REITs) outperformed the broader market this month.",
        "The consumer price index (CPI) rose 0.3% month-over-month in September."
    ]
    
    general_texts = [
        "The weather forecast predicts sunny skies with temperatures reaching 75 degrees.",
        "Scientists discovered a new species of butterfly in the Amazon rainforest.",
        "The novel explores themes of love, loss, and redemption in modern society.",
        "Cooking pasta requires boiling water and adding salt for proper seasoning.",
        "The museum exhibition features works from the Renaissance period.",
        "Children played in the park while parents watched from nearby benches.",
        "The recipe calls for fresh ingredients and careful preparation techniques.",
        "Music has the power to evoke emotions and create lasting memories.",
        "The library contains thousands of books covering various academic subjects.",
        "Gardening requires patience, knowledge, and regular maintenance of plants.",
        "The movie received critical acclaim for its innovative storytelling approach.",
        "Exercise and proper nutrition are essential for maintaining good health.",
        "The artist used vibrant colors to create a striking visual composition.",
        "Technology continues to evolve and shape our daily lives.",
        "Friendship is built on trust, understanding, and mutual respect."
    ]
    
    # Save to files
    with open("financial_texts.txt", "w") as f:
        for text in financial_texts:
            f.write(text + "\n")
    
    with open("general_texts.txt", "w") as f:
        for text in general_texts:
            f.write(text + "\n")
    
    print("✅ Sample text files created: financial_texts.txt, general_texts.txt")

def run_financial_analysis():
    """Run financial feature analysis"""
    print("="*80)
    print("AUTOINTERP LIGHT - FINANCIAL FEATURE ANALYSIS")
    print("="*80)
    
    # Create sample texts if they don't exist
    if not Path("financial_texts.txt").exists() or not Path("general_texts.txt").exists():
        create_sample_texts()
    
    cmd = [
        sys.executable, "feature_activation_analyzer.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--domain_texts", "financial_texts.txt",
        "--general_texts", "general_texts.txt",
        "--layer_idx", "16",
        "--top_n", "30",
        "--domain_name", "financial",
        "--output_dir", "results"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_custom_analysis():
    """Run custom analysis with user inputs"""
    print("="*80)
    print("AUTOINTERP LIGHT - CUSTOM ANALYSIS")
    print("="*80)
    
    base_model = input("Enter base model name (default: meta-llama/Llama-2-7b-hf): ").strip()
    if not base_model:
        base_model = "meta-llama/Llama-2-7b-hf"
    
    sae_model = input("Enter SAE model path: ").strip()
    if not sae_model:
        sae_model = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    domain_texts = input("Enter domain texts file path: ").strip()
    if not domain_texts:
        domain_texts = "financial_texts.txt"
    
    general_texts = input("Enter general texts file path: ").strip()
    if not general_texts:
        general_texts = "general_texts.txt"
    
    layer_idx = input("Enter layer index (default: 16): ").strip()
    if not layer_idx:
        layer_idx = "16"
    
    top_n = input("Enter number of top features (default: 30): ").strip()
    if not top_n:
        top_n = "30"
    
    domain_name = input("Enter domain name (default: financial): ").strip()
    if not domain_name:
        domain_name = "financial"
    
    cmd = [
        sys.executable, "feature_activation_analyzer.py",
        "--base_model", base_model,
        "--sae_model", sae_model,
        "--domain_texts", domain_texts,
        "--general_texts", general_texts,
        "--layer_idx", layer_idx,
        "--top_n", top_n,
        "--domain_name", domain_name,
        "--output_dir", "results"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="AutoInterp Light Runner")
    parser.add_argument("--mode", choices=["financial", "custom"], default="financial",
                       help="Analysis mode: financial (default) or custom")
    
    args = parser.parse_args()
    
    if args.mode == "financial":
        run_financial_analysis()
    elif args.mode == "custom":
        run_custom_analysis()

if __name__ == "__main__":
    main()
