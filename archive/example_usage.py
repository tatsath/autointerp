#!/usr/bin/env python3
"""
Example Usage Scripts for Generic Feature Analysis System
"""

import subprocess
import sys

def example_1_llama2_7b_financial():
    """Example 1: Llama-2-7B with financial domain analysis"""
    print("="*80)
    print("EXAMPLE 1: Llama-2-7B Financial Analysis")
    print("="*80)
    
    cmd = [
        sys.executable, "generic_master_script.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--top_n", "10",
        "--domain", "financial",
        "--output_dir", "results_llama2_7b_financial"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def example_2_llama2_7b_medical():
    """Example 2: Llama-2-7B with medical domain analysis"""
    print("="*80)
    print("EXAMPLE 2: Llama-2-7B Medical Analysis")
    print("="*80)
    
    cmd = [
        sys.executable, "generic_master_script.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--top_n", "15",
        "--domain", "medical",
        "--output_dir", "results_llama2_7b_medical"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def example_3_custom_model():
    """Example 3: Custom model with different parameters"""
    print("="*80)
    print("EXAMPLE 3: Custom Model Analysis")
    print("="*80)
    
    cmd = [
        sys.executable, "generic_master_script.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/path/to/your/sae/model",
        "--top_n", "20",
        "--layer_idx", "20",
        "--domain", "legal",
        "--labeling_model", "meta-llama/Llama-2-7b-chat-hf",
        "--explainer_model", "openai/gpt-4o",
        "--output_dir", "results_custom_legal"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def example_4_step_by_step():
    """Example 4: Run individual steps"""
    print("="*80)
    print("EXAMPLE 4: Step-by-Step Analysis")
    print("="*80)
    
    # Step 1: Feature Analysis only
    print("Step 1: Feature Analysis")
    cmd1 = [
        sys.executable, "generic_master_script.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--top_n", "10",
        "--steps", "analysis",
        "--output_dir", "results_step_by_step"
    ]
    subprocess.run(cmd1)
    
    # Step 2: Feature Labeling only
    print("Step 2: Feature Labeling")
    cmd2 = [
        sys.executable, "generic_master_script.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--top_n", "10",
        "--steps", "labeling",
        "--output_dir", "results_step_by_step"
    ]
    subprocess.run(cmd2)

def example_5_individual_components():
    """Example 5: Run individual components directly"""
    print("="*80)
    print("EXAMPLE 5: Individual Components")
    print("="*80)
    
    # Run feature analysis directly
    print("Running feature analysis directly...")
    cmd = [
        sys.executable, "generic_feature_analysis.py",
        "--base_model", "meta-llama/Llama-2-7b-hf",
        "--sae_model", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
        "--top_n", "5",
        "--output_dir", "results_individual"
    ]
    subprocess.run(cmd)
    
    # Run feature labeling directly
    print("Running feature labeling directly...")
    cmd = [
        sys.executable, "generic_feature_labeling.py",
        "--analysis_file", "results_individual/top_5_features_analysis.json",
        "--domain", "financial",
        "--output_dir", "results_individual"
    ]
    subprocess.run(cmd)

def main():
    """Main function to run examples"""
    print("Generic Feature Analysis System - Example Usage")
    print("="*80)
    print("Available examples:")
    print("1. Llama-2-7B Financial Analysis")
    print("2. Llama-2-7B Medical Analysis") 
    print("3. Custom Model Analysis")
    print("4. Step-by-Step Analysis")
    print("5. Individual Components")
    print("="*80)
    
    choice = input("Enter example number (1-5) or 'all' to run all examples: ").strip()
    
    if choice == "1":
        example_1_llama2_7b_financial()
    elif choice == "2":
        example_2_llama2_7b_medical()
    elif choice == "3":
        example_3_custom_model()
    elif choice == "4":
        example_4_step_by_step()
    elif choice == "5":
        example_5_individual_components()
    elif choice.lower() == "all":
        print("Running all examples...")
        example_1_llama2_7b_financial()
        example_2_llama2_7b_medical()
        example_4_step_by_step()
        example_5_individual_components()
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
