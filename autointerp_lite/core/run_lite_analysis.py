#!/usr/bin/env python3
"""
AutoInterp Lite - Flexible Command Line Interface
Supports configurable prompts, top X features, and optional LLM-based labeling
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

from .feature_activation_analyzer import FeatureActivationAnalyzer

class AutoInterpLiteRunner:
    def __init__(self):
        self.default_config = {
            "base_model": "meta-llama/Llama-2-7b-hf",
            "sae_model": "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun",
            "layer_idx": 16,
            "top_n": 10,
            "domain_name": "financial",
            "output_dir": "results",
            "enable_labeling": False,
            "labeling_model": "meta-llama/Llama-2-7b-chat-hf",
            "labeling_provider": "offline",
            "prompt_file": None
        }
    
    def load_prompt_from_file(self, prompt_file: str) -> str:
        """Load custom prompt from file"""
        try:
            with open(prompt_file, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"❌ Prompt file not found: {prompt_file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading prompt file: {e}")
            sys.exit(1)
    
    def get_default_prompt(self) -> str:
        """Get default analysis prompt"""
        return """Analyze the following feature activations and provide a concise label (2-4 words) for what this feature detects:

Domain texts where feature activates strongly:
{domain_examples}

General texts where feature activates weakly:
{general_examples}

Feature specialization score: {specialization:.2f}

Provide a clear, concise label for this feature:"""
    
    def generate_llm_labels(self, features_df, domain_texts: List[str], general_texts: List[str], 
                          layer_idx: int, labeling_model: str, labeling_provider: str, 
                          prompt: str) -> List[str]:
        """Generate labels using LLM"""
        print(f"🤖 Generating labels using {labeling_model} ({labeling_provider})...")
        
        try:
            if labeling_provider == "offline":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                
                # Load model
                tokenizer = AutoTokenizer.from_pretrained(labeling_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    labeling_model,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                model.eval()
                
                labels = []
                for _, row in features_df.iterrows():
                    # Create prompt with examples
                    domain_examples = "\n".join(domain_texts[:3])  # Top 3 examples
                    general_examples = "\n".join(general_texts[:3])
                    
                    formatted_prompt = prompt.format(
                        domain_examples=domain_examples,
                        general_examples=general_examples,
                        specialization=row['specialization']
                    )
                    
                    # Generate label
                    inputs = tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    label = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    labels.append(label)
                
                return labels
                
            elif labeling_provider == "openrouter":
                import requests
                
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    print("❌ OPENROUTER_API_KEY environment variable not set")
                    return ["LLM labeling failed"] * len(features_df)
                
                labels = []
                for _, row in features_df.iterrows():
                    domain_examples = "\n".join(domain_texts[:3])
                    general_examples = "\n".join(general_texts[:3])
                    
                    formatted_prompt = prompt.format(
                        domain_examples=domain_examples,
                        general_examples=general_examples,
                        specialization=row['specialization']
                    )
                    
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": labeling_model,
                            "messages": [{"role": "user", "content": formatted_prompt}],
                            "max_tokens": 20,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        label = response.json()["choices"][0]["message"]["content"].strip()
                        labels.append(label)
                    else:
                        labels.append("API error")
                
                return labels
                
        except Exception as e:
            print(f"❌ Error in LLM labeling: {e}")
            return ["Labeling failed"] * len(features_df)
    
    def run_analysis(self, args):
        """Run the complete analysis"""
        print("="*80)
        print("AUTOINTERP LITE - FLEXIBLE ANALYSIS")
        print("="*80)
        
        # Load texts
        try:
            with open(args.domain_data, 'r') as f:
                domain_texts = [line.strip() for line in f if line.strip()]
            
            with open(args.general_data, 'r') as f:
                general_texts = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as e:
            print(f"❌ Data file not found: {e}")
            sys.exit(1)
        
        print(f"📊 Domain texts: {len(domain_texts)}")
        print(f"📊 General texts: {len(general_texts)}")
        print(f"🎯 Top features: {args.top_n}")
        print(f"🏷️  Labeling: {'Enabled' if args.enable_labeling else 'Disabled'}")
        if args.enable_labeling:
            print(f"🤖 Labeling model: {args.labeling_model}")
        
        # Create analyzer
        analyzer = FeatureActivationAnalyzer(
            base_model_name=args.base_model,
            sae_model_path=args.sae_model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_dir=args.output_dir
        )
        
        # Run feature analysis
        print("\n🔍 Analyzing feature activations...")
        top_features = analyzer.analyze_domain_features(
            domain_texts=domain_texts,
            general_texts=general_texts,
            layer_idx=args.layer_idx,
            top_n=args.top_n
        )
        
        # Generate labels
        if args.enable_labeling:
            # Load prompt
            if args.prompt_file:
                prompt = self.load_prompt_from_file(args.prompt_file)
                print(f"📝 Using custom prompt from: {args.prompt_file}")
            else:
                prompt = self.get_default_prompt()
                print("📝 Using default prompt")
            
            # Generate LLM labels
            llm_labels = self.generate_llm_labels(
                top_features, domain_texts, general_texts, args.layer_idx,
                args.labeling_model, args.labeling_provider, prompt
            )
            
            # Add LLM labels
            top_features = top_features.copy()
            top_features['llm_label'] = llm_labels
        else:
            # Use simple heuristic labels
            top_features_with_labels = analyzer.generate_feature_labels(
                top_features, domain_texts, general_texts, args.layer_idx
            )
            top_features = top_features_with_labels
        
        # Save results in results folder
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_file = results_dir / f"features_layer{args.layer_idx}.csv"
        top_features.to_csv(results_file, index=False)
        
        # Save summary
        summary = {
            'domain': args.domain_name,
            'layer': args.layer_idx,
            'base_model': args.base_model,
            'sae_model': args.sae_model,
            'top_n': args.top_n,
            'enable_labeling': args.enable_labeling,
            'labeling_model': args.labeling_model if args.enable_labeling else None,
            'labeling_provider': args.labeling_provider if args.enable_labeling else None,
            'prompt_file': args.prompt_file,
            'total_features_analyzed': len(top_features),
            'best_feature': int(top_features.iloc[0]['feature_number']),
            'best_specialization': float(top_features.iloc[0]['specialization']),
            'avg_specialization': float(top_features['specialization'].mean()),
            'results_file': str(results_file)
        }
        
        summary_file = results_dir / f"summary_layer{args.layer_idx}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📋 Summary saved to: {summary_file}")
        
        # Display top results
        print(f"\n🏆 Top {min(5, len(top_features))} Features:")
        print("-" * 80)
        for i, (_, row) in enumerate(top_features.head(5).iterrows()):
            label_col = 'llm_label' if args.enable_labeling and 'llm_label' in top_features.columns else 'label'
            label = row.get(label_col, 'No label')
            print(f"{i+1:2d}. Feature {row['feature_number']:3d} | Spec: {row['specialization']:6.2f} | {label}")
        
        return top_features, summary

def main():
    parser = argparse.ArgumentParser(
        description="AutoInterp Lite - Flexible Feature Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with top 10 features
  python run_lite_analysis.py --domain_texts financial.txt --general_texts general.txt --top_n 10
  
  # With LLM labeling using offline model
  python run_lite_analysis.py --domain_texts financial.txt --general_texts general.txt --top_n 10 --enable_labeling --labeling_model "Qwen/Qwen2.5-7B-Instruct"
  
  # With custom prompt file
  python run_lite_analysis.py --domain_texts financial.txt --general_texts general.txt --top_n 10 --enable_labeling --prompt_file custom_prompt.txt
  
  # Using OpenRouter API
  python run_lite_analysis.py --domain_texts financial.txt --general_texts general.txt --top_n 10 --enable_labeling --labeling_provider openrouter --labeling_model "openai/gpt-3.5-turbo"
        """
    )
    
    # Required arguments
    parser.add_argument("--domain_data", required=True, help="Path to domain data file (one per line)")
    parser.add_argument("--general_data", required=True, help="Path to general data file (one per line)")
    
    # Model arguments
    parser.add_argument("--base_model", required=True, help="Base model: HuggingFace ID (e.g., 'meta-llama/Llama-2-7b-hf') or local path")
    parser.add_argument("--sae_model", required=True, help="SAE model: HuggingFace ID (e.g., 'EleutherAI/sae-llama-3-8b-32x') or local path")
    parser.add_argument("--layer_idx", type=int, default=16, help="Layer index to analyze")
    
    # Model configuration
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    
    # Analysis arguments
    parser.add_argument("--top_n", type=int, default=10, help="Number of top features to analyze")
    parser.add_argument("--domain_name", default="financial", help="Domain name for output files")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    # Labeling arguments
    parser.add_argument("--enable_labeling", action="store_true", help="Enable LLM-based labeling")
    parser.add_argument("--labeling_model", default="meta-llama/Llama-2-7b-chat-hf", help="Model for labeling (chat models only)")
    parser.add_argument("--labeling_provider", choices=["offline", "openrouter"], default="offline", help="Labeling provider")
    parser.add_argument("--prompt_file", help="Path to custom prompt file (optional)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.enable_labeling and args.labeling_provider == "openrouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            print("❌ OPENROUTER_API_KEY environment variable required for OpenRouter provider")
            sys.exit(1)
    
    # Run analysis
    runner = AutoInterpLiteRunner()
    runner.run_analysis(args)

if __name__ == "__main__":
    main()
