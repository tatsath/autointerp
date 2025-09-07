#!/usr/bin/env python3
"""
Generic Feature Analysis System
Works with any SAE model and any number of top features
"""

import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse
import sys
import os
from safetensors import safe_open

class GenericFeatureAnalyzer:
    def __init__(self, base_model_name, sae_model_path, output_dir="results"):
        """
        Initialize the generic feature analyzer
        
        Args:
            base_model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            sae_model_path: Path to the SAE model directory
            output_dir: Directory to save results
        """
        self.base_model_name = base_model_name
        self.sae_model_path = sae_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Financial and general sentences for analysis
        self.financial_sentences = [
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
        
        self.general_sentences = [
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
    
    def load_sae_model(self, layer_idx=16):
        """Load the SAE model"""
        print(f"Loading SAE model from: {self.sae_model_path}")
        
        # Load SAE components
        sae_path = Path(self.sae_model_path)
        layer_name = f"layers.{layer_idx}"
        
        # Load encoder and decoder from safetensors
        sae_file = sae_path / layer_name / "sae.safetensors"
        cfg_file = sae_path / layer_name / "cfg.json"
        
        if not sae_file.exists() or not cfg_file.exists():
            raise FileNotFoundError(f"SAE model files not found in {sae_path / layer_name}")
        
        # Load config
        with open(cfg_file, 'r') as f:
            sae_config = json.load(f)
        
        # Load weights
        sae_weights = {}
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        with safe_open(sae_file, framework="pt", device=device) as f:
            for key in f.keys():
                sae_weights[key] = f.get_tensor(key)
        
        encoder_weight = sae_weights.get('encoder.weight')
        encoder_bias = sae_weights.get('encoder.bias')
        decoder_weight = sae_weights.get('decoder.weight')
        decoder_bias = sae_weights.get('decoder.bias')
        
        print(f"SAE model loaded: {encoder_weight.shape[0]} features, {encoder_weight.shape[1]} dimensions")
        return encoder_weight, encoder_bias, decoder_weight, decoder_bias
    
    def load_base_model(self):
        """Load the base model for feature extraction"""
        print(f"Loading base model: {self.base_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        return model, tokenizer
    
    def extract_hidden_states(self, model, tokenizer, sentences, layer_idx=16):
        """Extract hidden states from the specified layer"""
        print(f"Extracting hidden states from layer {layer_idx}...")
        
        device = next(model.parameters()).device
        hidden_states = []
        
        with torch.no_grad():
            for sentence in sentences:
                # Tokenize
                inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get hidden states
                outputs = model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_idx]  # Get specific layer
                
                # Average over sequence length
                hidden_state = hidden_state.mean(dim=1)  # [batch_size, hidden_dim]
                hidden_states.append(hidden_state.cpu())
        
        return torch.cat(hidden_states, dim=0)
    
    def analyze_feature_activations(self, encoder_weight, encoder_bias, financial_hidden, general_hidden, top_n=10):
        """Analyze feature activations and return top features"""
        print(f"Analyzing feature activations for top {top_n} features...")
        
        # Encode hidden states using SAE encoder
        device = encoder_weight.device
        financial_hidden = financial_hidden.to(device).to(encoder_weight.dtype)
        general_hidden = general_hidden.to(device).to(encoder_weight.dtype)
        
        if encoder_bias is not None:
            encoder_bias = encoder_bias.to(device)
            financial_activations = torch.nn.functional.linear(financial_hidden, encoder_weight, encoder_bias)
            general_activations = torch.nn.functional.linear(general_hidden, encoder_weight, encoder_bias)
        else:
            financial_activations = torch.nn.functional.linear(financial_hidden, encoder_weight)
            general_activations = torch.nn.functional.linear(general_hidden, encoder_weight)
        
        # Calculate average activations
        avg_financial = financial_activations.mean(dim=0)  # [n_features]
        avg_general = general_activations.mean(dim=0)
        specialization = avg_financial - avg_general
        
        # Get top features by specialization
        top_indices = torch.argsort(specialization, descending=True)[:top_n]
        
        results = []
        for i, idx in enumerate(top_indices):
            feature_idx = idx.item()
            results.append({
                'feature_number': feature_idx,
                'financial_activation': avg_financial[feature_idx].item(),
                'general_activation': avg_general[feature_idx].item(),
                'specialization': specialization[feature_idx].item(),
                'rank': i + 1
            })
        
        return results, financial_activations, general_activations
    
    def extract_top_activating_sentences(self, financial_activations, general_activations, top_features, n_sentences=5):
        """Extract top activating sentences for each feature"""
        print("Extracting top activating sentences...")
        
        feature_sentences = {}
        
        for feature_data in top_features:
            feature_idx = feature_data['feature_number']
            
            # Get activations for this feature
            financial_feature_acts = financial_activations[:, feature_idx]
            general_feature_acts = general_activations[:, feature_idx]
            
            # Get top financial sentences
            top_financial_indices = torch.argsort(financial_feature_acts, descending=True)[:n_sentences]
            top_general_indices = torch.argsort(general_feature_acts, descending=True)[:n_sentences]
            
            sentences_data = []
            
            # Add top financial sentences
            for idx in top_financial_indices:
                sentences_data.append({
                    'sentence': self.financial_sentences[idx.item()],
                    'activation': financial_feature_acts[idx].item(),
                    'type': 'financial'
                })
            
            # Add top general sentences
            for idx in top_general_indices:
                sentences_data.append({
                    'sentence': self.general_sentences[idx.item()],
                    'activation': general_feature_acts[idx].item(),
                    'type': 'general'
                })
            
            feature_sentences[str(feature_idx)] = sentences_data
        
        return feature_sentences
    
    def run_analysis(self, top_n=10, layer_idx=16):
        """Run the complete feature analysis"""
        print("="*80)
        print("GENERIC FEATURE ANALYSIS")
        print("="*80)
        print(f"Base Model: {self.base_model_name}")
        print(f"SAE Model: {self.sae_model_path}")
        print(f"Top Features: {top_n}")
        print(f"Layer: {layer_idx}")
        print("="*80)
        
        # Load models
        encoder_weight, encoder_bias, decoder_weight, decoder_bias = self.load_sae_model(layer_idx)
        model, tokenizer = self.load_base_model()
        
        # Extract hidden states
        financial_hidden = self.extract_hidden_states(model, tokenizer, self.financial_sentences, layer_idx)
        general_hidden = self.extract_hidden_states(model, tokenizer, self.general_sentences, layer_idx)
        
        # Analyze features
        top_features, financial_activations, general_activations = self.analyze_feature_activations(
            encoder_weight, encoder_bias, financial_hidden, general_hidden, top_n
        )
        
        # Extract top sentences
        feature_sentences = self.extract_top_activating_sentences(
            financial_activations, general_activations, top_features
        )
        
        # Save results
        results_file = self.output_dir / f"top_{top_n}_features_analysis.json"
        with open(results_file, 'w') as f:
            json.dump({
                'top_features': top_features,
                'feature_sentences': feature_sentences,
                'config': {
                    'base_model': self.base_model_name,
                    'sae_model': str(self.sae_model_path),
                    'top_n': top_n,
                    'layer_idx': layer_idx
                }
            }, f, indent=2)
        
        # Save CSV
        df = pd.DataFrame(top_features)
        csv_file = self.output_dir / f"top_{top_n}_features_analysis.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {results_file}")
        print(f"CSV saved to: {csv_file}")
        
        return top_features, feature_sentences

def main():
    parser = argparse.ArgumentParser(description="Generic Feature Analysis")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--sae_model", required=True, help="Path to SAE model directory")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top features to analyze")
    parser.add_argument("--layer_idx", type=int, default=16, help="Layer index for feature extraction")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = GenericFeatureAnalyzer(
        base_model_name=args.base_model,
        sae_model_path=args.sae_model,
        output_dir=args.output_dir
    )
    
    # Run analysis
    top_features, feature_sentences = analyzer.run_analysis(
        top_n=args.top_n,
        layer_idx=args.layer_idx
    )
    
    print(f"\nTop {args.top_n} features:")
    for feature in top_features:
        print(f"  Feature {feature['feature_number']}: Specialization = {feature['specialization']:.3f}")

if __name__ == "__main__":
    main()
