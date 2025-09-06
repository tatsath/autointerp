#!/usr/bin/env python3
"""
AutoInterp Light - Domain-Specific Feature Activation Analysis
Fast, lightweight feature analysis focused on activation patterns and domain-specific labels
"""

import torch
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors import safe_open
from typing import List, Dict, Tuple

class FeatureActivationAnalyzer:
    def __init__(self, base_model_name: str, sae_model_path: str, output_dir: str = "results"):
        """
        Initialize the feature activation analyzer
        
        Args:
            base_model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            sae_model_path: Path to the SAE model directory
            output_dir: Directory to save results
        """
        self.base_model_name = base_model_name
        self.sae_model_path = sae_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load models
        self.model, self.tokenizer = self._load_base_model()
        
    def _load_base_model(self):
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
    
    def _load_sae_model(self, layer_idx: int):
        """Load SAE model for specific layer"""
        print(f"Loading SAE model for layer {layer_idx}")
        
        sae_path = Path(self.sae_model_path)
        layer_name = f"layers.{layer_idx}"
        
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
        
        return encoder_weight, encoder_bias, sae_config
    
    def _extract_hidden_states(self, texts: List[str], layer_idx: int) -> torch.Tensor:
        """Extract hidden states from specified layer"""
        print(f"Extracting hidden states from layer {layer_idx}...")
        
        device = next(self.model.parameters()).device
        hidden_states = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_state = outputs.hidden_states[layer_idx]
                
                # Average over sequence length
                hidden_state = hidden_state.mean(dim=1)
                hidden_states.append(hidden_state.cpu())
        
        return torch.cat(hidden_states, dim=0)
    
    def _encode_activations(self, hidden_states: torch.Tensor, encoder_weight: torch.Tensor, encoder_bias: torch.Tensor = None) -> torch.Tensor:
        """Encode hidden states using SAE"""
        device = encoder_weight.device
        hidden_states = hidden_states.to(device).to(encoder_weight.dtype)
        
        if encoder_bias is not None:
            encoder_bias = encoder_bias.to(device)
            activations = torch.nn.functional.linear(hidden_states, encoder_weight, encoder_bias)
        else:
            activations = torch.nn.functional.linear(hidden_states, encoder_weight)
        
        return activations
    
    def analyze_domain_features(self, domain_texts: List[str], general_texts: List[str], 
                              layer_idx: int, top_n: int = 50) -> pd.DataFrame:
        """
        Analyze feature activations for domain-specific vs general texts
        
        Args:
            domain_texts: List of domain-specific texts (e.g., financial)
            general_texts: List of general texts
            layer_idx: Layer index to analyze
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature analysis results
        """
        print(f"Analyzing {len(domain_texts)} domain texts vs {len(general_texts)} general texts")
        
        # Load SAE model
        encoder_weight, encoder_bias, sae_config = self._load_sae_model(layer_idx)
        
        # Extract hidden states
        domain_hidden = self._extract_hidden_states(domain_texts, layer_idx)
        general_hidden = self._extract_hidden_states(general_texts, layer_idx)
        
        # Encode activations
        domain_activations = self._encode_activations(domain_hidden, encoder_weight, encoder_bias)
        general_activations = self._encode_activations(general_hidden, encoder_weight, encoder_bias)
        
        # Calculate metrics
        domain_avg = domain_activations.mean(dim=0)
        general_avg = general_activations.mean(dim=0)
        specialization = domain_avg - general_avg
        
        # Create results
        results = []
        for i in range(len(domain_avg)):
            results.append({
                'feature_number': i,
                'domain_activation': domain_avg[i].item(),
                'general_activation': general_avg[i].item(),
                'specialization': specialization[i].item(),
                'layer': layer_idx
            })
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(results)
        df = df.sort_values('specialization', ascending=False)
        
        # Get top features
        top_features = df.head(top_n)
        
        print(f"Top {top_n} features identified")
        print(f"Best feature: {top_features.iloc[0]['feature_number']} (specialization: {top_features.iloc[0]['specialization']:.3f})")
        
        return top_features
    
    def generate_feature_labels(self, top_features: pd.DataFrame, domain_texts: List[str], 
                              general_texts: List[str], layer_idx: int) -> pd.DataFrame:
        """Generate labels for top features using simple heuristics"""
        print("Generating feature labels...")
        
        # Load SAE model
        encoder_weight, encoder_bias, _ = self._load_sae_model(layer_idx)
        
        # Extract hidden states
        domain_hidden = self._extract_hidden_states(domain_texts, layer_idx)
        general_hidden = self._extract_hidden_states(general_texts, layer_idx)
        
        # Encode activations
        domain_activations = self._encode_activations(domain_hidden, encoder_weight, encoder_bias)
        general_activations = self._encode_activations(general_hidden, encoder_weight, encoder_bias)
        
        # Generate labels for each feature
        labels = []
        for _, row in top_features.iterrows():
            feature_idx = row['feature_number']
            
            # Get top activating texts
            domain_feature_acts = domain_activations[:, feature_idx]
            general_feature_acts = general_activations[:, feature_idx]
            
            # Find top activating domain text
            top_domain_idx = torch.argmax(domain_feature_acts).item()
            top_domain_text = domain_texts[top_domain_idx]
            
            # Generate simple label based on text content
            label = self._generate_simple_label(top_domain_text, row['specialization'])
            labels.append(label)
        
        # Add labels to results
        top_features = top_features.copy()
        top_features['label'] = labels
        
        return top_features
    
    def _generate_simple_label(self, text: str, specialization: float) -> str:
        """Generate a simple label based on text content and specialization"""
        # Simple keyword-based labeling
        text_lower = text.lower()
        
        if 'stock' in text_lower or 'market' in text_lower:
            return "Stock market trading"
        elif 'bank' in text_lower or 'loan' in text_lower:
            return "Banking and loans"
        elif 'crypto' in text_lower or 'bitcoin' in text_lower:
            return "Cryptocurrency"
        elif 'real estate' in text_lower or 'property' in text_lower:
            return "Real estate"
        elif 'insurance' in text_lower:
            return "Insurance"
        elif 'investment' in text_lower or 'portfolio' in text_lower:
            return "Investment management"
        elif 'earnings' in text_lower or 'revenue' in text_lower:
            return "Corporate earnings"
        elif 'debt' in text_lower or 'bond' in text_lower:
            return "Debt and bonds"
        elif 'merger' in text_lower or 'acquisition' in text_lower:
            return "M&A activity"
        elif 'inflation' in text_lower or 'cpi' in text_lower:
            return "Economic indicators"
        else:
            return f"Financial concept (spec: {specialization:.2f})"
    
    def run_analysis(self, domain_texts: List[str], general_texts: List[str], 
                    layer_idx: int, top_n: int = 50, domain_name: str = "financial") -> Dict:
        """
        Run complete feature activation analysis
        
        Args:
            domain_texts: List of domain-specific texts
            general_texts: List of general texts
            layer_idx: Layer index to analyze
            top_n: Number of top features to return
            domain_name: Name of the domain (for file naming)
            
        Returns:
            Dictionary with analysis results
        """
        print("="*80)
        print("AUTOINTERP LIGHT - FEATURE ACTIVATION ANALYSIS")
        print("="*80)
        print(f"Base Model: {self.base_model_name}")
        print(f"SAE Model: {self.sae_model_path}")
        print(f"Layer: {layer_idx}")
        print(f"Domain: {domain_name}")
        print(f"Top Features: {top_n}")
        print("="*80)
        
        # Analyze features
        top_features = self.analyze_domain_features(domain_texts, general_texts, layer_idx, top_n)
        
        # Generate labels
        top_features_with_labels = self.generate_feature_labels(top_features, domain_texts, general_texts, layer_idx)
        
        # Save results
        results_file = self.output_dir / f"{domain_name}_features_layer{layer_idx}.csv"
        top_features_with_labels.to_csv(results_file, index=False)
        
        # Save summary
        summary = {
            'domain': domain_name,
            'layer': layer_idx,
            'total_features_analyzed': len(top_features),
            'best_feature': int(top_features.iloc[0]['feature_number']),
            'best_specialization': float(top_features.iloc[0]['specialization']),
            'avg_specialization': float(top_features['specialization'].mean()),
            'results_file': str(results_file)
        }
        
        summary_file = self.output_dir / f"{domain_name}_summary_layer{layer_idx}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Analysis complete!")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        
        return {
            'features': top_features_with_labels,
            'summary': summary
        }

def main():
    parser = argparse.ArgumentParser(description="AutoInterp Light - Feature Activation Analysis")
    parser.add_argument("--base_model", required=True, help="Base model name")
    parser.add_argument("--sae_model", required=True, help="Path to SAE model directory")
    parser.add_argument("--domain_texts", required=True, help="Path to domain texts file (one per line)")
    parser.add_argument("--general_texts", required=True, help="Path to general texts file (one per line)")
    parser.add_argument("--layer_idx", type=int, default=16, help="Layer index to analyze")
    parser.add_argument("--top_n", type=int, default=50, help="Number of top features")
    parser.add_argument("--domain_name", default="financial", help="Domain name")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Load texts
    with open(args.domain_texts, 'r') as f:
        domain_texts = [line.strip() for line in f if line.strip()]
    
    with open(args.general_texts, 'r') as f:
        general_texts = [line.strip() for line in f if line.strip()]
    
    # Create analyzer
    analyzer = FeatureActivationAnalyzer(
        base_model_name=args.base_model,
        sae_model_path=args.sae_model,
        output_dir=args.output_dir
    )
    
    # Run analysis
    results = analyzer.run_analysis(
        domain_texts=domain_texts,
        general_texts=general_texts,
        layer_idx=args.layer_idx,
        top_n=args.top_n,
        domain_name=args.domain_name
    )

if __name__ == "__main__":
    main()
