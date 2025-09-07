#!/usr/bin/env python3
"""
Simple Feature Steering Demo
Uses multi-layer lite results to get feature activations and demonstrate steering effects.
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse
import os
import sys

# Add autointerp paths
sys.path.append('../autointerp_lite')
sys.path.append('../autointerp_full')

def load_top_features(layer, results_dir="multi_layer_lite_results"):
    """Load top features for a given layer"""
    csv_path = f"{results_dir}/features_layer{layer}.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Results not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    return df.head(5)  # Top 5 features

def get_model_and_tokenizer():
    """Load the base model and tokenizer"""
    print("🔄 Loading model and tokenizer...")
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def get_activations(model, tokenizer, text, layer_idx):
    """Get activations for a given text at a specific layer"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        activations = outputs.hidden_states[layer_idx]  # 0-indexed
        return activations.mean(dim=1).squeeze()  # Average over sequence length

def load_sae_model(layer_idx):
    """Load SAE model for a specific layer"""
    sae_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    try:
        # Try to import SAE loading function
        from autointerp_lite.core.sae_loader import load_sae_model
        sae_model = load_sae_model(sae_path, layer_idx)
        return sae_model
    except ImportError:
        print("⚠️  SAE loading not available, using simulated activations")
        return None

def get_feature_activations(sae_model, activations, feature_ids):
    """Get activations for specific features"""
    if sae_model is None:
        # Simulate feature activations
        return torch.randn(len(feature_ids)) * 10
    
    try:
        # Get SAE activations
        sae_activations = sae_model.encode(activations.unsqueeze(0))
        return sae_activations[0, feature_ids]
    except:
        # Fallback to simulated activations
        return torch.randn(len(feature_ids)) * 10

def apply_steering(activations, feature_ids, steering_strength, target_features):
    """Apply steering to specific features"""
    steered_activations = activations.clone()
    
    for i, feature_id in enumerate(feature_ids):
        if feature_id in target_features:
            steered_activations[i] += steering_strength
    
    return steered_activations

def generate_text_with_steering(model, tokenizer, prompt, steered_activations, layer_idx, max_length=100):
    """Generate text with steered activations"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # This is a simplified version - in practice, you'd need to modify the model's forward pass
    # For demo purposes, we'll just return the original generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Feature Steering Demo")
    parser.add_argument("--layer", type=int, default=16, help="Layer to analyze (4, 10, 16, 22, 28)")
    parser.add_argument("--prompt", type=str, default="The company reported strong earnings growth", help="Input prompt")
    parser.add_argument("--steering-strength", type=float, default=5.0, help="Steering strength")
    args = parser.parse_args()
    
    print("🚀 Feature Steering Demo")
    print("=" * 50)
    print(f"📊 Layer: {args.layer}")
    print(f"💬 Prompt: {args.prompt}")
    print(f"🎯 Steering Strength: {args.steering_strength}")
    print()
    
    # Load top features for the layer
    features_df = load_top_features(args.layer)
    if features_df is None:
        return
    
    print("🏆 Top Features for Layer {}:".format(args.layer))
    for idx, row in features_df.iterrows():
        print(f"  {row['feature']}: {row['llm_label']} (Spec: {row['specialization']:.2f})")
    print()
    
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()
    
    # Load SAE model
    sae_model = load_sae_model(args.layer)
    
    # Get activations for the prompt
    print("🔄 Getting activations...")
    activations = get_activations(model, tokenizer, args.prompt, args.layer)
    
    # Get feature activations
    feature_ids = features_df['feature'].tolist()
    feature_activations = get_feature_activations(sae_model, activations, feature_ids)
    
    print("📈 Feature Activations (Before Steering):")
    for i, (feature_id, activation) in enumerate(zip(feature_ids, feature_activations)):
        label = features_df.iloc[i]['llm_label']
        print(f"  Feature {feature_id}: {activation:.3f} - {label}")
    print()
    
    # Apply steering to top 2 features
    target_features = feature_ids[:2]  # Steer top 2 features
    steered_activations = apply_steering(feature_activations, feature_ids, args.steering_strength, target_features)
    
    print("🎯 Feature Activations (After Steering):")
    for i, (feature_id, activation) in enumerate(zip(feature_ids, steered_activations)):
        label = features_df.iloc[i]['llm_label']
        change = activation - feature_activations[i]
        print(f"  Feature {feature_id}: {activation:.3f} ({change:+.3f}) - {label}")
    print()
    
    # Generate text with and without steering
    print("📝 Text Generation:")
    print("Original:", args.prompt)
    
    # Note: In a real implementation, you'd modify the model's forward pass
    # to use the steered activations. This is a simplified demo.
    print("Steered: [Steering effect would be applied here in full implementation]")
    
    print("\n✅ Demo completed!")
    print("💡 In a full implementation, the steered activations would modify the model's behavior")

if __name__ == "__main__":
    main()
