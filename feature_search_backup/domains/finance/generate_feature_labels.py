#!/usr/bin/env python3
"""
Simple script to generate feature labels by:
1. Collecting sentences that activate features (high confidence)
2. Collecting sentences that don't activate features
3. Using an explainer model to generate labels
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import os

# Configuration
CONFIG_PATH = "config.json"
SCORES_FILE = "scores/top_features_scores.json"
OUTPUT_FILE = "scores/top_features_with_labels.json"

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

MODEL_PATH = config['model_path']
SAE_PATH = config['sae_path']
LAYER = 19  # From sae_id: blocks.19.hook_resid_post
DATASET_PATH = config['dataset_path']
N_SAMPLES = 1000  # Number of sentences to check
TOP_K_ACTIVATING = 10  # Top activating sentences
TOP_K_NON_ACTIVATING = 10  # Top non-activating sentences

# Explainer model (use same as base model to save memory)
USE_SAME_MODEL_FOR_EXPLAINER = True
EXPLAINER_MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # Only used if USE_SAME_MODEL_FOR_EXPLAINER is False

def load_sae_weights(sae_path, layer):
    """Load SAE encoder weights"""
    print(f"Loading SAE from {sae_path}...")
    
    from safetensors import safe_open
    
    # Try layers.{layer}/sae.safetensors structure
    layer_dir = os.path.join(sae_path, f"layers.{layer}")
    sae_file = os.path.join(layer_dir, "sae.safetensors")
    
    if os.path.exists(sae_file):
        print(f"  Found SAE at: {sae_file}")
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
            return {"encoder": encoder, "encoder_bias": encoder_bias}
    
    # Try blocks.{layer}.hook_resid_post.safetensors
    sae_file = os.path.join(sae_path, f"blocks.{layer}.hook_resid_post.safetensors")
    if os.path.exists(sae_file):
        print(f"  Found SAE at: {sae_file}")
        with safe_open(sae_file, framework="pt", device="cpu") as f:
            encoder = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias")
            return {"encoder": encoder, "encoder_bias": encoder_bias}
    
    # Try loading SAE using sae_lens
    try:
        from sae_lens import SAE
        sae = SAE.from_pretrained(sae_path)
        return {
            "encoder": sae.W_enc,
            "encoder_bias": sae.b_enc
        }
    except Exception as e:
        print(f"Error loading SAE: {e}")
        raise

def get_feature_activation(text, model, tokenizer, sae_weights, layer, device):
    """Get feature activations for a text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # hidden_states[0] is embedding, hidden_states[1] is layer 0, ..., hidden_states[20] is layer 19
        hidden_states = outputs.hidden_states[layer + 1]  # +1 because embedding is at index 0
        
        encoder = sae_weights["encoder"].to(hidden_states.device).to(hidden_states.dtype)
        feature_activations = torch.matmul(hidden_states, encoder.T)
        
        if "encoder_bias" in sae_weights:
            encoder_bias = sae_weights["encoder_bias"].to(hidden_states.device).to(hidden_states.dtype)
            feature_activations = feature_activations + encoder_bias
        
        # Apply ReLU
        feature_activations = torch.relu(feature_activations)
        
        # Use max activation across sequence length
        max_activations = feature_activations.max(dim=1)[0].squeeze(0)
        
    return max_activations.cpu().numpy()

def collect_activating_sentences(feature_idx, model, tokenizer, sae_weights, dataset, layer, device, n_samples=1000, top_k=10):
    """Collect top activating and non-activating sentences for a feature"""
    print(f"  Collecting sentences for feature {feature_idx}...")
    
    activations_list = []
    sentences_list = []
    
    # Sample sentences from dataset
    for i, example in enumerate(tqdm(dataset, desc=f"    Processing", total=min(n_samples, len(dataset) if hasattr(dataset, '__len__') else n_samples), leave=False)):
        if i >= n_samples:
            break
        
        # Handle different dataset formats
        if isinstance(example, dict):
            text = example.get('text', '') or example.get('content', '') or example.get('article', '') or str(example)
        else:
            text = str(example)
        
        if not text or len(text.strip()) < 10:
            continue
        
        try:
            activations = get_feature_activation(text, model, tokenizer, sae_weights, layer, device)
            feature_activation = activations[feature_idx]
            
            # Only add if activation is positive (ReLU already applied)
            if feature_activation > 0:
                activations_list.append(feature_activation)
                sentences_list.append(text)
        except Exception as e:
            if i < 3:  # Print first few errors for debugging
                print(f"    Error processing sentence {i}: {e}")
            continue
    
    if not activations_list:
        print(f"    ⚠️  No positive activations found for feature {feature_idx}")
        return [], []
    
    # Get top activating and non-activating
    activations_array = np.array(activations_list)
    
    # For non-activating, we need sentences with zero or very low activation
    # So we'll also check sentences that didn't activate at all
    if len(activations_array) < top_k:
        print(f"    ⚠️  Only found {len(activations_array)} activating sentences (need {top_k})")
        top_k = len(activations_array)
    
    top_indices = np.argsort(activations_array)[-top_k:][::-1] if len(activations_array) > 0 else []
    
    # For non-activating, use the ones with lowest activation from what we have
    # Or we could sample more sentences specifically looking for zeros
    bottom_k = min(top_k, len(activations_array))
    bottom_indices = np.argsort(activations_array)[:bottom_k] if len(activations_array) > 0 else []
    
    top_sentences = [(sentences_list[i], float(activations_array[i])) for i in top_indices]
    bottom_sentences = [(sentences_list[i], float(activations_array[i])) for i in bottom_indices]
    
    print(f"    Found {len(top_sentences)} activating sentences (max activation: {max(activations_array) if len(activations_array) > 0 else 0:.4f})")
    
    return top_sentences, bottom_sentences

def generate_label(feature_idx, activating_sentences, non_activating_sentences, explainer_model, explainer_tokenizer, device):
    """Generate label using explainer model"""
    
    # Format prompt - truncate sentences to first 100 chars to keep prompt short
    activating_text = "\n".join([f"{i+1}. \"{sent[:100]}...\" (activation: {act:.4f})" 
                                 for i, (sent, act) in enumerate(activating_sentences[:5])])
    non_activating_text = "\n".join([f"{i+1}. \"{sent[:100]}...\" (activation: {act:.4f})" 
                                      for i, (sent, act) in enumerate(non_activating_sentences[:5])])
    
    prompt = f"""Analyze this neural network feature pattern.

Feature {feature_idx} activates on:
{activating_text}

Feature {feature_idx} does NOT activate on:
{non_activating_text}

Provide ONLY a 2-5 word label describing what this feature detects. Just the label, nothing else:
Label:"""
    
    # Generate label
    inputs = explainer_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = explainer_model.generate(
            **inputs,
            max_new_tokens=15,
            temperature=0.2,
            do_sample=True,
            pad_token_id=explainer_tokenizer.eos_token_id,
            eos_token_id=explainer_tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    response = explainer_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    label = response.strip()
    
    # Clean up label - extract just the label part
    # Remove common prefixes
    for prefix in ["Label:", "label:", "The label is", "The feature detects", "This feature detects"]:
        if prefix in label:
            label = label.split(prefix)[-1].strip()
    
    # Remove quotes
    label = label.strip('"').strip("'").strip()
    
    # Take only first line and limit to 50 chars
    label = label.split('\n')[0].strip()
    if len(label) > 50:
        label = label[:50].rsplit(' ', 1)[0]  # Cut at word boundary
    
    # Remove any trailing punctuation that looks like it's part of the sentence
    if label.endswith('.') and len(label) > 10:
        label = label[:-1].strip()
    
    return label

def main():
    print("🚀 Generating feature labels from activating sentences")
    print("=" * 60)
    
    # Load feature indices
    with open(SCORES_FILE, 'r') as f:
        scores_data = json.load(f)
    
    feature_indices = scores_data['feature_indices']
    feature_scores = scores_data['scores']
    
    print(f"📊 Processing {len(feature_indices)} features")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Load base model
    print(f"📥 Loading model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try to use device_map, fallback to manual device placement
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    except (ValueError, ImportError):
        # Fallback: load to CPU/GPU manually
        print("  Using manual device placement...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        if device == "cuda":
            model = model.to(device)
    model.eval()
    
    # Load SAE weights
    sae_weights = load_sae_weights(SAE_PATH, LAYER)
    print(f"✅ SAE loaded: {sae_weights['encoder'].shape[0]} features")
    
    # Use same model for explainer to save memory
    if USE_SAME_MODEL_FOR_EXPLAINER:
        print("📥 Using same model for explainer (saving memory)")
        explainer_model = model
        explainer_tokenizer = tokenizer
    else:
        print(f"📥 Loading explainer model: {EXPLAINER_MODEL}")
        explainer_tokenizer = AutoTokenizer.from_pretrained(EXPLAINER_MODEL)
        try:
            explainer_model = AutoModelForCausalLM.from_pretrained(
                EXPLAINER_MODEL,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
        except (ValueError, ImportError):
            explainer_model = AutoModelForCausalLM.from_pretrained(
                EXPLAINER_MODEL,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            if device == "cuda":
                explainer_model = explainer_model.to(device)
        explainer_model.eval()
        
        if explainer_tokenizer.pad_token is None:
            explainer_tokenizer.pad_token = explainer_tokenizer.eos_token
    
    # Load dataset
    print(f"📥 Loading dataset: {DATASET_PATH}")
    try:
        dataset = load_dataset(DATASET_PATH, split="train", streaming=False)
        # Limit dataset size
        if hasattr(dataset, '__len__'):
            max_samples = min(N_SAMPLES * 2, len(dataset))
            dataset = dataset.select(range(max_samples))
        else:
            dataset = list(dataset.take(N_SAMPLES * 2))
        print(f"✅ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        print(f"⚠️  Error loading dataset: {e}")
        print("   Trying streaming mode...")
        dataset = load_dataset(DATASET_PATH, split="train", streaming=True)
        dataset = list(dataset.take(N_SAMPLES * 2))
        print(f"✅ Dataset loaded: {len(dataset)} samples (streaming)")
    print()
    
    # Process each feature
    results = []
    for idx, (feat_idx, score) in enumerate(zip(feature_indices, feature_scores)):
        print(f"[{idx+1}/{len(feature_indices)}] Processing feature {feat_idx}...")
        
        try:
            # Collect activating and non-activating sentences
            activating_sents, non_activating_sents = collect_activating_sentences(
                feat_idx, model, tokenizer, sae_weights, dataset, LAYER, device, 
                n_samples=N_SAMPLES, top_k=TOP_K_ACTIVATING
            )
            
            if not activating_sents:
                print(f"  ⚠️  No activating sentences found, using 'N/A'")
                label = "N/A"
            else:
                # Generate label
                label = generate_label(
                    feat_idx, activating_sents, non_activating_sents,
                    explainer_model, explainer_tokenizer, device
                )
                print(f"  ✅ Label: {label}")
            
            results.append({
                'feature_index': feat_idx,
                'score': score,
                'label': label,
                'f1_score': None  # Not computed in this simple approach
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'feature_index': feat_idx,
                'score': score,
                'label': 'N/A',
                'f1_score': None
            })
        
        print()
    
    # Save results
    output_data = {
        'quantile_threshold': scores_data['quantile_threshold'],
        'quantile_value': scores_data['quantile_value'],
        'features': results
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved labels to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

