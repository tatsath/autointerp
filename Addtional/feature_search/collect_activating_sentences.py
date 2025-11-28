#!/usr/bin/env python3
"""Collect token-level context windows that activate each feature"""

import json
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors import safe_open
from tqdm import tqdm

# Config
FEATURE_LIST = "test_results/feature_list.json"
CONFIG_PATH = "domains/finance/config.json"  # Use same config as feature search
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
SAE_PATH = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
DATASET_PATH = "jyanimaulik/yahoo_finance_stockmarket_news"
TOKENS_STR_PATH = "domains/finance/finance_tokens.json"
LAYER = 19
N_SAMPLES = 500
TOP_K_POS = 20  # Top activating context windows
TOP_K_NEG = 20  # Bottom activating context windows (non-activating)

def load_sae(sae_path, layer):
    layer_dir = f"{sae_path}/layers.{layer}"
    sae_file = f"{layer_dir}/sae.safetensors"
    with safe_open(sae_file, framework="pt", device="cpu") as f:
        return {"encoder": f.get_tensor("encoder.weight"), "encoder_bias": f.get_tensor("encoder.bias")}

def load_finance_vocab(tokens_str_path, tokenizer):
    """
    Load finance vocabulary and convert to token IDs.
    Uses EXACT same logic as compute_score.py:
    - Groups tokens by normalized string (treat " A", "A", " a", "a" as same)
    - Returns list of groups, where each group is a list of token sequences
    """
    with open(tokens_str_path, 'r') as f:
        tokens_str = json.load(f)
    
    # Group tokens by normalized string (EXACT same as compute_score.py)
    grouped_tokens = defaultdict(list)
    for str_token in tokens_str:
        # we treat " A", "A", " a", "a" as the same token
        normalized_str = str_token.lstrip().lower()
        token_ids = tokenizer.encode(str_token, add_special_tokens=False)
        grouped_tokens[normalized_str].append(torch.tensor(token_ids, dtype=torch.long))
    
    # Return as list of groups (each group contains multiple token sequences)
    return list(grouped_tokens.values())

def _compute_single_mask(tokens, ids_of_interest, expand_range):
    """
    Compute mask for a single token sequence with expansion.
    EXACT same logic as compute_score.py RollingMean._compute_single_mask
    """
    seq_len = tokens.size(0)
    ids_len = len(ids_of_interest)
    
    mask = torch.zeros(seq_len, dtype=torch.bool, device=tokens.device)
    if ids_len > seq_len:
        return mask

    # Use unfold for vectorized matching (same as compute_score.py)
    ids_of_interest = ids_of_interest.view(1, -1)
    windows = tokens.unfold(0, ids_len, 1)  # [seq_len - ids_len + 1, ids_len]
    matches = (windows == ids_of_interest).all(dim=1)
    window_indices = torch.nonzero(matches, as_tuple=False).squeeze(-1)
    
    if len(window_indices) == 0:
        return mask

    # Mark all tokens in matched sequences
    offsets = torch.arange(ids_len, device=tokens.device)
    spans = window_indices.unsqueeze(1) + offsets.unsqueeze(0)
    spans_flat = spans.reshape(-1)
    mask[spans_flat] = True

    # Apply expand_range (EXACT same logic as compute_score.py)
    left, right = expand_range
    if left != 0 or right != 0:
        pos_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if len(pos_indices) > 0:
            starts = torch.clamp(pos_indices - left, min=0)
            ends = torch.clamp(pos_indices + right, max=seq_len - 1)
            # Use delta/cumsum approach (same as compute_score.py)
            delta = torch.zeros(seq_len + 1, dtype=torch.int32, device=tokens.device)
            delta[starts] += 1
            delta[ends + 1] -= 1
            coverage = delta.cumsum(dim=0)
            coverage = coverage[:-1]  # Trim last element
            mask = coverage > 0

    return mask

def find_vocab_matches(tokens, domain_tokens, expand_range, ignore_tokens=None):
    """
    Find positions where vocabulary tokens appear and expand context windows.
    Uses EXACT same logic as compute_score.py RollingMean.update
    Returns a boolean mask marking positive (vocab match) positions.
    """
    mask_combined = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
    
    # Handle ignore_tokens (same as compute_score.py)
    if ignore_tokens is not None and len(ignore_tokens) > 0:
        ignore_tensor = torch.tensor(ignore_tokens, dtype=torch.long, device=tokens.device)
        ignore_mask = torch.isin(tokens, ignore_tensor)
    else:
        ignore_mask = torch.zeros_like(tokens, dtype=torch.bool)
    
    # Process each token group (same as compute_score.py)
    for seq_group in domain_tokens:
        group_mask = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
        for seq in seq_group:
            seq = seq.to(tokens.device)
            seq_mask = _compute_single_mask(tokens, seq, expand_range)
            group_mask |= seq_mask
        
        # Exclude ignored tokens (same as compute_score.py)
        group_mask = group_mask & (~ignore_mask)
        mask_combined |= group_mask
    
    return mask_combined

def extract_context_windows(tokens, mask, tokenizer, feature_acts=None, max_windows=50):
    """
    Extract context windows from token positions marked in mask.
    Note: mask already includes expand_range expansion, so we just extract windows around marked positions.
    Returns list of (text, activation) tuples.
    """
    contexts = []
    positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    
    if len(positions) == 0:
        return contexts
    
    # Extract individual windows around each marked position
    # To avoid duplicates, we'll sample positions if there are too many
    if len(positions) > max_windows:
        # Sample evenly spaced positions
        indices = torch.linspace(0, len(positions) - 1, max_windows, dtype=torch.long)
        positions = positions[indices]
    
    # Group consecutive positions into windows (since mask already includes expansion)
    # Extract windows around each marked position
    seen_windows = set()
    for pos in positions:
        pos_idx = pos.item()
        # Extract a small window around this position (the mask already expanded, so just get local context)
        # Use a fixed small window size for extraction
        window_size = 10  # Extract ±5 tokens around each position for readability
        window_start = max(0, pos_idx - window_size // 2)
        window_end = min(len(tokens), pos_idx + window_size // 2 + 1)
        window_tokens = tokens[window_start:window_end]
        
        # Create a hash to avoid duplicate windows
        window_hash = tuple(window_tokens.tolist())
        if window_hash in seen_windows:
            continue
        seen_windows.add(window_hash)
        
        # Get activation at center position (or max in window if feature_acts provided)
        if feature_acts is not None:
            window_acts = feature_acts[window_start:window_end]
            activation = window_acts.max().item()
        else:
            activation = 0.0
        
        # Decode to text
        try:
            text = tokenizer.decode(window_tokens.tolist(), skip_special_tokens=True)
            if text and len(text.strip()) > 3:  # Only keep non-empty windows
                contexts.append((text.strip(), activation))
        except:
            continue
    
    return contexts

def get_token_activations_batch(tokens_list, model, encoder, encoder_bias, layer, device):
    """Get feature activations for a batch of token sequences"""
    with torch.no_grad():
        # Pad sequences to same length
        max_len = max(t.size(0) for t in tokens_list)
        batch_tokens = []
        for tokens in tokens_list:
            padded = torch.zeros(max_len, dtype=tokens.dtype, device=tokens.device)
            padded[:tokens.size(0)] = tokens
            batch_tokens.append(padded)
        batch_tokens = torch.stack(batch_tokens)  # [batch_size, max_len]
        
        # Get hidden states for batch
        outputs = model(batch_tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[layer + 1]  # [batch_size, max_len, hidden_dim]
        
        # Apply SAE encoder (encoder already on device)
        acts = torch.matmul(hidden, encoder.T) + encoder_bias
        acts = torch.relu(acts)  # [batch_size, max_len, n_features]
        
        # Return list of activations (trimmed to original lengths)
        results = []
        for i, tokens in enumerate(tokens_list):
            seq_len = tokens.size(0)
            results.append(acts[i, :seq_len])  # [seq_len, n_features]
        
        return results

def main():
    # Load config from same file used in feature search
    import os
    config_dir = os.path.dirname(os.path.abspath(CONFIG_PATH))
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Use EXACT same parameters as feature search
    EXPAND_RANGE = tuple(config['expand_range'])
    IGNORE_TOKENS = config.get('ignore_tokens', [])
    # Get paths from config, resolve relative paths relative to config file location
    dataset_path = config.get('dataset_path', DATASET_PATH)
    tokens_str_path = config.get('tokens_str_path', TOKENS_STR_PATH)
    if not os.path.isabs(tokens_str_path):
        tokens_str_path = os.path.join(config_dir, tokens_str_path)
    
    print(f"Using config from: {CONFIG_PATH}")
    print(f"expand_range: {EXPAND_RANGE}")
    print(f"ignore_tokens: {IGNORE_TOKENS}")
    print(f"dataset_path: {dataset_path}")
    print(f"tokens_str_path: {tokens_str_path}")
    
    # Load feature list
    with open(FEATURE_LIST) as f:
        data = json.load(f)
    feature_indices = data["feature_indices"]
    
    # Load model & SAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16 if device=="cuda" else torch.float32)
    if device == "cuda":
        model = model.to(device)
    model.eval()
    sae_weights = load_sae(SAE_PATH, LAYER)
    
    # Move encoder weights to device once (not on every call)
    encoder = sae_weights["encoder"].to(device)
    encoder_bias = sae_weights["encoder_bias"].to(device)
    if device == "cuda":
        encoder = encoder.to(torch.float16)
        encoder_bias = encoder_bias.to(torch.float16)
    
    # Load finance vocabulary using EXACT same logic as compute_score.py
    print("Loading finance vocabulary...")
    domain_tokens = load_finance_vocab(tokens_str_path, tokenizer)
    print(f"Loaded {len(domain_tokens)} vocabulary token groups")
    print(f"Using expand_range: {EXPAND_RANGE} (left={EXPAND_RANGE[0]}, right={EXPAND_RANGE[1]})")
    
    # Load dataset
    dataset = load_dataset(dataset_path, split="train", streaming=False)
    dataset = dataset.select(range(min(N_SAMPLES * 2, len(dataset))))
    
    # Pre-process and tokenize all texts in batches
    print("Pre-processing dataset...")
    batch_size = 8  # Process 8 texts at a time
    tokenized_texts = []
    texts_list = []
    
    for example in tqdm(dataset, desc="Tokenizing", total=min(N_SAMPLES, len(dataset))):
        text = example.get('text', '') or example.get('content', '') or str(example)
        if not text or len(text.strip()) < 10:
            continue
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            if tokens.size(1) == 0:
                continue
            tokens = tokens.squeeze(0).to(device)
            tokenized_texts.append(tokens)
            texts_list.append(text)
            
            if len(tokenized_texts) >= N_SAMPLES:
                break
        except:
            continue
    
    print(f"Tokenized {len(tokenized_texts)} texts")
    
    # Collect context windows for each feature
    results = {}
    for feat_idx in tqdm(feature_indices, desc="Processing features"):
        pos_contexts = []  # (context_text, activation_value)
        neg_contexts = []  # (context_text, activation_value)
        
        # Process texts in batches
        for batch_start in range(0, len(tokenized_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(tokenized_texts))
            batch_tokens = tokenized_texts[batch_start:batch_end]
            
            # Get activations for batch
            batch_acts = get_token_activations_batch(batch_tokens, model, encoder, encoder_bias, LAYER, device)
            
            # Process each text in batch
            for i, tokens in enumerate(batch_tokens):
                feature_acts = batch_acts[i][:, feat_idx]  # [seq_len]
                
                # Find vocabulary matches using EXACT same logic as compute_score.py
                vocab_mask = find_vocab_matches(tokens, domain_tokens, EXPAND_RANGE, IGNORE_TOKENS)
                
                # Extract positive contexts (vocab match positions)
                if vocab_mask.any():
                    pos_mask = vocab_mask
                    pos_windows = extract_context_windows(tokens, pos_mask, tokenizer, feature_acts, max_windows=100)
                    pos_contexts.extend(pos_windows)
                
                # Extract negative contexts (non-vocab positions)
                neg_mask = ~vocab_mask
                if neg_mask.any():
                    # Sample some negative positions to avoid too many
                    neg_positions = torch.nonzero(neg_mask, as_tuple=False).squeeze(-1)
                    if len(neg_positions) > 100:
                        sample_indices = torch.randperm(len(neg_positions))[:50]
                        neg_positions = neg_positions[sample_indices]
                    
                    # Create mask for sampled negative positions
                    sampled_neg_mask = torch.zeros_like(neg_mask)
                    sampled_neg_mask[neg_positions] = True
                    
                    neg_windows = extract_context_windows(tokens, sampled_neg_mask, tokenizer, feature_acts, max_windows=50)
                    neg_contexts.extend(neg_windows)
        
        # Sort by activation and select top/bottom
        if pos_contexts:
            pos_contexts.sort(key=lambda x: x[1], reverse=True)
            top_pos = pos_contexts[:TOP_K_POS]
            pos_snippets = [ctx[0] for ctx in top_pos]
        else:
            pos_snippets = []
        
        if neg_contexts:
            neg_contexts.sort(key=lambda x: x[1], reverse=False)
            bottom_neg = neg_contexts[:TOP_K_NEG]
            neg_snippets = [ctx[0] for ctx in bottom_neg]
        else:
            neg_snippets = []
        
        # Calculate statistics
        if pos_contexts:
            pos_acts = [ctx[1] for ctx in pos_contexts]
            max_act = max(pos_acts)
            mean_act = np.mean(pos_acts)
        else:
            max_act = 0.0
            mean_act = 0.0
        
        if neg_contexts:
            neg_acts = [ctx[1] for ctx in neg_contexts]
            min_act = min(neg_acts)
        else:
            min_act = 0.0
        
        results[feat_idx] = {
            "pos_contexts": pos_snippets,
            "neg_contexts": neg_snippets,
            "max_activation": float(max_act),
            "mean_activation": float(mean_act),
            "min_activation": float(min_act)
        }
    
    # Save results
    with open("test_results/activating_sentences.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved activating context windows to test_results/activating_sentences.json")
    print(f"   Note: These are token-level context windows, not full sentences")

if __name__ == "__main__":
    main()
