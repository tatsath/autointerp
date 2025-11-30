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

# Config - Updated paths for new structure
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_LIST = os.path.join(BASE_DIR, "results", "1_search", "feature_list.json")
# Config path - can be overridden via environment or command line
CONFIG_PATH = os.getenv("CONFIG_PATH", os.path.join(BASE_DIR, "1. search", "domains", "finance", "config.json"))
MODEL_PATH = os.getenv("MODEL_PATH", "meta-llama/Llama-3.1-8B-Instruct")
SAE_PATH = os.getenv("SAE_PATH", "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU")
DATASET_PATH = os.getenv("DATASET_PATH", "jyanimaulik/yahoo_finance_stockmarket_news")
TOKENS_STR_PATH = os.getenv("TOKENS_STR_PATH", os.path.join(BASE_DIR, "1. search", "domains", "finance", "finance_tokens.json"))
LAYER = 19
N_SAMPLES = 500
TOP_K_POS = 20  # Top activating context windows
TOP_K_NEG = 20  # Bottom activating context windows (non-activating)

def load_sae(sae_path, layer):
    """
    Load SAE weights. Supports both standard (layers.{layer}) and FinBERT (encoder.layer.{layer}.output) formats.
    """
    import re
    
    # Check if this is a FinBERT-style path (encoder.layer.{layer}.output)
    finbert_match = re.search(r'encoder\.layer\.(\d+)\.output', sae_path)
    if finbert_match:
        # FinBERT: SAE is directly in the sae_path directory
        layer_dir = sae_path
    else:
        # Standard format: layers.{layer} subdirectory
        layer_dir = f"{sae_path}/layers.{layer}"
    
    # Try sae_weights.safetensors first (converted format), then sae.safetensors
    sae_file = f"{layer_dir}/sae_weights.safetensors"
    if not os.path.exists(sae_file):
        sae_file = f"{layer_dir}/sae.safetensors"
    
    if not os.path.exists(sae_file):
        raise FileNotFoundError(f"SAE file not found in {layer_dir}")
    
    with safe_open(sae_file, framework="pt", device="cpu") as f:
        # Handle both formats: converted (W_enc) and original (encoder.weight)
        if "W_enc" in f.keys():
            # Converted format
            encoder_weight = f.get_tensor("W_enc").T  # Transpose to [d_sae, d_in]
            encoder_bias = f.get_tensor("b_enc") if "b_enc" in f.keys() else torch.zeros(encoder_weight.shape[0])
        elif "encoder.weight" in f.keys():
            # Original format
            encoder_weight = f.get_tensor("encoder.weight")
            encoder_bias = f.get_tensor("encoder.bias") if "encoder.bias" in f.keys() else torch.zeros(encoder_weight.shape[0])
        else:
            raise ValueError(f"Unknown SAE format in {sae_file}. Expected W_enc or encoder.weight")
        
        return {"encoder": encoder_weight, "encoder_bias": encoder_bias}

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

def get_token_activations_batch(tokens_list, model, encoder, encoder_bias, layer, device, is_finbert=False):
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
    print("\n" + "=" * 80)
    print("🏷️  Collecting Activating Examples")
    print("=" * 80)
    
    # Allow override via environment variables
    search_output_dir = os.getenv("SEARCH_OUTPUT_DIR", None)
    config_path_to_use = CONFIG_PATH
    
    if search_output_dir:
        global FEATURE_LIST
        FEATURE_LIST = os.path.join(search_output_dir, "feature_list.json")
        print(f">>> Using search output from: {search_output_dir}")
        
        # Try to load config from search output directory first
        search_config_path = os.path.join(search_output_dir, "config.json")
        if os.path.exists(search_config_path):
            config_path_to_use = search_config_path
            print(f">>> Using config from search output: {search_config_path}")
    
    # Load config from same file used in feature search
    config_dir = os.path.dirname(os.path.abspath(config_path_to_use))
    if os.path.exists(config_path_to_use):
        with open(config_path_to_use, 'r') as f:
            config = json.load(f)
    else:
        print(f"⚠️  Warning: Config file not found at {config_path_to_use}, using defaults")
        config = {}
    
    # Use EXACT same parameters as feature search
    EXPAND_RANGE = tuple(config.get('expand_range', [1, 2]))
    IGNORE_TOKENS = config.get('ignore_tokens', [])
    # Get paths from config, resolve relative paths relative to base directory
    dataset_path = config.get('dataset_path', DATASET_PATH)
    tokens_str_path = config.get('tokens_str_path', TOKENS_STR_PATH)
    if not os.path.isabs(tokens_str_path):
        # Resolve relative to base directory (where the original config would be)
        tokens_str_path = os.path.join(BASE_DIR, "1. search", tokens_str_path)
    
    # Get model and SAE paths from config (from search output)
    model_path = config.get('model_path', MODEL_PATH)
    sae_path = config.get('sae_path', SAE_PATH)
    sae_id = config.get('sae_id', None)
    
    # Extract layer from sae_id or sae_path if available
    layer = LAYER
    import re
    if sae_id:
        # Try standard format: blocks.{layer}
        layer_match = re.search(r'blocks\.(\d+)', sae_id)
        if layer_match:
            layer = int(layer_match.group(1))
        else:
            # Try FinBERT format: encoder.layer.{layer}.output
            finbert_match = re.search(r'encoder\.layer\.(\d+)\.', sae_id)
            if finbert_match:
                layer = int(finbert_match.group(1))
    # Also try extracting from sae_path (for FinBERT)
    if layer == LAYER:  # Only if not already extracted
        finbert_path_match = re.search(r'encoder\.layer\.(\d+)\.output', sae_path)
        if finbert_path_match:
            layer = int(finbert_path_match.group(1))
    
    print(f"Using config from: {config_path_to_use}")
    print(f"expand_range: {EXPAND_RANGE}")
    print(f"ignore_tokens: {IGNORE_TOKENS}")
    print(f"dataset_path: {dataset_path}")
    print(f"tokens_str_path: {tokens_str_path}")
    print(f"model_path: {model_path}")
    print(f"sae_path: {sae_path}")
    print(f"layer: {layer}")
    
    # Load feature list
    print("\n📋 Step 1: Loading feature list...")
    with open(FEATURE_LIST) as f:
        data = json.load(f)
    feature_indices = data["feature_indices"]
    print(f">>> Found {len(feature_indices)} features to process")
    
    # Load model & SAE
    print("\n📋 Step 2: Loading model and SAE...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Handle models that need trust_remote_code
    models_requiring_trust_remote_code = ["nemotron", "nvidia"]
    needs_trust_remote_code = any(keyword in model_path.lower() for keyword in models_requiring_trust_remote_code)
    
    # Handle FinBERT (BERT model, not causal LM)
    is_finbert = "finbert" in model_path.lower()
    if is_finbert:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.float16 if device=="cuda" else torch.float32,
            trust_remote_code=needs_trust_remote_code
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            dtype=torch.float16 if device=="cuda" else torch.float32,
            trust_remote_code=needs_trust_remote_code
        )
    if device == "cuda":
        model = model.to(device)
    model.eval()
    sae_weights = load_sae(sae_path, layer)
    
    # Store is_finbert flag for later use
    globals()['is_finbert'] = is_finbert
    
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
    # Check if it's a saved dataset (from save_to_disk)
    if os.path.isdir(dataset_path) and os.path.exists(os.path.join(dataset_path, "dataset_info.json")):
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, dict) and "train" in dataset:
            dataset = dataset["train"]
    else:
        dataset = load_dataset(dataset_path, split="train", streaming=False)
    dataset = dataset.select(range(min(N_SAMPLES * 2, len(dataset))))
    
    # Pre-process and tokenize all texts in batches
    print("Pre-processing dataset...")
    # Reduce batch size for Nemotron models (they use more memory)
    if "nemotron" in model_path.lower() or "nvidia" in model_path.lower():
        batch_size = 2  # Process 2 texts at a time for Nemotron
    else:
        batch_size = 8  # Process 8 texts at a time
    tokenized_texts = []
    texts_list = []
    
    # FinBERT has max_length of 512, truncate if needed
    max_length = 512 if is_finbert else None
    
    for example in tqdm(dataset, desc="Tokenizing", total=min(N_SAMPLES, len(dataset))):
        text = example.get('text', '') or example.get('content', '') or str(example)
        if not text or len(text.strip()) < 10:
            continue
        
        try:
            tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt", 
                                     max_length=max_length, truncation=(max_length is not None))
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
    print("\n📋 Step 3: Collecting context windows for each feature...")
    print(f">>> Processing {len(feature_indices)} features across {len(tokenized_texts)} texts")
    print(">>> This may take several minutes...")
    print()
    results = {}
    for feat_idx in tqdm(feature_indices, desc="Processing features"):
        pos_contexts = []  # (context_text, activation_value)
        neg_contexts = []  # (context_text, activation_value)
        
        # Process texts in batches
        for batch_start in range(0, len(tokenized_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(tokenized_texts))
            batch_tokens = tokenized_texts[batch_start:batch_end]
            
            # Get activations for batch
            batch_acts = get_token_activations_batch(batch_tokens, model, encoder, encoder_bias, layer, device, is_finbert=globals().get('is_finbert', False))
            
            # Process each text in batch
            for i, tokens in enumerate(batch_tokens):
                # Check if feature index is valid for this SAE
                if feat_idx >= batch_acts[i].shape[1]:
                    print(f"⚠️  Warning: Feature {feat_idx} is out of bounds (SAE has {batch_acts[i].shape[1]} features), skipping...")
                    continue
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
    
    # Save results - use model-specific output directory if provided
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "results", "2_labeling_lite"))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "activating_sentences.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Saved activating context windows to {output_file}")
    print(f"   Note: These are token-level context windows, not full sentences")

if __name__ == "__main__":
    main()
