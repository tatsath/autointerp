#!/usr/bin/env python3
"""
Extract top activating examples for finance features using SaeVisRunner.
Similar to SAE-Reasoning paper approach but outputs JSONL instead of HTML.
"""

import os
import json
import fire
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset

# Patch transformer_lens for Nemotron support (before importing HookedTransformer)
import transformer_lens.loading_from_pretrained as loading_from_pretrained
_original_get_official_model_name = loading_from_pretrained.get_official_model_name

def patched_get_official_model_name(model_name: str):
    """Patched version that supports Nemotron and FinBERT."""
    if "finbert" in model_name.lower():
        return "google-bert/bert-base-uncased"
    if "nemotron" in model_name.lower() or ("nvidia" in model_name.lower() and "nemotron" in model_name.lower()):
        # Nemotron is Llama-based, use Llama-3.1-8B-Instruct as base
        return "meta-llama/Llama-3.1-8B-Instruct"
    return _original_get_official_model_name(model_name)

loading_from_pretrained.get_official_model_name = patched_get_official_model_name

from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


def _detect_activation_extraction_path(model_path: str, layer: int, sae_d_in: int, model_d_model: int):
    """
    Auto-detect the path to extract activations for models with dimension mismatches.
    
    Returns a config dict with extraction path information, or None if cannot be determined.
    """
    model_name_lower = model_path.lower()
    
    # Known architectures and their activation extraction paths
    # Format: (model_pattern, extraction_path, expected_dim_pattern)
    known_patterns = [
        # Nemotron: mixer activations
        (["nemotron", "nvidia"], ["backbone", "layers", layer, "mixer"], 4480),
        # Llama/GPT: standard transformer blocks
        (["llama", "gpt", "mistral"], ["transformer", "h", layer], None),
        # Generic transformer
        (["transformer"], ["transformer", "h", layer], None),
    ]
    
    for patterns, path, expected_dim in known_patterns:
        if any(pattern in model_name_lower for pattern in patterns):
            if expected_dim is None or expected_dim == sae_d_in:
                return {
                    "type": "standard" if "mixer" not in path else "mixer",
                    "path": path,
                    "expected_dim": sae_d_in
                }
    
    # Fallback: try common structures
    return {
        "type": "auto_detect",
        "path": None,  # Will be determined at runtime
        "expected_dim": sae_d_in
    }


def load_sae_local(sae_path: str, sae_id: str, device: str):
    """Load SAE from local path (exact same approach as compute_score.py)."""
    # Support both standard (blocks.{layer}) and FinBERT (encoder.layer.{layer}.output) formats
    layer_match = re.match(r"blocks\.(\d+)\.", sae_id)
    finbert_match = None
    layer_num = None
    
    if layer_match:
        layer_num = int(layer_match.group(1))
        layer_path = os.path.join(sae_path, f"layers.{layer_num}")
    else:
        # Try FinBERT format: encoder.layer.{layer}.output
        finbert_match = re.search(r"encoder\.layer\.(\d+)\.", sae_id)
        if finbert_match:
            layer_num = int(finbert_match.group(1))
            # For FinBERT, sae_path already points to the layer directory
            layer_path = sae_path
        else:
            # Also try extracting from sae_path if sae_id doesn't match
            finbert_path_match = re.search(r"encoder\.layer\.(\d+)\.output", sae_path)
            if finbert_path_match:
                layer_num = int(finbert_path_match.group(1))
                layer_path = sae_path
            else:
                raise ValueError(f"Invalid sae_id format: {sae_id}. Expected 'blocks.N.hook_resid_post' or 'encoder.layer.N.output'")
    
    if not os.path.exists(layer_path):
        raise FileNotFoundError(f"SAE layer path not found: {layer_path}")
    
    print(f">>> Loading SAE from local path: {layer_path}")
    
    # Load config
    cfg_file = os.path.join(layer_path, "cfg.json")
    with open(cfg_file, 'r') as f:
        cfg_dict = json.load(f)
    
    # Map num_latents to d_sae if needed
    if "num_latents" in cfg_dict and "d_sae" not in cfg_dict:
        cfg_dict["d_sae"] = cfg_dict["num_latents"]
    
    # Map d_in to d_in if needed
    if "d_in" not in cfg_dict and "d_model" in cfg_dict:
        cfg_dict["d_in"] = cfg_dict["d_model"]
    
    # Convert FinBERT hook_name to transformer_lens format
    hook_name = sae_id
    if finbert_match or "encoder.layer" in sae_id:
        # Convert encoder.layer.10.output -> blocks.10.hook_resid_post
        hook_name = f"blocks.{layer_num}.hook_resid_post"
    
    # Add all required fields with defaults
    defaults = {
        "hook_name": hook_name,
        "hook_layer": layer_num,
        "hook_head_index": None,
        "architecture": "standard",
        "activation_fn_str": cfg_dict.get("activation_fn_str", cfg_dict.get("activation", "topk")),
        "k": cfg_dict.get("k", 16),
        "apply_b_dec_to_input": False,
        "finetuning_scaling_factor": False,
        "dataset_trust_remote_code": False,
        "normalize_activations": "none",
        "prepend_bos": False,
        "device": device,
        "sae_lens_training_version": None
    }
    for key, value in defaults.items():
        if key not in cfg_dict:
            cfg_dict[key] = value
    
    # Ensure k is in config for TopK
    if cfg_dict.get("activation_fn_str") == "topk" and "activation_fn_kwargs" not in cfg_dict:
        cfg_dict["activation_fn_kwargs"] = {"k": cfg_dict.get("k", 16)}
    
    # Create SAE config and model - use SAEConfig.from_dict which handles the class mismatch
    from sae_lens import SAEConfig, TopKSAE, StandardSAE
    sae_cfg = SAEConfig.from_dict(cfg_dict)
    
    # Determine which SAE class to use based on config type
    cfg_type_name = type(sae_cfg).__name__
    if 'TopK' in cfg_type_name:
        sae = TopKSAE(sae_cfg).to(device)
    else:
        sae = StandardSAE(sae_cfg).to(device)
    
    # Load weights with key conversion
    from safetensors import safe_open
    weights_file = os.path.join(layer_path, "sae.safetensors")
    if not os.path.exists(weights_file):
        weights_file = os.path.join(layer_path, "sae_weights.safetensors")
    
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"SAE weights file not found in {layer_path}")
    
    state_dict = {}
    with safe_open(weights_file, framework="pt", device=device) as f:
        if "encoder.weight" in f.keys():
            state_dict["W_enc"] = f.get_tensor("encoder.weight").T
        elif "W_enc" in f.keys():
            state_dict["W_enc"] = f.get_tensor("W_enc")
        
        if "encoder.bias" in f.keys():
            state_dict["b_enc"] = f.get_tensor("encoder.bias")
        elif "b_enc" in f.keys():
            state_dict["b_enc"] = f.get_tensor("b_enc")
        else:
            state_dict["b_enc"] = torch.zeros(state_dict["W_enc"].shape[1], device=device)
        
        if "W_dec" in f.keys():
            state_dict["W_dec"] = f.get_tensor("W_dec")
        
        if "b_dec" in f.keys():
            state_dict["b_dec"] = f.get_tensor("b_dec")
        else:
            state_dict["b_dec"] = torch.zeros(state_dict["W_enc"].shape[0], device=device)
    
    sae.load_state_dict(state_dict)
    return sae


def _process_sequence_data(seq_data, all_activation_data, feature_id=None, debug=False):
    """Helper function to process a SequenceData object and add to all_activation_data."""
    feat_acts = seq_data.feat_acts
    original_index = seq_data.original_index
    qualifying_token_index = seq_data.qualifying_token_index
    
    # Convert to lists if tensors
    if isinstance(feat_acts, torch.Tensor):
        feat_acts = feat_acts.cpu().tolist()
    elif not isinstance(feat_acts, list):
        feat_acts = [feat_acts]
    
    # original_index and qualifying_token_index might be single values or lists
    if isinstance(original_index, torch.Tensor):
        original_index = original_index.cpu().tolist()
    if isinstance(qualifying_token_index, torch.Tensor):
        qualifying_token_index = qualifying_token_index.cpu().tolist()
    
    # If they're single values, make them lists
    if not isinstance(original_index, list):
        original_index = [original_index]
    if not isinstance(qualifying_token_index, list):
        qualifying_token_index = [qualifying_token_index]
    
    if len(feat_acts) == 0:
        return
    
    # Debug: Check if activations are all zeros
    if debug and feature_id is not None:
        non_zero_acts = [a for a in feat_acts if abs(float(a)) > 1e-6]
        if len(non_zero_acts) == 0:
            print(f"      WARNING: Feature {feature_id} has all zero activations! feat_acts type: {type(seq_data.feat_acts)}, shape: {getattr(seq_data.feat_acts, 'shape', 'N/A') if hasattr(seq_data.feat_acts, 'shape') else 'N/A'}")
        else:
            print(f"      Feature {feature_id}: {len(non_zero_acts)}/{len(feat_acts)} non-zero activations, max: {max(non_zero_acts):.4f}")
    
    # Get token_ids if available
    token_ids = None
    if hasattr(seq_data, 'token_ids'):
        token_ids = seq_data.token_ids
    
    # Create activation data tuples
    # feat_acts, original_index, and qualifying_token_index should have the same length
    min_len = min(len(feat_acts), len(original_index), len(qualifying_token_index))
    for i in range(min_len):
        act_val = float(feat_acts[i])
        seq_idx = int(original_index[i])
        token_idx = int(qualifying_token_index[i])
        all_activation_data.append((act_val, seq_idx, token_idx, i, token_ids))


def extract_examples_from_sae_vis_data(sae_vis_data, feature_id: int, max_examples: int = 20, tokenizer=None, tokens_dataset=None):
    """
    Extract examples from SaeVisData structure.
    Follows the same pattern as save_feature_centric_vis in sae_dashboard.
    """
    examples = []
    
    if not hasattr(sae_vis_data, 'feature_data_dict'):
        return examples
    
    feature_data_dict = sae_vis_data.feature_data_dict
    if feature_id not in feature_data_dict:
        return examples
    
    feature_data = feature_data_dict[feature_id]
    
    # Collect all examples from all sources
    all_activation_data = []
    
    # Try prompt_data first (often has better activation data)
    if hasattr(feature_data, 'prompt_data') and feature_data.prompt_data is not None:
        prompt_data = feature_data.prompt_data
        if hasattr(prompt_data, 'feat_acts') and hasattr(prompt_data, 'original_index') and hasattr(prompt_data, 'qualifying_token_index'):
            # prompt_data is a SequenceData directly
            debug_first = feature_id == list(sae_vis_data.feature_data_dict.keys())[0] if hasattr(sae_vis_data, 'feature_data_dict') and len(sae_vis_data.feature_data_dict) > 0 else False
            if debug_first:
                print(f"      Trying prompt_data for feature {feature_id}")
            _process_sequence_data(prompt_data, all_activation_data, feature_id=feature_id, debug=debug_first)
    
    # Access sequence_data.seq_group_data (SequenceMultiGroupData contains grouped examples)
    seq_group_data = None
    if hasattr(feature_data, 'sequence_data'):
        sequence_data = feature_data.sequence_data
        if hasattr(sequence_data, 'seq_group_data') and sequence_data.seq_group_data is not None:
            seq_group_data = sequence_data.seq_group_data
    
    if seq_group_data is None:
        # If we got data from prompt_data, use it
        if len(all_activation_data) > 0:
            # Sort and extract top examples
            all_activation_data.sort(key=lambda x: x[0], reverse=True)
            for act_val, seq_idx, token_idx, data_idx, token_ids in all_activation_data[:max_examples]:
                example_dict = {
                    'activation': act_val,
                    'sequence_index': seq_idx,
                    'token_position': token_idx,
                }
                # Extract text (same logic as below)
                if tokens_dataset is not None and seq_idx < len(tokens_dataset):
                    try:
                        seq_tokens = tokens_dataset[seq_idx]["tokens"]
                        if isinstance(seq_tokens, torch.Tensor):
                            seq_tokens = seq_tokens.cpu().tolist()
                        context_window = 20
                        start_idx = max(0, token_idx - context_window)
                        end_idx = min(len(seq_tokens), token_idx + context_window + 1)
                        context_tokens = seq_tokens[start_idx:end_idx]
                        if tokenizer is not None and len(context_tokens) > 0:
                            decoded_text = tokenizer.decode(context_tokens, skip_special_tokens=True).strip()
                            if len(decoded_text) > 300:
                                decoded_text = decoded_text[:300] + "..."
                            example_dict['text'] = decoded_text
                        else:
                            example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
                    except Exception as e:
                        example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
                else:
                    example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
                examples.append(example_dict)
            return examples
        return examples
    
    # Debug: Check what's in seq_group_data
    if feature_id == list(sae_vis_data.feature_data_dict.keys())[0]:
        print(f"      DEBUG feature {feature_id}: seq_group_data type={type(seq_group_data)}")
        if isinstance(seq_group_data, dict):
            print(f"      DEBUG: seq_group_data is dict with {len(seq_group_data)} keys: {list(seq_group_data.keys())[:5]}")
        elif isinstance(seq_group_data, list):
            print(f"      DEBUG: seq_group_data is list with {len(seq_group_data)} items")
            if len(seq_group_data) > 0:
                print(f"      DEBUG: first item type={type(seq_group_data[0])}, has feat_acts={hasattr(seq_group_data[0], 'feat_acts')}")
                if hasattr(seq_group_data[0], 'feat_acts'):
                    feat_acts = seq_group_data[0].feat_acts
                    print(f"      DEBUG: first item feat_acts len={len(feat_acts) if hasattr(feat_acts, '__len__') else 'N/A'}")
    
    # seq_group_data can be a dict or list of SequenceData objects
    # Collect all examples from all groups
    all_activation_data = []
    
    if isinstance(seq_group_data, dict):
        groups = seq_group_data.values()
    elif isinstance(seq_group_data, list):
        groups = seq_group_data
    else:
        return examples
    
    # Debug flag for first feature
    debug_first = feature_id == list(sae_vis_data.feature_data_dict.keys())[0] if hasattr(sae_vis_data, 'feature_data_dict') and len(sae_vis_data.feature_data_dict) > 0 else False
    
    for group_data in groups:
        # Check if it's SequenceGroupData (has seq_data) or SequenceData (has feat_acts directly)
        if hasattr(group_data, 'seq_data'):
            # It's SequenceGroupData, iterate through seq_data list
            seq_data_list = group_data.seq_data
            if not isinstance(seq_data_list, list):
                continue
            for actual_seq_data in seq_data_list:
                if not hasattr(actual_seq_data, 'feat_acts'):
                    continue
                _process_sequence_data(actual_seq_data, all_activation_data, feature_id=feature_id, debug=debug_first)
        elif hasattr(group_data, 'feat_acts'):
            # It's SequenceData directly
            _process_sequence_data(group_data, all_activation_data, feature_id=feature_id, debug=debug_first)
        else:
            continue
    
    # Sort by activation descending, but if all activations are zero, prioritize diversity
    all_activation_data.sort(key=lambda x: x[0], reverse=True)
    
    # Check if all activations are zero and if all examples are from the same sequence
    all_zero = all(len(str(x[0])) == 3 and x[0] == 0.0 for x in all_activation_data) if all_activation_data else True
    unique_sequences = set(x[1] for x in all_activation_data) if all_activation_data else set()
    all_same_sequence = len(unique_sequences) <= 1
    
    # If all examples are from the same sequence and we have tokens_dataset, sample from different sequences
    if all_same_sequence and tokens_dataset is not None and len(tokens_dataset) > 1:
        print(f"      WARNING: All examples from same sequence ({list(unique_sequences)[0] if unique_sequences else 'N/A'}), sampling from different sequences")
        # Sample diverse examples from different sequences
        selected_examples = []
        seen_sequences = set()
        import random
        # Get random sequence indices
        available_sequences = list(range(min(len(tokens_dataset), 1000)))  # Limit to first 1000 sequences
        random.shuffle(available_sequences)
        
        for seq_idx in available_sequences:
            if len(selected_examples) >= max_examples:
                break
            if seq_idx in seen_sequences:
                continue
            
            # Get a random token position from this sequence
            try:
                seq_tokens = tokens_dataset[seq_idx]["tokens"]
                if isinstance(seq_tokens, torch.Tensor):
                    seq_tokens = seq_tokens.cpu().tolist()
                seq_len = len(seq_tokens)
                if seq_len > 10:
                    # Sample a token position (avoid first/last few tokens)
                    token_idx = random.randint(10, seq_len - 10)
                    selected_examples.append((0.0, seq_idx, token_idx, 0, None))
                    seen_sequences.add(seq_idx)
            except Exception as e:
                continue
    else:
        # Extract top examples with diversity
        selected_examples = []
        seen_sequences = set()
        seen_text_keys = set()
        
        # First pass: get examples with highest activations (if not all zero)
        for act_val, seq_idx, token_idx, data_idx, token_ids in all_activation_data:
            if len(selected_examples) >= max_examples:
                break
            
            # Create a text key to avoid duplicate text
            text_key = f"{seq_idx}_{token_idx}"
            
            # If all activations are zero, prioritize diversity (different sequences)
            if all_zero:
                # Prioritize different sequences
                if seq_idx not in seen_sequences:
                    selected_examples.append((act_val, seq_idx, token_idx, data_idx, token_ids))
                    seen_sequences.add(seq_idx)
                    seen_text_keys.add(text_key)
            else:
                # Normal case: use activation values
                if text_key not in seen_text_keys:
                    selected_examples.append((act_val, seq_idx, token_idx, data_idx, token_ids))
                    seen_text_keys.add(text_key)
                    seen_sequences.add(seq_idx)
        
        # If we still need more examples and all were zero, get diverse ones
        if len(selected_examples) < max_examples and all_zero:
            for act_val, seq_idx, token_idx, data_idx, token_ids in all_activation_data:
                if len(selected_examples) >= max_examples:
                    break
                text_key = f"{seq_idx}_{token_idx}"
                if text_key not in seen_text_keys:
                    selected_examples.append((act_val, seq_idx, token_idx, data_idx, token_ids))
                    seen_text_keys.add(text_key)
    
    # Extract top examples
    for act_val, seq_idx, token_idx, data_idx, token_ids in selected_examples:
        example_dict = {
            'activation': act_val,
            'sequence_index': seq_idx,
            'token_position': token_idx,
        }
        
        # Try to get text from tokens_dataset if available
        if tokens_dataset is not None and seq_idx < len(tokens_dataset):
            try:
                # Get the sequence tokens
                seq_tokens = tokens_dataset[seq_idx]["tokens"]
                if isinstance(seq_tokens, torch.Tensor):
                    seq_tokens = seq_tokens.cpu().tolist()
                elif isinstance(seq_tokens, list):
                    pass  # Already a list
                else:
                    raise ValueError(f"Unexpected seq_tokens type: {type(seq_tokens)}")
                
                # Get context around the token (e.g., ±20 tokens for better context)
                context_window = 20
                start_idx = max(0, token_idx - context_window)
                end_idx = min(len(seq_tokens), token_idx + context_window + 1)
                context_tokens = seq_tokens[start_idx:end_idx]
                
                if tokenizer is not None and len(context_tokens) > 0:
                    # Decode the context tokens to get full text
                    decoded_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                    # Clean up the text
                    decoded_text = decoded_text.strip()
                    # Limit length for readability
                    if len(decoded_text) > 300:
                        decoded_text = decoded_text[:300] + "..."
                    example_dict['text'] = decoded_text
                else:
                    example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
            except Exception as e:
                print(f"      Warning: Failed to extract text for feature {feature_id}, seq {seq_idx}, token {token_idx}: {e}")
                example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
        elif tokenizer is not None and token_ids is not None:
            try:
                # Try to get token from token_ids
                if isinstance(token_ids, torch.Tensor):
                    if len(token_ids.shape) > 1:
                        tokens = token_ids[data_idx].cpu().tolist()
                    else:
                        tokens = token_ids[data_idx].cpu().tolist() if data_idx < len(token_ids) else [token_ids[0].item()]
                else:
                    tokens = token_ids[data_idx] if data_idx < len(token_ids) else token_ids[0]
                
                if isinstance(tokens, (list, torch.Tensor)):
                    if isinstance(tokens, torch.Tensor):
                        tokens = tokens.cpu().tolist()
                    example_dict['text'] = tokenizer.decode(tokens, skip_special_tokens=True)
                else:
                    example_dict['text'] = tokenizer.decode([tokens], skip_special_tokens=True)
            except Exception as e:
                example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
        else:
            example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
        
        examples.append(example_dict)
    
    return examples


def compute_examples_finance(
    model_path: str,
    sae_path: str,
    dataset_path: str,
    feature_list_path: str,
    output_path: str,
    sae_id: str = None,
    column_name: str = "text",
    minibatch_size_features: int = 256,
    minibatch_size_tokens: int = 64,
    n_samples: int = 5000,
    max_examples_per_feature: int = 20,
    device: str = None
):
    """
    Extract top activating examples for finance features.
    
    Args:
        model_path: Path to model (HuggingFace repo or local)
        sae_path: Path to SAE (HuggingFace repo or local directory)
        dataset_path: Path to dataset (HuggingFace repo)
        feature_list_path: Path to JSON file with feature indices and scores
        output_path: Path to output JSONL file
        sae_id: SAE identifier (e.g., "blocks.19.hook_resid_post")
        column_name: Dataset column name containing text
        minibatch_size_features: Batch size for feature processing
        minibatch_size_tokens: Batch size for token processing
        n_samples: Number of samples to process
        max_examples_per_feature: Maximum examples to extract per feature
        device: Device to use (e.g., "cuda:7", "cuda:8", or None for auto)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    
    print("\n" + "=" * 80)
    print("🔬 Extracting Feature Examples (Advanced)")
    print("=" * 80)
    
    # Step 1: Load feature list
    print("\n📋 Step 1: Loading feature list...")
    print(f">>> Loading from: {feature_list_path}")
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_indices = feature_data['feature_indices']
    feature_scores = feature_data['scores']
    
    # Create mapping from feature_id to score
    feature_score_map = {feat_id: score for feat_id, score in zip(feature_indices, feature_scores)}
    
    print(f">>> Found {len(feature_indices)} features to process")
    
    # Step 2: Load model and SAE
    print("\n📋 Step 2: Loading SAE and model...")
    print(f">>> Model: {model_path}")
    print(f">>> SAE: {sae_path}")
    is_local_path = os.path.exists(sae_path) and os.path.isdir(sae_path)
    
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    elif is_local_path:
        sae = load_sae_local(sae_path, sae_id, device)
    else:
        # HuggingFace repo path
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)
    
    # Load model - handle models that require trust_remote_code
    # Common models that need this: Nemotron, some custom models
    models_requiring_trust_remote_code = ["nemotron", "nvidia"]
    needs_trust_remote_code = any(
        keyword in model_path.lower() for keyword in models_requiring_trust_remote_code
    )
    
    # Use device_map="auto" if device is cuda to let transformers handle placement
    # Otherwise use the specified device
    if device and device.startswith("cuda"):
        # Try to use device_map for better memory management
        try:
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=needs_trust_remote_code,
            )
        except Exception as e:
            print(f">>> Warning: device_map='auto' failed: {e}")
            print(f">>> Falling back to device={device}")
            try:
                model = HookedTransformer.from_pretrained_no_processing(
                    model_path,
                    dtype=torch.bfloat16,
                    device=device,
                    trust_remote_code=needs_trust_remote_code,
                )
            except Exception as e2:
                # If loading fails, try with trust_remote_code=True as fallback
                if "trust_remote_code" not in str(e2).lower() and not needs_trust_remote_code:
                    print(f">>> Warning: Model loading failed, retrying with trust_remote_code=True...")
                    model = HookedTransformer.from_pretrained_no_processing(
                        model_path,
                        dtype=torch.bfloat16,
                        device=device,
                        trust_remote_code=True,
                    )
                else:
                    raise
    else:
        try:
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device=device,
                trust_remote_code=needs_trust_remote_code,
            )
        except Exception as e:
            # If loading fails, try with trust_remote_code=True as fallback
            if "trust_remote_code" not in str(e).lower() and not needs_trust_remote_code:
                print(f">>> Warning: Model loading failed, retrying with trust_remote_code=True...")
                model = HookedTransformer.from_pretrained_no_processing(
                    model_path,
                    dtype=torch.bfloat16,
                    device=device,
                    trust_remote_code=True,
                )
            else:
                raise
    
    # Make pad token different from bos/eos
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    print("\n📋 Loading dataset...")
    print(f">>> Dataset: {dataset_path}")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    print(f">>> Dataset loaded: {len(dataset):,} samples")
    
    if column_name == "tokens":
        token_dataset = dataset
    else:
        print(">>> Tokenizing dataset...")
        # Get context size from config or use default
        context_size = getattr(sae.cfg, 'context_size', None)
        if context_size is None:
            # Try metadata
            if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
                context_size = sae.cfg.metadata.get('context_size', None)
        if context_size is None:
            # Try config file
            if is_local_path:
                cfg_file = os.path.join(sae_path, "cfg.json")
                if os.path.exists(cfg_file):
                    with open(cfg_file, 'r') as f:
                        cfg_dict = json.load(f)
                    context_size = cfg_dict.get('context_size', None)
        if context_size is None:
            context_size = getattr(sae.cfg, 'max_length', 1024)
        
        # For FinBERT, ensure max_length is 512
        is_finbert_model = "finbert" in model_path.lower()
        if is_finbert_model and context_size > 512:
            print(f">>> FinBERT detected: limiting context_size from {context_size} to 512")
            context_size = 512
        
        # Get prepend_bos from config or use default (check both prepend_bos and prepend_bos_token)
        prepend_bos = getattr(sae.cfg, 'prepend_bos', getattr(sae.cfg, 'prepend_bos_token', False))
        # FinBERT doesn't use BOS token
        if is_finbert_model:
            prepend_bos = False
        
        token_dataset = tokenize_and_concatenate(
            dataset=dataset,
            tokenizer=model.tokenizer,
            streaming=False,
            max_length=context_size,
            column_name=column_name,
            add_bos_token=prepend_bos,
            num_proc=4
        )
    
    # Step 3: Create SaeVisConfig and run SaeVisRunner
    print("\n📋 Step 3: Running SaeVisRunner...")
    print(f">>> Processing {n_samples:,} samples")
    print(f">>> Extracting examples for {len(feature_indices)} features")
    print(">>> This may take several minutes...")
    print()
    
    # Get hook_point early (needed for dimension mismatch patch)
    hook_point = None
    if sae_id:
        hook_point = sae_id
    else:
        # Try to get from SAE config metadata or cfg
        if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
            hook_point = sae.cfg.metadata.get('hook_name', None)
        if not hook_point:
            hook_point = getattr(sae.cfg, 'hook_name', None)
        # If still None, try to extract from sae_path
        if not hook_point and is_local_path:
            cfg_file = os.path.join(sae_path, "cfg.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r') as f:
                    cfg_dict = json.load(f)
                hook_point = cfg_dict.get('hook_name', None)
    
    # Convert FinBERT format to transformer_lens format if needed
    if hook_point and "encoder.layer" in hook_point:
        finbert_match = re.search(r"encoder\.layer\.(\d+)\.", hook_point)
        if finbert_match:
            layer_num = int(finbert_match.group(1))
            hook_point = f"blocks.{layer_num}.hook_resid_post"
    
    if hook_point is None:
        raise ValueError("Could not determine hook_point. Please provide sae_id parameter or ensure SAE config has hook_name.")
    
    # Extract layer number early (needed for dimension mismatch detection)
    layer_match = re.match(r"blocks\.(\d+)\.", hook_point)
    if not layer_match:
        # Try FinBERT format: encoder.layer.{layer}.output or blocks.{layer}.hook_resid_post
        finbert_match = re.search(r"encoder\.layer\.(\d+)\.", hook_point)
        if not finbert_match:
            finbert_match = re.search(r"blocks\.(\d+)\.", hook_point)
        if finbert_match:
            layer_match = finbert_match
    hook_layer = int(layer_match.group(1)) if layer_match else None
    
    # Check for dimension mismatches between model and SAE
    sae_d_in = getattr(sae.cfg, 'd_in', None)
    model_d_model = getattr(model.cfg, 'd_model', None) if hasattr(model, 'cfg') else None
    
    # Detect if we need special handling
    needs_special_patch = False
    patch_type = None
    activation_extraction_config = None
    
    if sae_d_in is not None:
        # Check if SAE d_in matches model d_model
        if model_d_model is not None and sae_d_in != model_d_model:
            print(f">>> Dimension mismatch detected: SAE d_in={sae_d_in}, Model d_model={model_d_model}")
            needs_special_patch = True
            
            if hook_layer is None:
                print(f">>> Warning: Could not extract layer number from sae_id: {sae_id}")
                print(f">>> Skipping dimension mismatch patch - may fail later")
            else:
                # Determine patch type and extraction method based on model architecture
                model_name_lower = model_path.lower()
                is_nemotron = "nemotron" in model_name_lower or ("nvidia" in model_name_lower and "nemotron" in model_name_lower)
                
                if is_nemotron and sae_d_in == 4480:
                    # Nemotron mixer dimension (4480) - specific handling
                    patch_type = "nemotron_mixer"
                    activation_extraction_config = {
                        "type": "mixer",
                        "path": ["backbone", "layers", hook_layer, "mixer"],
                        "expected_dim": 4480
                    }
                else:
                    # Use generic detection
                    patch_type = "generic_mismatch"
                    print(f">>> Generic dimension mismatch detected. Auto-detecting architecture...")
                    activation_extraction_config = _detect_activation_extraction_path(
                        model_path, hook_layer, sae_d_in, model_d_model
                    )
        elif model_d_model is None:
            # Can't verify, but check for known special cases
            model_name_lower = model_path.lower()
            is_nemotron = "nemotron" in model_name_lower or ("nvidia" in model_name_lower and "nemotron" in model_name_lower)
            if is_nemotron and sae_d_in == 4480:
                needs_special_patch = True
                patch_type = "nemotron_mixer"
                activation_extraction_config = {
                    "type": "mixer",
                    "path": ["backbone", "layers", hook_layer, "mixer"],
                    "expected_dim": 4480
                }
    
    # Patch model hook for special cases if needed
    raw_model = None
    if needs_special_patch:
        if patch_type == "nemotron_mixer":
            print(">>> Detected Nemotron with 4480-dim SAE (mixer dimension)")
            print(">>> Setting up Nemotron mixer activation extraction...")
        elif patch_type == "generic_mismatch":
            print(">>> Detected dimension mismatch between model and SAE")
            print(">>> Attempting to handle with raw transformers model...")
        else:
            raise ValueError(f"Unknown patch type: {patch_type}")
        from transformers import AutoModelForCausalLM
        
        if hook_layer is None:
            raise ValueError(f"Could not extract layer number from sae_id: {sae_id}")
        
        if activation_extraction_config is None:
            print(f">>> Warning: No activation extraction config determined")
            print(f">>> Attempting generic extraction...")
            activation_extraction_config = {
                "type": "auto_detect",
                "path": None,
                "expected_dim": sae_d_in
            }
        
        # Load raw transformers model for special activation extraction
        print(f">>> Loading raw transformers model for layer {hook_layer}...")
        raw_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": device} if device.startswith("cuda") else None,
            torch_dtype=torch.bfloat16
        )
        raw_model.eval()
        
        # Store activation extraction config for use in patching
        _activation_extraction_config = activation_extraction_config
        _sae_d_in = sae_d_in
        _hook_layer = hook_layer
        
        # Patch the model's run_with_hooks to intercept and replace activations
        original_run_with_hooks = model.run_with_hooks
        
        def _extract_special_activations(raw_model, tokens, extraction_config, layer):
            """Generic function to extract activations from raw model based on config."""
            special_activations = []
            
            def special_hook_fn(module, input, output):
                if isinstance(output, tuple):
                    special_activations.append(output[0].detach())
                else:
                    special_activations.append(output.detach())
            
            # Navigate to the target module based on extraction path
            target_module = raw_model
            if extraction_config and extraction_config.get("path"):
                path = extraction_config["path"]
                try:
                    for attr in path:
                        if isinstance(attr, int):
                            target_module = target_module[attr]
                        else:
                            target_module = getattr(target_module, attr)
                except (AttributeError, IndexError, TypeError) as e:
                    print(f">>> Error navigating extraction path: {e}")
                    return None
            else:
                # Auto-detect: try common structures
                if hasattr(raw_model, 'backbone') and hasattr(raw_model.backbone, 'layers'):
                    if hasattr(raw_model.backbone.layers[layer], 'mixer'):
                        target_module = raw_model.backbone.layers[layer].mixer
                    else:
                        target_module = raw_model.backbone.layers[layer]
                elif hasattr(raw_model, 'transformer') and hasattr(raw_model.transformer, 'h'):
                    target_module = raw_model.transformer.h[layer]
                else:
                    print(f">>> Error: Cannot auto-detect layer structure")
                    return None
            
            handle = target_module.register_forward_hook(special_hook_fn)
            
            try:
                with torch.no_grad():
                    device_for_model = next(raw_model.parameters()).device
                    _ = raw_model(input_ids=tokens.to(device_for_model))
            finally:
                handle.remove()
            
            if special_activations:
                return special_activations[0]
            return None
        
        def patched_run_with_hooks(self, *args, fwd_hooks=None, **kwargs):
            """Patched version that replaces hook activations with special activations for dimension mismatches."""
            if fwd_hooks is None:
                return original_run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
            
            # Check if the hook is for our target layer
            target_hook = None
            for hook_name, hook_fn in fwd_hooks:
                if hook_name == hook_point:
                    target_hook = (hook_name, hook_fn)
                    break
            
            if target_hook is None:
                return original_run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
            
            # Extract tokens from args (first arg should be tokens after self)
            tokens = args[0] if args else kwargs.get('input', None)
            if tokens is None or not isinstance(tokens, torch.Tensor):
                return original_run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
            
            # Extract special activations using generic extraction function
            special_act = _extract_special_activations(raw_model, tokens, _activation_extraction_config, _hook_layer)
            
            if special_act is not None:
                # Convert to correct device and dtype
                special_act = special_act.to(device=device, dtype=torch.bfloat16)
                
                # Verify dimension matches SAE
                if special_act.shape[-1] != _sae_d_in:
                    print(f">>> Warning: Extracted activation dim {special_act.shape[-1]} doesn't match SAE d_in {_sae_d_in}")
                    print(f">>> Falling back to standard path (may fail)")
                    return original_run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
                
                # Create a custom hook that stores special activations
                def custom_hook_fn(activation, hook):
                    # Replace with special activations - ensure shape matches expected format
                    # activation might be [batch, seq, dim] or [batch*seq, dim]
                    if len(special_act.shape) == 3 and len(activation.shape) == 2:
                        # Flatten batch and seq dimensions
                        special_act_flat = special_act.reshape(-1, special_act.shape[-1])
                        hook.ctx["activation"] = special_act_flat
                    elif len(special_act.shape) == 2 and len(activation.shape) == 3:
                        # Reshape to match 3D format
                        batch_size = activation.shape[0]
                        seq_len = activation.shape[1]
                        special_act_3d = special_act.reshape(batch_size, seq_len, -1)
                        hook.ctx["activation"] = special_act_3d
                    else:
                        hook.ctx["activation"] = special_act
                
                # Replace the hook function
                new_fwd_hooks = []
                for hook_name, hook_fn in fwd_hooks:
                    if hook_name == hook_point:
                        new_fwd_hooks.append((hook_name, custom_hook_fn))
                    else:
                        new_fwd_hooks.append((hook_name, hook_fn))
                
                return original_run_with_hooks(*args, fwd_hooks=new_fwd_hooks, **kwargs)
            else:
                print(f">>> Warning: Could not extract special activations, falling back to standard path")
                return original_run_with_hooks(*args, fwd_hooks=fwd_hooks, **kwargs)
        
        # Apply patch - bind the function to the model instance
        import types
        model.run_with_hooks = types.MethodType(patched_run_with_hooks, model)
        
        # Also patch the model's forward method to handle dimension mismatches in logit computation
        # This ensures consistency when SaeVisRunner computes logits
        original_forward = model.forward
        
        def patched_forward(self, *args, **kwargs):
            """Patched forward that ensures dimension consistency for logit computation."""
            # For now, just pass through - the hook patch should handle activations
            # If logit computation still fails, we may need more sophisticated patching
            return original_forward(*args, **kwargs)
        
        model.forward = types.MethodType(patched_forward, model)
    
    # Ensure device is a string for SaeVisConfig
    device_str = str(device) if isinstance(device, torch.device) else device
    
    feature_vis_config = SaeVisConfig(
        hook_point=hook_point,
        features=feature_indices,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device_str
    )
    
    # Patch einops.einsum to handle dimension mismatches in logit computation
    # Only apply this patch if we detected a dimension mismatch to maintain backward compatibility
    original_einsum = None
    if needs_special_patch:
        import einops
        original_einsum = einops.einsum
        
        def patched_einsum(pattern, *tensors, **kwargs):
            """Patched einsum that handles dimension mismatches in logit computation."""
            try:
                return original_einsum(pattern, *tensors, **kwargs)
            except RuntimeError as e:
                if "einsum" in str(e) and "subscript" in str(e) and "size" in str(e):
                    # Dimension mismatch detected - this is likely in logit computation
                    # Check if this is the logit computation pattern: "feats d_model, d_model d_vocab -> feats d_vocab"
                    if len(tensors) >= 2 and isinstance(pattern, str) and "d_model" in pattern and "d_vocab" in pattern:
                        print(f"\n>>> Warning: Dimension mismatch in logit computation (einsum)")
                        print(f">>> Skipping logit computation - returning zero logits")
                        print(f">>> Feature extraction data will still be available")
                        # Return zeros for logits to allow processing to continue
                        # The shape should be [feats, d_vocab] based on the pattern
                        if len(tensors) >= 2:
                            feats_dim = tensors[0].shape[0] if len(tensors[0].shape) >= 1 else 1
                            # Get d_vocab from model or second tensor
                            if len(tensors[1].shape) >= 1:
                                d_vocab = tensors[1].shape[-1]
                            else:
                                d_vocab = getattr(model.cfg, 'd_vocab', 50257)  # Default vocab size
                            device = tensors[0].device
                            dtype = tensors[0].dtype
                            # Return zero logits to allow processing to continue
                            return torch.zeros((feats_dim, d_vocab), device=device, dtype=dtype)
                    # For other einsum calls, re-raise
                    raise
                else:
                    raise
        
        # Apply patch only when dimension mismatch is detected
        einops.einsum = patched_einsum
    
    # Patch SAE's fold_W_dec_norm to skip for TopKSAE
    original_fold_W_dec_norm = None
    if hasattr(sae, 'fold_W_dec_norm'):
        original_fold_W_dec_norm = sae.fold_W_dec_norm
        def patched_fold_W_dec_norm(self):
            try:
                return original_fold_W_dec_norm()
            except NotImplementedError as e:
                if "TopKSAE" in str(e) or "topk" in str(e).lower() or "Folding W_dec_norm is not safe" in str(e):
                    print(f">>> Skipping fold_W_dec_norm() for TopK SAE (not safe for TopK)")
                    return
                else:
                    raise
        sae.fold_W_dec_norm = patched_fold_W_dec_norm.__get__(sae, type(sae))
    
    # Run SaeVisRunner (with patch applied if needed)
    # Prepare tokens - ensure we have the right format
    tokens_for_vis = token_dataset[:n_samples]["tokens"]
    if isinstance(tokens_for_vis, torch.Tensor):
        # If it's a single tensor, we might need to reshape
        if len(tokens_for_vis.shape) == 1:
            # Single sequence, reshape to [1, seq_len]
            tokens_for_vis = tokens_for_vis.unsqueeze(0)
        elif len(tokens_for_vis.shape) == 2:
            # Already in [batch, seq_len] format
            pass
    elif isinstance(tokens_for_vis, list):
        # Convert list to tensor
        if len(tokens_for_vis) > 0 and isinstance(tokens_for_vis[0], torch.Tensor):
            tokens_for_vis = torch.stack(tokens_for_vis)
        else:
            tokens_for_vis = torch.tensor(tokens_for_vis)
    
    print(f">>> Tokens shape for SaeVisRunner: {tokens_for_vis.shape if isinstance(tokens_for_vis, torch.Tensor) else type(tokens_for_vis)}")
    
    try:
        sae_vis_data = SaeVisRunner(feature_vis_config).run(
            encoder=sae,
            model=model,
            tokens=tokens_for_vis
        )
    except RuntimeError as e:
        if "einsum" in str(e) and "subscript" in str(e) and "size" in str(e):
            # Fallback error handling
            print(f"\n>>> Error: Dimension mismatch in logit computation")
            print(f">>> Error details: {e}")
            raise RuntimeError(
                f"Dimension mismatch in logit computation. "
                f"SAE d_in={sae_d_in}, Model d_model={model_d_model}. "
                f"Original error: {e}"
            ) from e
        else:
            raise
    finally:
        # Restore original einsum to maintain backward compatibility
        if needs_special_patch and original_einsum is not None:
            import einops
            einops.einsum = original_einsum
    
    # Step 4: Inspect structure (for debugging)
    print(">>> Inspecting SaeVisData structure...")
    print(f"    Type: {type(sae_vis_data)}")
    if hasattr(sae_vis_data, '__dict__'):
        print(f"    Attributes: {list(sae_vis_data.__dict__.keys())}")
    
    # Debug: Check feature_data_dict structure
    if hasattr(sae_vis_data, 'feature_data_dict'):
        print(f"    feature_data_dict type: {type(sae_vis_data.feature_data_dict)}")
        if isinstance(sae_vis_data.feature_data_dict, dict) and len(sae_vis_data.feature_data_dict) > 0:
            first_feat_id = list(sae_vis_data.feature_data_dict.keys())[0]
            first_feat_data = sae_vis_data.feature_data_dict[first_feat_id]
            print(f"    First feature ({first_feat_id}) type: {type(first_feat_data)}")
            if hasattr(first_feat_data, '__dict__'):
                print(f"    First feature attributes: {list(first_feat_data.__dict__.keys())}")
            elif isinstance(first_feat_data, dict):
                print(f"    First feature keys: {list(first_feat_data.keys())}")
            
            # Check sequence_data or prompt_data
            if hasattr(first_feat_data, 'sequence_data'):
                seq_data = first_feat_data.sequence_data
                print(f"    sequence_data type: {type(seq_data)}")
                if hasattr(seq_data, '__dict__'):
                    print(f"    sequence_data attributes: {list(seq_data.__dict__.keys())}")
            if hasattr(first_feat_data, 'prompt_data'):
                prompt_data = first_feat_data.prompt_data
                print(f"    prompt_data type: {type(prompt_data)}")
                if hasattr(prompt_data, '__dict__'):
                    print(f"    prompt_data attributes: {list(prompt_data.__dict__.keys())}")
    
    print("\n>>> SaeVisRunner complete!")
    
    # Step 4: Extract examples and write JSONL
    print("\n📋 Step 4: Extracting examples and saving...")
    print(f">>> Saving to: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Prepare tokens_dataset for extraction (limit to n_samples)
    tokens_dataset_for_extraction = None
    if 'token_dataset' in locals() and token_dataset is not None:
        try:
            # Convert to list if it's a dataset
            if hasattr(token_dataset, '__getitem__'):
                tokens_dataset_for_extraction = [token_dataset[i] for i in range(min(n_samples, len(token_dataset)))]
            else:
                tokens_dataset_for_extraction = token_dataset[:n_samples] if isinstance(token_dataset, list) else None
            print(f">>> Prepared tokens_dataset with {len(tokens_dataset_for_extraction)} sequences for extraction")
        except Exception as e:
            print(f">>> Warning: Could not prepare tokens_dataset: {e}")
            tokens_dataset_for_extraction = None
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for feat_id in feature_indices:
            examples = extract_examples_from_sae_vis_data(
                sae_vis_data, 
                feat_id, 
                max_examples=max_examples_per_feature,
                tokenizer=model.tokenizer,
                tokens_dataset=tokens_dataset_for_extraction
            )
            
            # Get score for this feature
            finance_score = feature_score_map.get(feat_id, 0.0)
            
            record = {
                "feature_id": feat_id,
                "finance_score": float(finance_score),
                "examples": examples
            }
            
            fout.write(json.dumps(record) + "\n")
            
            print(f"    Feature {feat_id}: extracted {len(examples)} examples")
    
    print(f">>> Done! Examples saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(compute_examples_finance)

