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
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


def load_sae_local(sae_path: str, sae_id: str, device: str):
    """Load SAE from local path (exact same approach as compute_score.py)."""
    layer_match = re.match(r"blocks\.(\d+)\.", sae_id)
    if not layer_match:
        raise ValueError(f"Invalid sae_id format: {sae_id}. Expected 'blocks.N.hook_resid_post'")
    
    layer_num = int(layer_match.group(1))
    layer_path = os.path.join(sae_path, f"layers.{layer_num}")
    
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
    
    # Add all required fields with defaults
    defaults = {
        "hook_name": sae_id,
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


def _process_sequence_data(seq_data, all_activation_data):
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
    
    # Access sequence_data.seq_group_data (SequenceMultiGroupData contains grouped examples)
    if not hasattr(feature_data, 'sequence_data'):
        return examples
    
    sequence_data = feature_data.sequence_data
    if not hasattr(sequence_data, 'seq_group_data') or sequence_data.seq_group_data is None:
        return examples
    
    seq_group_data = sequence_data.seq_group_data
    
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
                _process_sequence_data(actual_seq_data, all_activation_data)
        elif hasattr(group_data, 'feat_acts'):
            # It's SequenceData directly
            _process_sequence_data(group_data, all_activation_data)
        else:
            continue
    
    # Sort by activation descending and take top max_examples
    all_activation_data.sort(key=lambda x: x[0], reverse=True)
    
    # Extract top examples
    for act_val, seq_idx, token_idx, data_idx, token_ids in all_activation_data[:max_examples]:
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
                
                # Get context around the token (e.g., ±10 tokens)
                context_window = 10
                start_idx = max(0, token_idx - context_window)
                end_idx = min(len(seq_tokens), token_idx + context_window + 1)
                context_tokens = seq_tokens[start_idx:end_idx]
                
                if tokenizer is not None:
                    example_dict['text'] = tokenizer.decode(context_tokens, skip_special_tokens=True)
                else:
                    example_dict['text'] = f"Sequence {seq_idx}, token {token_idx}"
            except Exception as e:
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
    
    # Step 1: Load feature list
    print(">>> Loading feature list...")
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_indices = feature_data['feature_indices']
    feature_scores = feature_data['scores']
    
    # Create mapping from feature_id to score
    feature_score_map = {feat_id: score for feat_id, score in zip(feature_indices, feature_scores)}
    
    print(f">>> Found {len(feature_indices)} features to process")
    
    # Step 2: Load model and SAE
    print(">>> Loading SAE and LLM...")
    is_local_path = os.path.exists(sae_path) and os.path.isdir(sae_path)
    
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    elif is_local_path:
        sae = load_sae_local(sae_path, sae_id, device)
    else:
        # HuggingFace repo path
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)
    
    # Use device_map="auto" if device is cuda to let transformers handle placement
    # Otherwise use the specified device
    if device and device.startswith("cuda"):
        # Try to use device_map for better memory management
        try:
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            print(f">>> Warning: device_map='auto' failed: {e}")
            print(f">>> Falling back to device={device}")
            model = HookedTransformer.from_pretrained_no_processing(
                model_path,
                dtype=torch.bfloat16,
                device=device,
            )
    else:
        model = HookedTransformer.from_pretrained_no_processing(
            model_path,
            dtype=torch.bfloat16,
            device=device,
        )
    
    # Make pad token different from bos/eos
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    
    print(">>> Loading dataset...")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    
    if column_name == "tokens":
        token_dataset = dataset
    else:
        print(">>> Tokenizing dataset...")
        # Get context size from config or use default
        context_size = getattr(sae.cfg, 'context_size', None) or getattr(sae.cfg, 'max_length', 1024)
        # Get prepend_bos from config or use default (check both prepend_bos and prepend_bos_token)
        prepend_bos = getattr(sae.cfg, 'prepend_bos', getattr(sae.cfg, 'prepend_bos_token', False))
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
    print(">>> Creating SaeVisConfig...")
    # Ensure device is a string for SaeVisConfig
    device_str = str(device) if isinstance(device, torch.device) else device
    # Get hook_point from config or use sae_id
    hook_point = getattr(sae.cfg, 'hook_name', sae_id) if sae_id else getattr(sae.cfg, 'hook_name', None)
    if hook_point is None:
        raise ValueError("Could not determine hook_point. Please provide sae_id parameter.")
    
    feature_vis_config = SaeVisConfig(
        hook_point=hook_point,
        features=feature_indices,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device_str
    )
    
    print(">>> Running SaeVisRunner to collect feature activations and examples...")
    print(f">>> Processing {n_samples} samples...")
    
    sae_vis_data = SaeVisRunner(feature_vis_config).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:n_samples]["tokens"]
    )
    
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
    
    # Step 5: Extract examples and write JSONL
    print(">>> Extracting examples for each feature...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as fout:
        for feat_id in feature_indices:
            examples = extract_examples_from_sae_vis_data(
                sae_vis_data, 
                feat_id, 
                max_examples=max_examples_per_feature,
                tokenizer=model.tokenizer
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

