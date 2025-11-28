#!/usr/bin/env python3
"""
Extract top activating examples for finance features.
Reuses the exact pipeline from SAE-Reasoning compute_dashboard.py but outputs JSONL instead of HTML.
"""

import os
import json
import fire
from pathlib import Path

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from sae_lens import SAE
from sae_dashboard.sae_vis_data import SaeVisConfig
from sae_dashboard.sae_vis_runner import SaeVisRunner


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
    Extract examples for finance features - same pipeline as compute_dashboard.py but outputs JSONL.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    # STEP 1: Load SAE and LLM (same as compute_dashboard.py)
    print(">>> Loading SAE and LLM")
    is_local_path = os.path.exists(sae_path) and os.path.isdir(sae_path)
    
    if sae_id is None:
        sae = SAE.load_from_pretrained(sae_path, device=device)
    elif is_local_path:
        # Handle local SAE loading (from compute_score.py pattern)
        import re
        from safetensors import safe_open
        
        layer_match = re.match(r"blocks\.(\d+)\.", sae_id)
        if not layer_match:
            raise ValueError(f"Invalid sae_id format: {sae_id}")
        
        layer_num = int(layer_match.group(1))
        layer_path = os.path.join(sae_path, f"layers.{layer_num}")
        
        # Load config
        with open(os.path.join(layer_path, "cfg.json"), 'r') as f:
            cfg_dict = json.load(f)
        
        # Map fields
        if "num_latents" in cfg_dict and "d_sae" not in cfg_dict:
            cfg_dict["d_sae"] = cfg_dict["num_latents"]
        if "d_in" not in cfg_dict and "d_model" in cfg_dict:
            cfg_dict["d_in"] = cfg_dict["d_model"]
        
        # Add defaults
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
            "prepend_bos": cfg_dict.get("prepend_bos", False),
            "device": device,
            "sae_lens_training_version": None
        }
        for key, value in defaults.items():
            if key not in cfg_dict:
                cfg_dict[key] = value
        
        if cfg_dict.get("activation_fn_str") == "topk" and "activation_fn_kwargs" not in cfg_dict:
            cfg_dict["activation_fn_kwargs"] = {"k": cfg_dict.get("k", 16)}
        
        # Create SAE config
        from sae_lens import SAEConfig, TopKSAE, StandardSAE
        sae_cfg = SAEConfig.from_dict(cfg_dict)
        cfg_type_name = type(sae_cfg).__name__
        if 'TopK' in cfg_type_name:
            sae = TopKSAE(sae_cfg).to(device)
        else:
            sae = StandardSAE(sae_cfg).to(device)
        
        # Load weights
        weights_file = os.path.join(layer_path, "sae.safetensors")
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
    else:
        sae, _, _ = SAE.from_pretrained(sae_path, sae_id, device=device)

    # Load model - handle models that require trust_remote_code
    models_requiring_trust_remote_code = ["nemotron", "nvidia"]
    needs_trust_remote_code = any(
        keyword in model_path.lower() for keyword in models_requiring_trust_remote_code
    )
    
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
    # make pad token different from `bos` and `eos` to prevent removing `bos`/`eos` token during slicing
    if model.tokenizer.pad_token_id == model.tokenizer.eos_token_id:
        model.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # STEP 2: Load dataset (same as compute_dashboard.py)
    print(">>> Loading dataset")
    dataset = load_dataset(dataset_path, streaming=False, split="train")
    if column_name == "tokens":
        token_dataset = dataset
    else:
        print(">>> Tokenize dataset")
        # Get context_size from config or use default
        context_size = getattr(sae.cfg, 'context_size', None) or getattr(sae.cfg, 'max_length', 1024)
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

    # STEP 3: Load feature list (from JSON instead of CSV)
    print(">>> Loading feature list...")
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
    
    feature_indices = feature_data['feature_indices']
    feature_scores = feature_data['scores']
    feature_score_map = {feat_id: score for feat_id, score in zip(feature_indices, feature_scores)}
    
    print(f">>> Found {len(feature_indices)} features to process")

    # STEP 4: Build SaeVisConfig (same as compute_dashboard.py)
    hook_point = getattr(sae.cfg, 'hook_name', sae_id) if sae_id else getattr(sae.cfg, 'hook_name', None)
    if hook_point is None:
        raise ValueError("Could not determine hook_point. Please provide sae_id parameter.")
    
    feature_vis_config = SaeVisConfig(
        hook_point=hook_point,
        features=feature_indices,
        minibatch_size_features=minibatch_size_features,
        minibatch_size_tokens=minibatch_size_tokens,
        verbose=True,
        device=device
    )

    # STEP 5: Run SaeVisRunner (same as compute_dashboard.py)
    print(">>> Running SaeVisRunner to collect feature activations and examples...")
    print(f">>> Processing {n_samples} samples...")
    
    visualization_data = SaeVisRunner(feature_vis_config).run(
        encoder=sae,
        model=model,
        tokens=token_dataset[:n_samples]["tokens"]
    )

    # STEP 6: Extract examples from SaeVisData (instead of saving HTML)
    print(">>> Extracting examples from SaeVisData...")
    
    def extract_examples_for_feature(feature_data, max_examples=20):
        """Extract examples from FeatureData.sequence_data.seq_group_data with actual text"""
        examples = []
        
        if not hasattr(feature_data, 'sequence_data'):
            return examples
        
        sequence_data = feature_data.sequence_data
        if not hasattr(sequence_data, 'seq_group_data') or sequence_data.seq_group_data is None:
            return examples
        
        seq_group_data = sequence_data.seq_group_data
        all_activation_data = []
        
        # seq_group_data is a list of SequenceGroupData objects
        if isinstance(seq_group_data, list):
            for group_data in seq_group_data:
                if hasattr(group_data, 'seq_data'):
                    # SequenceGroupData contains a list of SequenceData
                    for seq_data in group_data.seq_data:
                        if hasattr(seq_data, 'feat_acts'):
                            feat_acts = seq_data.feat_acts
                            original_index = seq_data.original_index
                            qualifying_token_index = seq_data.qualifying_token_index
                            token_ids = getattr(seq_data, 'token_ids', None)
                            
                            # Convert to lists
                            if isinstance(feat_acts, torch.Tensor):
                                feat_acts = feat_acts.cpu().tolist()
                            if isinstance(original_index, torch.Tensor):
                                original_index = original_index.cpu().tolist()
                            if isinstance(qualifying_token_index, torch.Tensor):
                                qualifying_token_index = qualifying_token_index.cpu().tolist()
                            
                            # Handle single values vs lists
                            if not isinstance(feat_acts, list):
                                feat_acts = [feat_acts]
                            if not isinstance(original_index, list):
                                original_index = [original_index]
                            if not isinstance(qualifying_token_index, list):
                                qualifying_token_index = [qualifying_token_index]
                            
                            # Collect activation data with token_ids
                            min_len = min(len(feat_acts), len(original_index), len(qualifying_token_index))
                            for i in range(min_len):
                                act_val = float(feat_acts[i])
                                seq_idx = int(original_index[i])
                                token_idx = int(qualifying_token_index[i])
                                
                                # Get token_ids for this example
                                example_token_ids = None
                                if token_ids is not None:
                                    if isinstance(token_ids, torch.Tensor):
                                        if len(token_ids.shape) > 1:
                                            example_token_ids = token_ids[i].cpu().tolist() if i < len(token_ids) else None
                                        else:
                                            example_token_ids = token_ids.cpu().tolist()
                                    elif isinstance(token_ids, list):
                                        example_token_ids = token_ids[i] if i < len(token_ids) else None
                                
                                all_activation_data.append((act_val, seq_idx, token_idx, example_token_ids))
        
        # Sort by activation descending
        all_activation_data.sort(key=lambda x: x[0], reverse=True)
        
        # Extract top examples with actual text
        for act_val, seq_idx, token_idx, example_token_ids in all_activation_data[:max_examples]:
            # Try to get actual text from token_dataset
            text = f"Sequence {seq_idx}, token {token_idx}"  # fallback
            
            # First try: get from token_dataset if available
            if seq_idx < len(token_dataset):
                try:
                    seq_tokens = token_dataset[seq_idx]["tokens"]
                    if isinstance(seq_tokens, torch.Tensor):
                        seq_tokens = seq_tokens.cpu().tolist()
                    
                    # Get context around the token (e.g., ±10 tokens)
                    context_window = 10
                    start_idx = max(0, token_idx - context_window)
                    end_idx = min(len(seq_tokens), token_idx + context_window + 1)
                    context_tokens = seq_tokens[start_idx:end_idx]
                    
                    # Decode with tokenizer
                    text = model.tokenizer.decode(context_tokens, skip_special_tokens=True)
                    # Highlight the activating token position
                    if len(text) > 200:
                        # Truncate but keep the activating token visible
                        text = text[:200] + "..."
                except Exception as e:
                    pass
            
            # Second try: use example_token_ids if available
            if text.startswith("Sequence") and example_token_ids is not None:
                try:
                    if isinstance(example_token_ids, list):
                        # Decode the token(s)
                        text = model.tokenizer.decode(example_token_ids, skip_special_tokens=True)
                        if len(text) > 200:
                            text = text[:200] + "..."
                except Exception:
                    pass
            
            examples.append({
                'activation': act_val,
                'sequence_index': seq_idx,
                'token_position': token_idx,
                'text': text
            })
        
        return examples

    # STEP 7: Write JSONL output
    print(f">>> Writing examples to {output_path}...")
    with open(output_path, 'w') as fout:
        for feat_id in feature_indices:
            if feat_id not in visualization_data.feature_data_dict:
                continue
            
            feature_data = visualization_data.feature_data_dict[feat_id]
            examples = extract_examples_for_feature(feature_data, max_examples=max_examples_per_feature)
            
            record = {
                "feature_id": feat_id,
                "finance_score": float(feature_score_map[feat_id]),
                "examples": examples
            }
            
            fout.write(json.dumps(record) + "\n")
            print(f"    Feature {feat_id}: extracted {len(examples)} examples")
    
    print(f">>> Done! Examples saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(compute_examples_finance)

