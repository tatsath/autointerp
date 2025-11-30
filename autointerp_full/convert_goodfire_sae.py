#!/usr/bin/env python3
"""
Convert Goodfire HuggingFace SAE to autointerp_full format.
Downloads the model and creates the directory structure: layers.19/cfg.json and sae.safetensors
"""

import json
import torch
from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

# Configuration
REPO_ID = "Goodfire/Llama-3.1-8B-Instruct-SAE-l19"
LAYER = 19
OUTPUT_DIR = Path("/home/nvidia/Documents/Hariom/autointerp/autointerp_full/goodfire_sae_converted")

def main():
    print("🔄 Converting Goodfire SAE to autointerp_full format...")
    print(f"   Repository: {REPO_ID}")
    print(f"   Layer: {LAYER}")
    print(f"   Output: {OUTPUT_DIR}\n")
    
    # Download model file
    print("📥 Downloading model from HuggingFace...")
    model_path = hf_hub_download(REPO_ID, "Llama-3.1-8B-Instruct-SAE-l19.pth", repo_type="model")
    print(f"   Downloaded to: {model_path}\n")
    
    # Load checkpoint
    print("📂 Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"   Checkpoint type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"   Checkpoint keys: {list(checkpoint.keys())[:20]}")
        
        # Try to infer structure
        # Goodfire SAEs typically have: encoder.weight, encoder.bias, decoder.weight, b_dec
        # Or might be wrapped in a state_dict
        
        # Check if it's a state_dict with 'model' or 'sae' key
        if 'model' in checkpoint:
            sae_state = checkpoint['model']
        elif 'sae' in checkpoint:
            sae_state = checkpoint['sae']
        elif 'state_dict' in checkpoint:
            sae_state = checkpoint['state_dict']
        else:
            sae_state = checkpoint
        
        # Extract weights - try different possible key names
        encoder_weight = None
        encoder_bias = None
        decoder_weight = None
        b_dec = None
        
        # Try standard names
        if 'encoder_linear.weight' in sae_state:
            encoder_weight = sae_state['encoder_linear.weight']
            encoder_bias = sae_state.get('encoder_linear.bias', torch.zeros(encoder_weight.shape[0]))
            decoder_weight = sae_state.get('decoder_linear.weight', None)
            b_dec = sae_state.get('decoder_linear.bias', None)
        elif 'encoder.weight' in sae_state:
            encoder_weight = sae_state['encoder.weight']
            encoder_bias = sae_state.get('encoder.bias', torch.zeros(encoder_weight.shape[0]))
            decoder_weight = sae_state.get('decoder.weight', None)
            b_dec = sae_state.get('b_dec', None)
        elif 'W_enc' in sae_state:
            encoder_weight = sae_state['W_enc'].T  # Transpose if needed
            encoder_bias = sae_state.get('b_enc', torch.zeros(encoder_weight.shape[0]))
            decoder_weight = sae_state.get('W_dec', None)
            b_dec = sae_state.get('b_dec', None)
        else:
            # Print all keys to help debug
            print(f"\n   Available keys in checkpoint:")
            for key in list(sae_state.keys())[:30]:
                val = sae_state[key]
                if isinstance(val, torch.Tensor):
                    print(f"     {key}: Tensor {val.shape}")
                else:
                    print(f"     {key}: {type(val)}")
            raise ValueError("Could not find expected weight keys in checkpoint")
        
        # Get dimensions
        if encoder_weight is not None:
            d_sae, d_in = encoder_weight.shape
            print(f"\n📊 Detected dimensions:")
            print(f"   d_in (activation_dim): {d_in}")
            print(f"   d_sae (dict_size): {d_sae}")
            
            # Try to get k from config or checkpoint
            k = sae_state.get('k', None)
            if k is None:
                # Try to infer from config or use default
                k = 32  # Common default, may need adjustment
                print(f"   k: {k} (default, may need adjustment)")
            else:
                if isinstance(k, torch.Tensor):
                    k = k.item()
                print(f"   k: {k}")
        else:
            raise ValueError("Could not determine model dimensions")
        
        # Get base model config to verify d_in
        print("\n🔍 Verifying with base model config...")
        try:
            base_model_config = AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            hidden_size = base_model_config.hidden_size
            print(f"   Base model hidden_size: {hidden_size}")
            if d_in != hidden_size:
                print(f"   ⚠️  Warning: d_in ({d_in}) doesn't match hidden_size ({hidden_size})")
        except Exception as e:
            print(f"   Could not load base model config: {e}")
        
        # Create output directory
        layer_dir = OUTPUT_DIR / f"layers.{LAYER}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cfg.json
        cfg = {
            "d_in": int(d_in),
            "activation": "topk",
            "k": int(k),
            "num_latents": int(d_sae),
            "expansion_factor": float(d_sae / d_in),
            "normalize_decoder": True,
            "multi_topk": False,
            "skip_connection": False,
            "transcode": False,
            "hook_layer": LAYER,
            "hook_name": f"blocks.{LAYER}.hook_resid_post"
        }
        
        cfg_path = layer_dir / "cfg.json"
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        print(f"\n✅ Created config: {cfg_path}")
        
        # Prepare weights for safetensors
        # autointerp expects: encoder.weight [d_sae, d_in], encoder.bias [d_sae], W_dec [d_sae, d_in], b_dec [d_in]
        if decoder_weight is None:
            # If decoder not found, we might need to transpose encoder or use a different approach
            print("   ⚠️  Warning: decoder.weight not found, using encoder weight transpose")
            decoder_weight = encoder_weight.T
        
        # Ensure decoder is [d_sae, d_in] format
        if decoder_weight.shape == (d_in, d_sae):
            decoder_weight = decoder_weight.T
        elif decoder_weight.shape != (d_sae, d_in):
            raise ValueError(f"Unexpected decoder weight shape: {decoder_weight.shape}, expected ({d_sae}, {d_in}) or ({d_in}, {d_sae})")
        
        if b_dec is None:
            b_dec = torch.zeros(d_in)
        
        state_dict = {
            "encoder.weight": encoder_weight.contiguous().float(),
            "encoder.bias": encoder_bias.contiguous().float(),
            "W_dec": decoder_weight.contiguous().float(),
            "b_dec": b_dec.contiguous().float(),
        }
        
        # Save as safetensors
        sae_path = layer_dir / "sae.safetensors"
        save_file(state_dict, sae_path)
        print(f"✅ Created safetensors: {sae_path}")
        
        print(f"\n✅ Conversion complete!")
        print(f"   Output directory: {OUTPUT_DIR}")
        print(f"   Layer directory: {layer_dir}")
        print(f"   Use this path in your script: {OUTPUT_DIR}")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")

if __name__ == "__main__":
    main()

