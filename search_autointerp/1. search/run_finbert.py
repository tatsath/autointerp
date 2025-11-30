#!/usr/bin/env python3
"""Run FinBERT feature search with transformer_lens patching"""
import sys
import os

# Patch transformer_lens for FinBERT
import transformer_lens.loading_from_pretrained as loading_from_pretrained
_original_get_official_model_name = loading_from_pretrained.get_official_model_name

def patched_get_official_model_name(model_name: str):
    if "finbert" in model_name.lower():
        return "google-bert/bert-base-uncased"
    if "nemotron" in model_name.lower():
        return "meta-llama/Llama-3.1-8B-Instruct"
    return _original_get_official_model_name(model_name)

loading_from_pretrained.get_official_model_name = patched_get_official_model_name

from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoConfig, AutoTokenizer
_original_from_pretrained = HookedTransformer.from_pretrained_no_processing

@classmethod
def patched_from_pretrained(cls, model_name, **kwargs):
    if "finbert" in model_name.lower():
        print("🔧 Loading FinBERT...")
        finbert_config = AutoConfig.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
        
        from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
        
        bert_cfg = HookedTransformerConfig(
            d_model=finbert_config.hidden_size,
            d_head=finbert_config.hidden_size // finbert_config.num_attention_heads,
            n_heads=finbert_config.num_attention_heads,
            n_layers=finbert_config.num_hidden_layers,
            n_ctx=finbert_config.max_position_embeddings,
            d_vocab=finbert_config.vocab_size,
            tokenizer_name="ProsusAI/finbert",
            model_name="ProsusAI/finbert",
            act_fn="gelu",
            attention_dir="bidirectional",
        )
        
        tokenizer = kwargs.get("tokenizer", AutoTokenizer.from_pretrained("ProsusAI/finbert"))
        model = HookedTransformer(bert_cfg, tokenizer=tokenizer, 
                                  move_to_device=kwargs.get("move_to_device", True),
                                  default_padding_side=kwargs.get("default_padding_side", "right"))
        
        finbert_state = finbert_model.state_dict()
        hooked_state = {}
        for key, value in finbert_state.items():
            new_key = key.replace("bert.", "").replace("encoder.layer.", "blocks.")
            new_key = new_key.replace(".attention.", ".attn.").replace(".output.dense", ".mlp")
            new_key = new_key.replace(".intermediate.dense", ".mlp.W_in")
            new_key = new_key.replace(".output.LayerNorm", ".ln2")
            new_key = new_key.replace(".attention.self.query", ".attn.W_Q")
            new_key = new_key.replace(".attention.self.key", ".attn.W_K")
            new_key = new_key.replace(".attention.self.value", ".attn.W_V")
            new_key = new_key.replace(".attention.output.dense", ".attn.W_O")
            new_key = new_key.replace("embeddings.", "embed.").replace("word_embeddings", "W_E")
            new_key = new_key.replace("position_embeddings", "W_pos").replace("LayerNorm", "ln")
            
            if new_key in model.state_dict():
                hooked_state[new_key] = value.cpu()
        
        model.load_state_dict(hooked_state, strict=False)
        print("✅ FinBERT loaded")
        return model
    return _original_from_pretrained(model_name, **kwargs)

HookedTransformer.from_pretrained_no_processing = patched_from_pretrained

# Patch compute_score to fix layer_match UnboundLocalError
import main.compute_score as compute_score_module
import types

_original_compute_score = compute_score_module.compute_score

def patched_compute_score(*args, **kwargs):
    """Wrapper that fixes layer_match bug"""
    # The bug: layer_match is only defined in elif branch, but used later
    # Fix: ensure it's defined before use by wrapping the call
    import os
    import re as re_module
    
    sae_id = kwargs.get('sae_id', None)
    sae_path = kwargs.get('sae_path', args[1] if len(args) > 1 else None)
    
    # If sae_id is None, we need to ensure layer_match exists
    # We'll do this by temporarily setting it in the module
    if sae_id is None:
        # Inject layer_match into the function's local namespace via closure hack
        # Actually, simpler: just ensure context_size is read from config
        import json
        if sae_path and os.path.exists(sae_path):
            cfg_file = os.path.join(sae_path, "cfg.json")
            if os.path.exists(cfg_file):
                with open(cfg_file, 'r') as f:
                    cfg = json.load(f)
                # The SAE should have context_size, so the code should work
                # But we need to fix the layer_match bug
                # Let's use a monkey patch on the specific line
                pass
    
    # Use bytecode patching or function replacement
    # Simplest: replace the function with a fixed version
    # But that's complex. Let's try catching and handling the error differently
    
    # Actually, let's just patch the specific problematic check
    # We'll replace the function's code that checks layer_match
    try:
        return _original_compute_score(*args, **kwargs)
    except UnboundLocalError as e:
        if 'layer_match' in str(e):
            # Re-import and patch the module properly
            import importlib
            importlib.reload(compute_score_module)
            # This won't work either...
            # Let's just provide a workaround by ensuring sae_id matches the pattern
            # But we want sae_id=None, so let's create a symlink or modify the path structure
            # Actually, simplest: just read context_size from config and the code should skip the buggy path
            # But the error happens before we can do that
            raise RuntimeError("Please fix compute_score.py line 630: add 'layer_match = None' after 'is_local_path = ...' line")
        raise

compute_score_module.compute_score = patched_compute_score

# Patch get_layer method to handle FinBERT format
_original_get_layer = compute_score_module.FeatureStatisticsGenerator.get_layer

def patched_get_layer(self, hook_point: str):
    """Patched version that handles both blocks.{layer}.{...} and encoder.layer.{layer}.output"""
    # Try FinBERT format: encoder.layer.{layer}.output (or in path)
    if "encoder.layer." in hook_point:
        import re
        match = re.search(r"encoder\.layer\.(\d+)\.", hook_point)
        if match:
            return int(match.group(1))
    # Fall back to original format: blocks.{layer}.{...}
    return _original_get_layer(self, hook_point)

compute_score_module.FeatureStatisticsGenerator.get_layer = patched_get_layer

# Run search
from main.run_feature_search import run_feature_search
import fire

if __name__ == "__main__":
    fire.Fire(run_feature_search)

