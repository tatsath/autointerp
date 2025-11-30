#!/usr/bin/env python3
"""Run FinBERT advanced labeling with transformer_lens patching"""
import sys
import os

# Patch transformer_lens for FinBERT (same as search)
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

# Run advanced labeling
from run_labeling_advanced import run_labeling_advanced
import fire

if __name__ == "__main__":
    fire.Fire(run_labeling_advanced)



