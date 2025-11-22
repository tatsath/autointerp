#!/usr/bin/env python3
"""AutoInterp evaluation for FinBERT features from results.json with ACTION-ORIENTED prompts."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import csv

import torch
from safetensors.torch import load_file

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
AUTOINTERP_BASE = Path(__file__).parent.parent.parent.parent / "autointerp" / "autointerp_saeeval"
SAEBENCH_BASE = Path(__file__).parent.parent.parent.parent / "SAEBench"
RESULTS_JSON = Path(__file__).parent.parent.parent / "InterpUseCases_autointerp" / "FinbertSentiment" / "FeatureAlign" / "results.json"

if AUTOINTERP_BASE.exists():
    sys.path.insert(0, str(AUTOINTERP_BASE))
if SAEBENCH_BASE.exists():
    sys.path.insert(0, str(SAEBENCH_BASE))

# Patch transformer_lens for FinBERT
import transformer_lens.loading_from_pretrained as loading_from_pretrained
_original_get_official_model_name = loading_from_pretrained.get_official_model_name
loading_from_pretrained.get_official_model_name = lambda name: name if "finbert" in name.lower() else _original_get_official_model_name(name)

# Patch HookedTransformer to load FinBERT directly
from transformer_lens import HookedTransformer
from transformers import AutoModel, AutoConfig
_original_from_pretrained = HookedTransformer.from_pretrained_no_processing

@classmethod
def patched_from_pretrained(cls, model_name, **kwargs):
    if "finbert" in model_name.lower():
        print("🔧 Loading FinBERT directly (ProsusAI/finbert)...")
        # Load FinBERT config and model
        finbert_config = AutoConfig.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
        
        # Create model structure directly from FinBERT config without loading BERT-base weights
        from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
        from transformers import AutoTokenizer
        
        # Create config matching BERT-base structure (FinBERT uses same architecture)
        bert_cfg_dict = {
            "d_model": finbert_config.hidden_size,
            "d_head": finbert_config.hidden_size // finbert_config.num_attention_heads,
            "n_heads": finbert_config.num_attention_heads,
            "n_layers": finbert_config.num_hidden_layers,
            "n_ctx": finbert_config.max_position_embeddings,
            "d_vocab": finbert_config.vocab_size,
            "tokenizer_name": "ProsusAI/finbert",
            "model_name": "ProsusAI/finbert",
            "act_fn": "gelu",  # BERT uses GELU activation
            "attention_dir": "bidirectional",  # BERT uses bidirectional attention
        }
        bert_cfg = HookedTransformerConfig(**bert_cfg_dict)
        
        # Get tokenizer from kwargs or create new one
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
        # Create model structure directly (no BERT-base weights loaded)
        model = HookedTransformer(bert_cfg, tokenizer=tokenizer, 
                                  move_to_device=kwargs.get("move_to_device", True),
                                  default_padding_side=kwargs.get("default_padding_side", "right"))
        
        # Map FinBERT weights to transformer_lens format more carefully
        print("   Mapping FinBERT weights...")
        finbert_state = finbert_model.state_dict()
        hooked_state = {}
        
        # Better weight mapping: transformer_lens uses different naming
        for key, value in finbert_state.items():
            new_key = key
            # Remove 'bert.' prefix if present
            if new_key.startswith("bert."):
                new_key = new_key[5:]
            # Map encoder layers
            if "encoder.layer." in new_key:
                new_key = new_key.replace("encoder.layer.", "blocks.")
                # Map attention and MLP submodules
                new_key = new_key.replace(".attention.", ".attn.")
                new_key = new_key.replace(".attention.output", ".attn")
                new_key = new_key.replace(".output.dense", ".mlp")
                new_key = new_key.replace(".intermediate.dense", ".mlp.W_in")
                new_key = new_key.replace(".output.LayerNorm", ".ln2")
                new_key = new_key.replace(".attention.self.query", ".attn.W_Q")
                new_key = new_key.replace(".attention.self.key", ".attn.W_K")
                new_key = new_key.replace(".attention.self.value", ".attn.W_V")
                new_key = new_key.replace(".attention.output.dense", ".attn.W_O")
            # Map embeddings
            elif "embeddings." in new_key:
                new_key = new_key.replace("embeddings.", "embed.")
                new_key = new_key.replace("word_embeddings", "W_E")
                new_key = new_key.replace("position_embeddings", "W_pos")
                new_key = new_key.replace("token_type_embeddings", "W_token_type")
                new_key = new_key.replace("LayerNorm", "ln")
            
            if new_key in model.state_dict():
                hooked_state[new_key] = value.cpu()
        
        # Load mapped weights
        missing, unexpected = model.load_state_dict(hooked_state, strict=False)
        if missing:
            print(f"   ⚠️  {len(missing)} missing keys (using defaults)")
        if unexpected:
            print(f"   ⚠️  {len(unexpected)} unexpected keys ignored")
        
        print("   ✅ FinBERT weights loaded successfully")
        return model
    return _original_from_pretrained(model_name, **kwargs)

HookedTransformer.from_pretrained_no_processing = patched_from_pretrained

# Imports - try autointerp_saeeval (the package this script was designed for)
try:
    from autointerp import eval_config as autointerp_config
    from autointerp import main as autointerp_main
except ImportError:
    try:
        import autointerp_saeeval.autointerp.eval_config as autointerp_config
        import autointerp_saeeval.autointerp.main as autointerp_main
    except ImportError:
        if AUTOINTERP_BASE.exists():
            sys.path.insert(0, str(AUTOINTERP_BASE))
            from autointerp_saeeval.autointerp import eval_config as autointerp_config
            from autointerp_saeeval.autointerp import main as autointerp_main
        else:
            raise ImportError(f"Could not find autointerp_saeeval. Checked: {AUTOINTERP_BASE}")

# Patch AutoInterp to force evaluation of all features even with minimal activations
_original_gather_data = autointerp_main.AutoInterp.gather_data

def patched_gather_data(self):
    """Patched version that forces evaluation of all features, even with minimal activations."""
    # First, collect all activations for all features (not just alive ones)
    if self.cfg.override_latents is not None:
        # Temporarily set latents to all override_latents to collect their activations
        original_latents = self.latents
        self.latents = self.cfg.override_latents
    
    generation_examples, scoring_examples = _original_gather_data(self)
    
    # Restore original latents
    if self.cfg.override_latents is not None:
        self.latents = original_latents
    
    # If using override_latents, ensure all latents are evaluated
    if self.cfg.override_latents is not None:
        expected_latents = set(self.cfg.override_latents)
        latents_with_data = set(generation_examples.keys())
        missing_latents = expected_latents - latents_with_data
        
        if missing_latents:
            print(f"⚠️  {len(missing_latents)} features were skipped during data gathering. Creating minimal examples...")
            # For missing latents, create minimal examples using random sequences
            # This ensures they still get evaluated even if they don't activate
            import torch
            from autointerp.main import Examples, Example
            from autointerp.main import index_with_buffer
            
            dataset_size, seq_len = self.tokenized_dataset.shape
            buffer = self.cfg.buffer
            
            # Collect activations for missing features
            import sae_bench.sae_bench_utils.activation_collection as activation_collection
            acts_all = activation_collection.collect_sae_activations(
                self.tokenized_dataset,
                self.model,
                self.sae,
                self.cfg.llm_batch_size,
                self.sae.cfg.hook_layer,
                self.sae.cfg.hook_name,
                mask_bos_pad_eos_tokens=True,
                selected_latents=list(missing_latents),
                activation_dtype=torch.bfloat16,
            )
            
            # Create mapping from latent index to acts position
            latent_to_idx = {lat: i for i, lat in enumerate(sorted(missing_latents))}
            
            for latent_idx in missing_latents:
                # Create minimal examples using random sequences
                rand_indices = torch.stack([
                    torch.randint(0, dataset_size, (self.cfg.n_top_ex_for_generation,)),
                    torch.randint(buffer, seq_len - buffer, (self.cfg.n_top_ex_for_generation,))
                ], dim=-1)
                
                minimal_examples = []
                acts_idx = latent_to_idx[latent_idx]
                
                for idx_pair in rand_indices:
                    toks = index_with_buffer(self.tokenized_dataset, idx_pair.unsqueeze(0), buffer=buffer)[0]
                    # Get activations for this feature at this position
                    batch_idx, pos_idx = idx_pair[0].item(), idx_pair[1].item()
                    if batch_idx < acts_all.shape[0] and pos_idx < acts_all.shape[1]:
                        act_val = acts_all[batch_idx, pos_idx, acts_idx].item()
                    else:
                        act_val = 0.0
                    
                    # Create list of activations (all zeros except maybe one position)
                    acts_list = [0.0] * len(toks)
                    center_pos = buffer
                    if center_pos < len(acts_list):
                        acts_list[center_pos] = act_val
                    
                    minimal_examples.append(Example(
                        toks=toks.cpu().tolist(),
                        acts=acts_list,
                        act_threshold=0.0,  # Very low threshold
                        model=self.model
                    ))
                
                generation_examples[latent_idx] = Examples(minimal_examples)
                scoring_examples[latent_idx] = Examples(minimal_examples)
                print(f"   ✅ Added examples for feature {latent_idx}")
    
    return generation_examples, scoring_examples

autointerp_main.AutoInterp.gather_data = patched_gather_data

try:
    import sae_bench.custom_saes.custom_sae_config as custom_sae_config
    import sae_bench.custom_saes.topk_sae as topk_sae
    import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
except ImportError:
    raise ImportError(f"Could not import sae_bench modules. Checked path: {SAEBENCH_BASE}")

# Patch dataset_utils to use 'headline' column for financial-news dataset and fix tokenization
_original_load_and_tokenize = dataset_utils.load_and_tokenize_dataset
_original_tokenize_and_concat = dataset_utils.tokenize_and_concat_dataset

def patched_tokenize_and_concat_dataset(tokenizer, dataset, seq_len, add_bos=True, max_tokens=None):
    """Patched version that adds truncation for BERT models."""
    full_text = tokenizer.eos_token.join(dataset) if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token else "\n".join(dataset)
    
    # divide into chunks to speed up tokenization
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [
        full_text[i * chunk_length : (i + 1) * chunk_length] for i in range(num_chunks)
    ]
    # Tokenize with truncation to prevent sequence length errors, but use a larger max_length
    # to ensure we get enough tokens. We'll truncate later when batching.
    max_tokenizer_length = min(seq_len * 2, 512)  # Use 2x seq_len or 512, whichever is smaller
    tokens = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=max_tokenizer_length)["input_ids"].flatten()
    
    # remove pad token
    if tokenizer.pad_token_id is not None:
        tokens = tokens[tokens != tokenizer.pad_token_id]
    
    # Now truncate to max_tokens if specified
    if max_tokens is not None:
        tokens = tokens[: max_tokens + seq_len + 1]
    
    num_tokens = len(tokens)
    num_batches = num_tokens // seq_len
    
    # drop last batch if not full
    tokens = tokens[: num_batches * seq_len]
    import einops
    tokens = einops.rearrange(
        tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
    )
    
    if add_bos and hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
        tokens[:, 0] = tokenizer.bos_token_id
    return tokens

dataset_utils.tokenize_and_concat_dataset = patched_tokenize_and_concat_dataset

def patched_load_and_tokenize_dataset(dataset_name, ctx_len, num_tokens, tokenizer, column_name="text", add_bos=True):
    if "financial-news" in dataset_name:
        column_name = "headline"
    # BERT models don't use BOS token, they use [CLS] token
    # Also ensure tokenizer has proper truncation settings
    if hasattr(tokenizer, 'model_max_length'):
        if tokenizer.model_max_length > ctx_len:
            tokenizer.model_max_length = ctx_len
    # For BERT/FinBERT, set add_bos=False
    add_bos = False
    
    # Try with requested num_tokens, but if dataset doesn't have enough, use what's available
    try:
        tokens = _original_load_and_tokenize(dataset_name, ctx_len, num_tokens, tokenizer, column_name, add_bos)
        return tokens
    except AssertionError:
        # Dataset doesn't have enough tokens, try with progressively smaller amounts
        print(f"⚠️  Dataset doesn't have {num_tokens} tokens, trying with smaller amounts...")
        for reduced_tokens in [num_tokens // 2, num_tokens // 4, num_tokens // 10, 500_000, 250_000, 100_000, 50_000, 25_000, 10_000]:
            try:
                tokens = _original_load_and_tokenize(dataset_name, ctx_len, reduced_tokens, tokenizer, column_name, add_bos)
                print(f"✅ Using {reduced_tokens} tokens (available: {tokens.shape[0] * tokens.shape[1]})")
                return tokens
            except AssertionError:
                continue
        # If all fail, try to get whatever is available without assertion
        # Load dataset and get tokens without strict assertion
        dataset = dataset_utils.get_dataset_list_of_strs(dataset_name, column_name, 100, num_tokens * 5)
        tokens = dataset_utils.tokenize_and_concat_dataset(
            tokenizer, dataset, ctx_len, add_bos=add_bos, max_tokens=None
        )
        actual_tokens = tokens.shape[0] * tokens.shape[1]
        print(f"⚠️  Using all available tokens: {actual_tokens} (requested: {num_tokens})")
        return tokens
dataset_utils.load_and_tokenize_dataset = patched_load_and_tokenize_dataset

# Config
MODEL_NAME = "ProsusAI/finbert"
# Match the original script's path structure
# Original: SCRIPT_DIR.parent / "Autointerp_saebench" (from EndtoEnd/Autointerp/)
# Our script is in autointerp/autointerp_full_optimized_finbert/, so we need to go to InterpUseCases_autointerp/EndtoEnd/Autointerp_saebench
SAE_PATH = SCRIPT_DIR.parent.parent / "InterpUseCases_autointerp" / "EndtoEnd" / "Autointerp_saebench" / "finbert_sae_converted"
if not SAE_PATH.exists():
    SAE_PATH = SCRIPT_DIR.parent.parent / "InterpUseCases_autointerp" / "EndtoEnd" / "Autointerp" / "topk_sae_converted"  # Fallback
LAYER = 10
OUTPUT_DIR = SCRIPT_DIR / "autointerp_results"
API_KEY_FILE = SCRIPT_DIR.parent.parent / "InterpUseCases_autointerp" / "EndtoEnd" / "Autointerp_saebench" / "openai_api_key.txt"

# Load features from results.json - extract all unique feature IDs
print(f"📊 Loading feature IDs from: {RESULTS_JSON}")
if not RESULTS_JSON.exists():
    raise FileNotFoundError(f"Results JSON not found: {RESULTS_JSON}")

with open(RESULTS_JSON) as f:
    results_data = json.load(f)
    
# Extract all unique feature IDs from top_features in all entries
feature_ids = set()
for item in results_data:
    for feature in item.get("top_features", []):
        feature_id = feature.get("feature_id")
        if feature_id is not None:
            feature_ids.add(int(feature_id))

feature_ids = sorted(list(feature_ids))
print(f"📊 Found {len(feature_ids)} unique feature IDs: {feature_ids}")

# Load SAE
layer_dir = SAE_PATH / f"layers.{LAYER}"
if not layer_dir.exists():
    raise FileNotFoundError(f"SAE layer directory not found: {layer_dir}")

with open(layer_dir / "cfg.json") as f:
    cfg = json.load(f)

state_dict = load_file(str(layer_dir / "sae.safetensors"))
device = "cuda:0" if torch.cuda.is_available() else "cpu"
sae = topk_sae.TopKSAE(d_in=cfg["d_in"], d_sae=cfg["num_latents"], k=cfg["k"],
                      model_name=MODEL_NAME, hook_layer=LAYER, device=device,
                      dtype=torch.float32, hook_name=f"blocks.{LAYER}.hook_resid_post")
sae.load_state_dict({"W_enc": state_dict["encoder.weight"].T, "b_enc": state_dict["encoder.bias"],
                     "W_dec": state_dict["W_dec"], "b_dec": state_dict["b_dec"], "k": torch.tensor(cfg["k"])})
sae.cfg = custom_sae_config.CustomSAEConfig(MODEL_NAME, cfg["d_in"], cfg["num_latents"], LAYER,
                                           f"blocks.{LAYER}.hook_resid_post", context_size=512)

# Run evaluation
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if not API_KEY_FILE.exists():
    raise FileNotFoundError(f"API key file not found: {API_KEY_FILE}")

with open(API_KEY_FILE) as f:
    api_key = f.read().strip()

config = autointerp_config.AutoInterpEvalConfig(
    model_name=MODEL_NAME,
    n_latents=None,  # Required when using override_latents
    override_latents=feature_ids,
    dataset_name="ashraq/financial-news",
    total_tokens=1_000_000,  # Increased to 1M tokens to ensure all features activate
    llm_context_size=512,
    llm_batch_size=16,
    llm_dtype="float32",
    random_seed=42,
    n_top_ex_for_generation=15, 
    n_iw_sampled_ex_for_generation=10,
    n_top_ex_for_scoring=5,
    n_random_ex_for_scoring=15,
    n_iw_sampled_ex_for_scoring=5,
    dead_latent_threshold=-1.0,  # Negative threshold to force evaluation of all features (even with zero activations)
    act_threshold_frac=0.00001,  # Extremely low threshold (0.001% of max) to allow all features to activate
    max_tokens_in_explanation=40,
    use_demos_in_explanation=True,
    # Action-oriented parameters
    num_examples_per_explainer_prompt=1,  # Keep explanations focused
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
results = autointerp_main.run_eval(
    config, [(f"finbert_layer{LAYER}_all_features", sae)], device, api_key, str(OUTPUT_DIR),
    force_rerun=True, save_logs_path=str(OUTPUT_DIR / f"autointerp_finbert_all_features_{timestamp}.txt"),
    artifacts_path=str(OUTPUT_DIR / f"artifacts_{timestamp}")
)

# Generate CSV summary similar to topk_sae_results_summary_enhanced.csv
print(f"\n📊 Generating CSV summary...")
csv_data = []
for sae_key, result in results.items():
    if isinstance(result, dict) and "eval_result_unstructured" in result:
        unstructured = result["eval_result_unstructured"]
        if isinstance(unstructured, dict):
            for latent_id, latent_data in unstructured.items():
                if isinstance(latent_data, dict):
                    explanation = latent_data.get("explanation", "")
                    score = latent_data.get("score", 0.0)
                    predictions = latent_data.get("predictions", [])
                    correct_seqs = latent_data.get("correct seqs", [])
                    
                    # Calculate detection metrics (similar to enhanced CSV)
                    pred_set = set(predictions) if isinstance(predictions, list) else set()
                    correct_set = set(correct_seqs) if isinstance(correct_seqs, list) else set()
                    
                    tp = len(pred_set & correct_set)
                    fp = len(pred_set - correct_set)
                    fn = len(correct_set - pred_set)
                    tn = max(0, 14 - (tp + fp + fn))  # Assuming 14 total examples
                    
                    total = tp + fp + fn + tn
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    accuracy = (tp + tn) / total if total > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    csv_data.append({
                        "layer": LAYER,
                        "feature": latent_id,
                        "label": explanation[:80] if explanation else "",  # Truncate for readability
                        "f1_score": f"{f1:.4f}",
                        "autointerp_score": f"{score:.4f}",
                        "detection_f1": f"{f1:.4f}",
                        "detection_precision": f"{precision:.4f}",
                        "detection_recall": f"{recall:.4f}",
                        "detection_accuracy": f"{accuracy:.4f}",
                        "detection_tp": tp,
                        "detection_fp": fp,
                        "detection_fn": fn,
                        "detection_tn": tn
                    })

# Sort by feature ID (convert to int for sorting)
csv_data.sort(key=lambda x: int(x["feature"]) if str(x["feature"]).isdigit() else 999)

# Write CSV
csv_path = OUTPUT_DIR / f"finbert_all_features_summary_{timestamp}.csv"
fieldnames = ["layer", "feature", "label", "f1_score", "autointerp_score", 
              "detection_f1", "detection_precision", "detection_recall", "detection_accuracy",
              "detection_tp", "detection_fp", "detection_fn", "detection_tn"]
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    if csv_data:
        writer.writerows(csv_data)

print(f"   ✅ CSV saved to: {csv_path}")
print(f"\n✅ Complete! Results saved to: {OUTPUT_DIR}")
print(f"   - Log file: {OUTPUT_DIR / f'autointerp_finbert_all_features_{timestamp}.txt'}")
print(f"   - CSV summary: {csv_path}")
print(f"   - Artifacts: {OUTPUT_DIR / f'artifacts_{timestamp}'}")
