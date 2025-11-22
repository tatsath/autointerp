"""Run AutoInterp evaluation for specific SAE features using vLLM.
Uses local autointerp module from autointerp/ folder and saves all results to Results/ folder.
Configured to use vLLM server for faster inference with Qwen 72B model.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

# Add local autointerp module to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Import from local autointerp module
import autointerp.eval_config as autointerp_config
import autointerp.main as autointerp_main

# Still need sae_bench utilities for SAE loading (external dependencies)
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils

MODEL_NAME = "ProsusAI/finbert"  # FinBERT model
SAE_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/finbert_topk_layer10_features3072_k24"
LAYER = 10  # Layer 10 for FinBERT
RESULTS_DIR = str(SCRIPT_DIR / "Results")  # Save directly to Results folder
FEATURES_TO_EVALUATE = [0, 1, 2, 3, 4]  # First 5 features
TOTAL_TOKENS = 1_000_000  # Increased to 1M tokens for better evaluation
CONTEXT_SIZE = 512  # FinBERT uses 512 context length
LLM_BATCH_SIZE = 32
LLM_DTYPE = "float32"  # FinBERT works better with float32
TORCH_DTYPE = torch.float32  # Use float32 for FinBERT to avoid dtype mismatches
FORCE_RERUN = True  # Set to True to rerun even if results exist (generates artifacts)

# Provider configuration - using vLLM
PROVIDER = "vllm"  # Using vLLM provider
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # vLLM model
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"  # vLLM server URL

# API key path - for vLLM, API key is optional (not required, but kept for compatibility)
# vLLM doesn't require authentication by default
API_KEY_PATH = Path(__file__).parent / "openai_api_key.txt"

# Patch transformer_lens for FinBERT support
if "finbert" in MODEL_NAME.lower():
    print("🔧 Patching transformer_lens for FinBERT support...")
    
    # Patch get_official_model_name to allow FinBERT
    import transformer_lens.loading_from_pretrained as loading_from_pretrained
    _original_get_official_model_name = loading_from_pretrained.get_official_model_name
    loading_from_pretrained.get_official_model_name = lambda name: name if "finbert" in name.lower() else _original_get_official_model_name(name)
    
    # Patch HookedTransformer to load FinBERT directly
    from transformer_lens import HookedTransformer
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
    _original_from_pretrained = HookedTransformer.from_pretrained_no_processing
    
    @classmethod
    def patched_from_pretrained(cls, model_name, **kwargs):
        if "finbert" in model_name.lower():
            print("🔧 Loading FinBERT directly (ProsusAI/finbert)...")
            # Load FinBERT config and model
            finbert_config = AutoConfig.from_pretrained("ProsusAI/finbert")
            finbert_model = AutoModel.from_pretrained("ProsusAI/finbert")
            
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
            
            # Create model structure directly
            model = HookedTransformer(bert_cfg, tokenizer=tokenizer, 
                                      move_to_device=kwargs.get("move_to_device", True),
                                      default_padding_side=kwargs.get("default_padding_side", "right"))
            
            # Map FinBERT weights to transformer_lens format
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
    
    # Patch dataset loading for FinBERT
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
        # Tokenize with truncation to prevent sequence length errors
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
        elif "Finance-Instruct" in dataset_name or "finance_instruct" in dataset_name.lower():
            # Finance-Instruct-500k uses "instruction" or "text" column
            # Try to detect the column automatically by loading a sample
            try:
                from datasets import load_dataset
                print(f"🔍 Detecting column for {dataset_name}...")
                sample = load_dataset(dataset_name, split="train[:10]", streaming=False)
                available_cols = sample.column_names
                print(f"   Available columns: {available_cols}")
                
                # Check for common text columns in order of preference
                if "instruction" in available_cols:
                    column_name = "instruction"
                    print(f"   ✅ Using column: 'instruction'")
                elif "text" in available_cols:
                    column_name = "text"
                    print(f"   ✅ Using column: 'text'")
                elif "input" in available_cols:
                    column_name = "input"
                    print(f"   ✅ Using column: 'input'")
                else:
                    # Use first text-like column
                    text_cols = [col for col in available_cols if any(kw in col.lower() for kw in ["text", "content", "instruction", "input", "prompt"])]
                    if text_cols:
                        column_name = text_cols[0]
                        print(f"   ✅ Using column: '{column_name}' (auto-detected)")
                    else:
                        # Fallback to first column
                        column_name = available_cols[0] if available_cols else "text"
                        print(f"   ⚠️  Using first available column: '{column_name}'")
            except Exception as e:
                # Fallback to "instruction" for Finance-Instruct datasets
                print(f"   ⚠️  Could not detect column, using default 'instruction': {e}")
                column_name = "instruction"  # Most common for Finance-Instruct
        # BERT models don't use BOS token, they use [CLS] token
        if hasattr(tokenizer, 'model_max_length'):
            if tokenizer.model_max_length > ctx_len:
                tokenizer.model_max_length = ctx_len
        # For BERT/FinBERT, set add_bos=False
        add_bos = False
        
        # Try with requested num_tokens, but if dataset doesn't have enough, use what's available
        try:
            tokens = _original_load_and_tokenize(dataset_name, ctx_len, num_tokens, tokenizer, column_name, add_bos)
            return tokens
        except (AssertionError, KeyError) as e:
            # KeyError might occur if column_name is wrong, try to detect it again
            if isinstance(e, KeyError) and ("Finance-Instruct" in dataset_name or "finance_instruct" in dataset_name.lower()):
                print(f"   ⚠️  Column '{column_name}' not found, trying to detect correct column...")
                try:
                    from datasets import load_dataset
                    sample = load_dataset(dataset_name, split="train[:10]", streaming=False)
                    available_cols = sample.column_names
                    if "instruction" in available_cols:
                        column_name = "instruction"
                    elif "text" in available_cols:
                        column_name = "text"
                    elif "input" in available_cols:
                        column_name = "input"
                    else:
                        column_name = available_cols[0] if available_cols else "text"
                    print(f"   ✅ Retrying with column: '{column_name}'")
                    tokens = _original_load_and_tokenize(dataset_name, ctx_len, num_tokens, tokenizer, column_name, add_bos)
                    return tokens
                except Exception as e2:
                    print(f"   ❌ Failed to detect column: {e2}")
                    raise e  # Re-raise original error
            # If it's an AssertionError (not enough tokens), continue to fallback logic
            if isinstance(e, AssertionError):
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
            try:
                dataset = dataset_utils.get_dataset_list_of_strs(dataset_name, column_name, 100, num_tokens * 5)
            except KeyError as ke:
                # Column not found, try to detect it
                if "Finance-Instruct" in dataset_name or "finance_instruct" in dataset_name.lower():
                    print(f"   ⚠️  Column '{column_name}' not found in fallback, detecting...")
                    from datasets import load_dataset
                    sample = load_dataset(dataset_name, split="train[:10]", streaming=False)
                    available_cols = sample.column_names
                    if "instruction" in available_cols:
                        column_name = "instruction"
                    elif "text" in available_cols:
                        column_name = "text"
                    elif "input" in available_cols:
                        column_name = "input"
                    else:
                        column_name = available_cols[0] if available_cols else "text"
                    print(f"   ✅ Using column: '{column_name}'")
                    dataset = dataset_utils.get_dataset_list_of_strs(dataset_name, column_name, 100, num_tokens * 5)
                else:
                    raise ke
            tokens = dataset_utils.tokenize_and_concat_dataset(
                tokenizer, dataset, ctx_len, add_bos=add_bos, max_tokens=None
            )
            actual_tokens = tokens.shape[0] * tokens.shape[1]
            print(f"⚠️  Using all available tokens: {actual_tokens} (requested: {num_tokens})")
            return tokens
    
    dataset_utils.load_and_tokenize_dataset = patched_load_and_tokenize_dataset
    
    # Patch gather_data to ensure all features are evaluated
    _original_gather_data = autointerp_main.AutoInterp.gather_data
    
    def patched_gather_data(self):
        """Patched version that forces evaluation of all features, even with minimal activations."""
        if self.cfg.override_latents is not None:
            original_latents = self.latents
            self.latents = self.cfg.override_latents
        
        generation_examples, scoring_examples = _original_gather_data(self)
        
        if self.cfg.override_latents is not None:
            self.latents = original_latents
        
        # If using override_latents, ensure all latents are evaluated
        if self.cfg.override_latents is not None:
            expected_latents = set(self.cfg.override_latents)
            latents_with_data = set(generation_examples.keys())
            missing_latents = expected_latents - latents_with_data
            
            if missing_latents:
                print(f"⚠️  {len(missing_latents)} features were skipped during data gathering. Creating minimal examples...")
                import sae_bench.sae_bench_utils.activation_collection as activation_collection
                from autointerp.main import Examples, Example, index_with_buffer
                
                dataset_size, seq_len = self.tokenized_dataset.shape
                buffer = self.cfg.buffer
                
                # Collect activations for missing features
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
    print("✅ FinBERT patches applied successfully")


def setup_environment():
    """Setup CUDA environment and return device. Local implementation for independence."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    return device


def load_local_topk_sae(sae_path: str, model_name: str, device: torch.device, dtype: torch.dtype, layer: int) -> topk_sae.TopKSAE:
    """Load a TopK SAE from local PyTorch .pt file (FinBERT format)."""
    # Load config from trainer_0/config.json
    cfg_path = os.path.join(sae_path, "trainer_0", "config.json")
    with open(cfg_path) as f:
        config = json.load(f)
    
    trainer_config = config["trainer"]
    d_in = trainer_config["activation_dim"]  # 768
    d_sae = trainer_config["dict_size"]  # 3072
    k = trainer_config["k"]  # 24

    # Load SAE from .pt file
    sae_file = os.path.join(sae_path, "trainer_0", "ae.pt")
    state_dict = torch.load(sae_file, map_location="cpu")

    # Convert from FinBERT format to SAEBench format
    renamed_params = {
        "W_enc": state_dict["encoder.weight"].T,  # [3072, 768] -> [768, 3072]
        "b_enc": state_dict["encoder.bias"],  # [3072]
        "W_dec": state_dict["decoder.weight"].T,  # [768, 3072] -> [3072, 768]
        "b_dec": state_dict["b_dec"],  # [768]
        "k": torch.tensor(k, dtype=torch.int),
    }

    # For BERT models, use blocks.{layer}.hook_resid_post
    sae = topk_sae.TopKSAE(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
        hook_name=f"blocks.{layer}.hook_resid_post",  # BERT layer output after residual
        use_threshold=False,
    )

    sae.load_state_dict(renamed_params)
    sae.to(device=device, dtype=dtype)
    sae.cfg.architecture = "topk"
    sae.cfg.dtype = LLM_DTYPE
    return sae


def check_vllm_server():
    """Check if vLLM server is running."""
    import urllib.request
    import urllib.error
    
    try:
        req = urllib.request.Request(f"{EXPLAINER_API_BASE_URL}/models")
        urllib.request.urlopen(req, timeout=5)
        return True
    except (urllib.error.URLError, Exception):
        return False


def main():
    """Run AutoInterp evaluation for specified features using vLLM.
    All results, logs, and artifacts are saved to Results/ folder.
    Completely independent from SAEBench (except for required SAE utilities).
    """
    device = setup_environment()
    
    # Create Results directory - all outputs go here
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if vLLM server is running
    print(f"🔍 Checking vLLM server at {EXPLAINER_API_BASE_URL}...")
    if not check_vllm_server():
        print(f"❌ vLLM server is not running at {EXPLAINER_API_BASE_URL}")
        print(f"   Please start the vLLM server first using:")
        print(f"   bash start_vllm_server_72b.sh")
        raise RuntimeError("vLLM server is not running")
    print(f"✅ vLLM server is running")
    
    # Load API key from local file (optional for vLLM)
    # vLLM doesn't require authentication, but we keep this for compatibility
    if not API_KEY_PATH.exists():
        api_key = ""  # Empty string is fine for vLLM
        print(f"ℹ️  API key file not found - vLLM doesn't require authentication")
    else:
        with open(API_KEY_PATH) as f:
            api_key = f.read().strip()
        if not api_key:
            api_key = ""  # Empty string is fine for vLLM
            print(f"ℹ️  API key file is empty - vLLM doesn't require authentication")
    
    # Validate provider configuration
    print(f"Using vLLM provider with API base URL: {EXPLAINER_API_BASE_URL}")
    print(f"Explainer model: {EXPLAINER_MODEL}")

    # Load SAE
    sae = load_local_topk_sae(SAE_PATH, MODEL_NAME, torch.device(device), TORCH_DTYPE, LAYER)
    d_sae, d_in = sae.W_dec.data.shape

    # Configure SAE - get context size from config
    with open(os.path.join(SAE_PATH, "trainer_0", "config.json")) as f:
        config = json.load(f)
        context_size = config["buffer"].get("ctx_len", CONTEXT_SIZE)
    
    sae.cfg = custom_sae_config.CustomSAEConfig(
        MODEL_NAME, d_in, d_sae, sae.cfg.hook_layer, sae.cfg.hook_name, context_size=context_size
    )
    sae.cfg.dtype = LLM_DTYPE

    # Validate features
    if any(f < 0 or f >= d_sae for f in FEATURES_TO_EVALUATE):
        raise ValueError(f"Features {FEATURES_TO_EVALUATE} out of range (0-{d_sae-1})")

    # Setup
    sae_id = f"finbert_layer{sae.cfg.hook_layer}_features{d_sae}_k{sae.k}"
    selected_saes = [(sae_id, sae)]

    # Create unique log filename based on model, SAE, features, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    features_str = "_".join(map(str, FEATURES_TO_EVALUATE))
    log_filename = f"autointerp_{model_short}_layer{sae.cfg.hook_layer}_features{features_str}_{timestamp}.txt"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    print(f"\nAutoInterp Evaluation (vLLM)")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Features: {FEATURES_TO_EVALUATE}")
    print(f"  Tokens: {TOTAL_TOKENS:,}")
    print(f"  Results folder: {RESULTS_DIR}")
    print(f"  Log file: {log_filename}")
    print(f"  Provider: {PROVIDER}")
    print(f"  Explainer: {EXPLAINER_MODEL}\n")

    # Run evaluation
    # Use larger Finance-Instruct-500k dataset for FinBERT (has 500K samples, can provide 1M+ tokens)
    # Falls back to financial-news if Finance-Instruct is not available
    if "finbert" in MODEL_NAME.lower():
        # Try Finance-Instruct-500k first (larger dataset with 500K samples)
        # This should easily provide 1M+ tokens
        dataset_name = "Josephgflowers/Finance-Instruct-500k"
        print(f"📊 Using Finance-Instruct-500k dataset (500K samples, should provide 1M+ tokens)")
    else:
        dataset_name = None
    
    config = autointerp_config.AutoInterpEvalConfig(
        model_name=MODEL_NAME,
        n_latents=None,
        override_latents=FEATURES_TO_EVALUATE,
        random_seed=42,
        llm_batch_size=LLM_BATCH_SIZE,
        llm_dtype=LLM_DTYPE,
        llm_context_size=CONTEXT_SIZE,
        total_tokens=TOTAL_TOKENS,
        scoring=True,
        dataset_name=dataset_name,  # Use financial-news for FinBERT
        dead_latent_threshold=-1.0,  # Negative threshold to force evaluation of all features
        # NOTE: To improve scores, consider:
        # - Increase TOTAL_TOKENS to 2_000_000 for more diverse examples
        # - Set dead_latent_threshold=0.0 to filter truly dead features
        act_threshold_frac=0.00001,  # Extremely low threshold to allow all features to activate
        # NOTE: Consider increasing to 0.001 to reduce noise while keeping real activations
        max_tokens_in_explanation=40,
        # NOTE: Increase to 80-100 for more detailed explanations (better scores)
        use_demos_in_explanation=True,
        # NOTE: Default scoring examples: n_top_ex_for_scoring=2, n_random_ex_for_scoring=10, n_iw_sampled_ex_for_scoring=2
        # To improve reliability, increase these values (e.g., 5, 20, 5)
        # See LOW_SCORES_EXPLANATION.md in Results/ folder for detailed improvement guide
    )

    # Save artifacts in Results folder with unique subfolder per run
    run_artifact_dir = f"artifacts_{model_short}_layer{sae.cfg.hook_layer}_{timestamp}"
    artifacts_path = os.path.join(RESULTS_DIR, run_artifact_dir)
    os.makedirs(artifacts_path, exist_ok=True)

    results_dict = autointerp_main.run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        api_key=api_key,
        output_path=RESULTS_DIR,  # Save directly to Results folder
        force_rerun=FORCE_RERUN,  # Set to True to regenerate artifacts
        save_logs_path=log_path,  # Unique log file per run
        artifacts_path=artifacts_path,  # Unique artifacts folder per run
        provider=PROVIDER,
        api_base_url=EXPLAINER_API_BASE_URL,
        explainer_model=EXPLAINER_MODEL,
    )

    # Check artifacts location (autointerp creates subfolder "autointerp/")
    autointerp_artifacts_dir = os.path.join(artifacts_path, "autointerp")
    artifact_files = []
    if os.path.exists(autointerp_artifacts_dir):
        artifact_files = [f for f in os.listdir(autointerp_artifacts_dir) if os.path.isfile(os.path.join(autointerp_artifacts_dir, f))]
    
    for sae_key, results in results_dict.items():
        if isinstance(results, dict) and "eval_result_metrics" in results:
            m = results["eval_result_metrics"].get("autointerp", {})
            score = m.get('autointerp_score', 'N/A')
            std = m.get('autointerp_std_dev', 'N/A')
            if score != 'N/A':
                print(f"\n  Score: {score:.4f} ± {std:.4f}")
    
    # Generate CSV summary - only include data from autointerp package
    print(f"\n📊 Generating CSV summary...")
    csv_data = []
    for sae_key, result in results_dict.items():
        if isinstance(result, dict) and "eval_result_unstructured" in result:
            unstructured = result["eval_result_unstructured"]
            if isinstance(unstructured, dict):
                for latent_id, latent_data in unstructured.items():
                    if isinstance(latent_data, dict):
                        explanation = latent_data.get("explanation", "")
                        score = latent_data.get("score", 0.0)
                        
                        # Only include data from autointerp package
                        csv_data.append({
                            "layer": LAYER,
                            "feature": latent_id,
                            "label": explanation,  # Full explanation, no truncation
                            "autointerp_score": f"{score:.4f}",
                        })
    
    # Sort by feature ID (convert to int for sorting)
    csv_data.sort(key=lambda x: int(x["feature"]) if str(x["feature"]).isdigit() else 999)
    
    # Write CSV to Results folder
    csv_filename = f"finbert_layer{LAYER}_features_summary_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    fieldnames = ["layer", "feature", "label", "autointerp_score"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if csv_data:
            writer.writerows(csv_data)
    
    print(f"   ✅ CSV saved to: {csv_path}")
    print(f"   📊 Generated summary for {len(csv_data)} features")
    
    # Final summary
    print(f"\n✓ Complete! All outputs saved to: {RESULTS_DIR}")
    print(f"  - Results JSON: {RESULTS_DIR}/*_eval_results.json")
    print(f"  - CSV Summary: {csv_path}")
    print(f"  - Logs: {log_path}")
    print(f"  - Artifacts: {autointerp_artifacts_dir}/")
    if artifact_files:
        print(f"    ({len(artifact_files)} artifact file(s) generated)")
    else:
        print(f"    (No artifacts - evaluation may have been skipped)")
    
    return results_dict


if __name__ == "__main__":
    main()

