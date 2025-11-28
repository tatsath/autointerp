"""Run AutoInterp evaluation for top 100 Nemotron finance features.
Minimalistic version - all important parameters are at the top.
Results saved to autointerp_saeeval/Results folder.
"""

import csv
import json
import os
import sys
import re
from datetime import datetime
from pathlib import Path

import torch
import requests

# ============================================================================
# ⚠️ MOST IMPORTANT CONFIGURATION - Edit these values
# ============================================================================

# Model and SAE paths
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE_CHECKPOINT_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/nemotron_nano_layer28_features35840_k64/trainer_0/ae.pt"
SAE_CONFIG_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/nemotron_nano_layer28_features35840_k64/trainer_0/config.json"

# Evaluation parameters
LAYER = 28

# Feature selection - choose ONE option:
# Option 1: Pass feature IDs directly
FEATURES_TO_EVALUATE = None  # Set to None to extract from summary file (using TOP_K_FEATURES)
                              # Or set to list of feature IDs, e.g., [18529, 6105, 8982, ...]

# Option 2: Extract top K features from summary file (ranked by score)
# Only used if FEATURES_TO_EVALUATE is None
TOP_K_FEATURES = 100  # Number of top features to extract from summary file
FEATURES_SUMMARY_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_finance_features/top_finance_features_summary.txt"

TOTAL_TOKENS = 2_000_000  # ⚠️ Number of tokens sampled from dataset
DATASET_NAME = "ashraq/financial-news"  # ⚠️ CRITICAL: Must be financial dataset!

# LLM configuration
CONTEXT_SIZE = 128
LLM_BATCH_SIZE = 32
LLM_DTYPE = "bfloat16"

# vLLM server configuration
PROVIDER = "vllm"
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"

# Output directory (saves to autointerp_saeeval/Results)
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = str(SCRIPT_DIR / "Results")
API_KEY_PATH = SCRIPT_DIR / "openai_api_key.txt"
FORCE_RERUN = True

# ============================================================================
# Setup paths and imports
# ============================================================================

# Add paths for imports
AUTOINTERP_BASE = SCRIPT_DIR
SAEBENCH_BASE = Path(__file__).parent.parent.parent.parent / "SAEBench"

if AUTOINTERP_BASE.exists():
    sys.path.insert(0, str(AUTOINTERP_BASE))
if SAEBENCH_BASE.exists():
    sys.path.insert(0, str(SAEBENCH_BASE))

# Import modules
from autointerp import eval_config as autointerp_config
from autointerp import main as autointerp_main
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.topk_sae as topk_sae
import transformer_lens.loading_from_pretrained as loading_from_pretrained
from transformer_lens import HookedTransformer

# ============================================================================
# Nemotron-specific patches (handles 4480-dim mixer vs 4096-dim Llama)
# ============================================================================

def setup_nemotron_patches():
    """Setup patches for Nemotron architecture compatibility."""
    # Patch model name mapping (Nemotron -> Llama for transformer_lens)
    _original_get_official_model_name = loading_from_pretrained.get_official_model_name
    def patched_get_official_model_name(model_name: str):
        if "nemotron" in model_name.lower():
            return "meta-llama/Llama-3.1-8B-Instruct"
        return _original_get_official_model_name(model_name)
    loading_from_pretrained.get_official_model_name = patched_get_official_model_name
    
    # Patch HookedTransformer to allow trust_remote_code
    _original_from_pretrained = HookedTransformer.from_pretrained_no_processing
    @classmethod
    def patched_from_pretrained(cls, model_name, **kwargs):
        if "nemotron" in model_name.lower():
            kwargs.setdefault("trust_remote_code", True)
        return _original_from_pretrained(model_name, **kwargs)
    HookedTransformer.from_pretrained_no_processing = patched_from_pretrained
    
    # Patch activation collection for 4480-dim mixer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    
    _original_collect = activation_collection.collect_sae_activations
    _original_sparsity = activation_collection.get_feature_activation_sparsity
    
    # Global variables for Nemotron model (must be declared at module level)
    import sys
    if not hasattr(sys.modules[__name__], '_nemotron_model'):
        sys.modules[__name__]._nemotron_model = None
        sys.modules[__name__]._nemotron_tokenizer = None
        sys.modules[__name__]._nemotron_layer = None
    
    def patched_collect(tokens, model, sae, batch_size, layer, hook_name, 
                       mask_bos_pad_eos_tokens=False, selected_latents=None, activation_dtype=None):
        # Access module-level globals
        module = sys.modules[__name__]
        # Check if SAE expects 4480 dim (Nemotron mixer dimension)
        if hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480:
            if module._nemotron_model is None or module._nemotron_layer != layer:
                print(f"   Loading raw Nemotron model for layer {layer}...")
                # Clear CUDA cache before loading
                torch.cuda.empty_cache()
                # Use device_map="auto" to better manage memory, or "cpu" if GPU is full
                try:
                    module._nemotron_model = AutoModelForCausalLM.from_pretrained(
                        MODEL_NAME, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("   ⚠️ GPU OOM, trying CPU offloading...")
                        torch.cuda.empty_cache()
                        module._nemotron_model = AutoModelForCausalLM.from_pretrained(
                            MODEL_NAME, trust_remote_code=True, device_map="cpu", torch_dtype=torch.float16, low_cpu_mem_usage=True
                        )
                    else:
                        raise
                module._nemotron_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                module._nemotron_layer = layer
                module._nemotron_model.eval()
            
            sae_acts = []
            device = next(module._nemotron_model.parameters()).device
            from tqdm import tqdm
            for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Collecting mixer activations"):
                tokens_BL = tokens[i : i + batch_size].to(device)
                with torch.no_grad():
                    mixer_activations = []
                    def hook_fn(module_hook, input, output):
                        mixer_activations.append(output.detach())
                    handle = module._nemotron_model.backbone.layers[layer].mixer.register_forward_hook(hook_fn)
                    _ = module._nemotron_model(input_ids=tokens_BL)
                    handle.remove()
                    mixer_act = mixer_activations[0]  # [batch, seq_len, 4480]
                
                mixer_act = mixer_act.to(dtype=sae.dtype, device=sae.device)
                sae_act_BLF = sae.encode(mixer_act)
                if selected_latents is not None:
                    sae_act_BLF = sae_act_BLF[:, :, selected_latents]
                if mask_bos_pad_eos_tokens:
                    from sae_bench.sae_bench_utils.activation_collection import get_bos_pad_eos_mask
                    attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, module._nemotron_tokenizer)
                else:
                    attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)
                attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)
                sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]
                if activation_dtype is not None:
                    sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)
                sae_acts.append(sae_act_BLF.cpu())
            return torch.cat(sae_acts, dim=0)
        return _original_collect(tokens, model, sae, batch_size, layer, hook_name,
                                mask_bos_pad_eos_tokens, selected_latents, activation_dtype)
    
    def patched_sparsity(tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens=False):
        if hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480:
            sae_acts = patched_collect(tokens, model, sae, batch_size, layer, hook_name,
                                      mask_bos_pad_eos_tokens, selected_latents=None, activation_dtype=None)
            running_sum_F = (sae_acts > 0).to(dtype=torch.float32).sum(dim=(0, 1))
            total_tokens = sae_acts.shape[0] * sae_acts.shape[1]
            sparsity_F = running_sum_F / total_tokens if total_tokens > 0 else torch.zeros_like(running_sum_F)
            return sparsity_F.to(sae.device)
        return _original_sparsity(tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens)
    
    activation_collection.collect_sae_activations = patched_collect
    activation_collection.get_feature_activation_sparsity = patched_sparsity
    print("✅ Nemotron patches applied")

# Apply patches
if "nemotron" in MODEL_NAME.lower():
    setup_nemotron_patches()

# Patch dataset loading for financial-news dataset (uses 'headline' column, not 'text')
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
_original_load_and_tokenize = dataset_utils.load_and_tokenize_dataset

def patched_load_and_tokenize_dataset(dataset_name, ctx_len, num_tokens, tokenizer, column_name="text", add_bos=True):
    """Patched version that uses 'headline' column for financial-news dataset and handles token limits."""
    if "financial-news" in dataset_name:
        column_name = "headline"
        print(f"   Using 'headline' column for financial-news dataset")
    
    # Try with requested num_tokens, but if dataset doesn't have enough, use what's available
    try:
        return _original_load_and_tokenize(dataset_name, ctx_len, num_tokens, tokenizer, column_name, add_bos)
    except AssertionError:
        # Dataset might not have enough tokens in the format expected, try with progressively smaller amounts
        print(f"   ⚠️  Dataset loading issue with {num_tokens:,} tokens, trying with smaller amounts...")
        # Try 1M first (user requested), then progressively smaller
        for reduced_tokens in [1_000_000, num_tokens // 2, num_tokens // 4, 750_000, 500_000, 250_000, 100_000]:
            try:
                tokens = _original_load_and_tokenize(dataset_name, ctx_len, reduced_tokens, tokenizer, column_name, add_bos)
                actual_tokens = tokens.shape[0] * tokens.shape[1]
                print(f"   ✅ Using {reduced_tokens:,} tokens (actual: {actual_tokens:,})")
                return tokens
            except AssertionError:
                if reduced_tokens == 1_000_000:
                    print(f"   ⚠️  1M tokens also failed, trying smaller amounts...")
                continue
        # If all fail, try to bypass assertion by loading more data than needed
        print(f"   ⚠️  Trying to load 1M tokens with increased data multiplier...")
        # The issue is that get_dataset_list_of_strs uses num_tokens * 5 for total_chars
        # We need to request much more data to get enough tokens after tokenization
        # For headlines, estimate ~20-30 tokens per headline, so we need ~50K headlines for 1M tokens
        # At ~100 chars per headline, that's ~5M chars, but we'll use 30M to be safe
        try:
            # Manually load more data to ensure we get 1M tokens
            # Use a much larger multiplier to ensure we get enough tokens
            dataset_list = dataset_utils.get_dataset_list_of_strs(dataset_name, column_name, 100, 30_000_000)  # 30M chars for 1M tokens
            # Don't use max_tokens limit, let it load all available
            tokens = dataset_utils.tokenize_and_concat_dataset(
                tokenizer, dataset_list, ctx_len, add_bos=add_bos, max_tokens=None
            )
            actual_tokens = tokens.shape[0] * tokens.shape[1]
            # Truncate to 1M if we got more
            if actual_tokens > 1_000_000:
                # Reshape to truncate to exactly 1M tokens
                num_batches = 1_000_000 // ctx_len
                tokens = tokens[:num_batches, :]
                actual_tokens = tokens.shape[0] * tokens.shape[1]
            if actual_tokens >= 1_000_000:
                print(f"   ✅ Using 1,000,000 tokens (actual: {actual_tokens:,})")
                return tokens
            else:
                print(f"   ⚠️  Got {actual_tokens:,} tokens (less than 1M), trying 500K...")
        except Exception as e:
            print(f"   ⚠️  Direct load failed: {e}, trying fallback...")
        
        # Last resort: try with 500K
        try:
            tokens = _original_load_and_tokenize(dataset_name, ctx_len, 500_000, tokenizer, column_name, add_bos)
            actual_tokens = tokens.shape[0] * tokens.shape[1]
            print(f"   ✅ Using fallback: {actual_tokens:,} tokens")
            return tokens
        except AssertionError:
            # Final fallback: load whatever is available
            dataset_list = dataset_utils.get_dataset_list_of_strs(dataset_name, column_name, 100, 500_000 * 10)
            tokens = dataset_utils.tokenize_and_concat_dataset(
                tokenizer, dataset_list, ctx_len, add_bos=add_bos, max_tokens=None
            )
            actual_tokens = tokens.shape[0] * tokens.shape[1]
            print(f"   ✅ Using all available: {actual_tokens:,} tokens")
            return tokens

dataset_utils.load_and_tokenize_dataset = patched_load_and_tokenize_dataset

# ============================================================================
# Helper functions
# ============================================================================

def check_vllm_server():
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"{EXPLAINER_API_BASE_URL.rstrip('/v1')}/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ vLLM server is running at {EXPLAINER_API_BASE_URL}")
            return True
    except Exception as e:
        print(f"❌ vLLM server check failed: {e}")
        print(f"   Start it with: python -m vllm.entrypoints.openai.api_server --model {EXPLAINER_MODEL} --port 8002")
    return False

def extract_top_features(summary_path: str, top_k: int = 100):
    """Extract top K feature indices from summary file."""
    feature_indices = []
    pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
    with open(summary_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                feature_indices.append(int(match.group(1)))
                if len(feature_indices) >= top_k:
                    break
    if len(feature_indices) < top_k:
        print(f"⚠️  Warning: Only found {len(feature_indices)} features, expected {top_k}")
    return feature_indices

def load_nemotron_sae(sae_checkpoint_path: str, config_path: str, model_name: str, 
                     device: torch.device, dtype: torch.dtype, layer: int) -> topk_sae.TopKSAE:
    """Load Nemotron SAE from checkpoint and convert to SAEBench format."""
    print(f"\n📂 Loading SAE from: {sae_checkpoint_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    trainer_config = config["trainer"]
    dict_size = trainer_config["dict_size"]
    k = trainer_config["k"]
    activation_dim = trainer_config["activation_dim"]
    
    checkpoint = torch.load(sae_checkpoint_path, map_location="cpu")
    W_enc = checkpoint["encoder.weight"].T  # [d_sae, d_in] -> [d_in, d_sae]
    b_enc = checkpoint["encoder.bias"]
    W_dec = checkpoint["decoder.weight"].T  # [d_in, d_sae] -> [d_sae, d_in]
    b_dec = checkpoint.get("b_dec", torch.zeros(activation_dim))
    
    hook_name = f"blocks.{layer}.hook_resid_post"
    sae = topk_sae.TopKSAE(
        d_in=activation_dim, d_sae=dict_size, k=k,
        model_name=model_name, hook_layer=layer, device=device, dtype=dtype,
        hook_name=hook_name, use_threshold=False,
    )
    sae.load_state_dict({
        "W_enc": W_enc.to(dtype=dtype), "b_enc": b_enc.to(dtype=dtype),
        "W_dec": W_dec.to(dtype=dtype), "b_dec": b_dec.to(dtype=dtype),
        "k": torch.tensor(k, dtype=torch.int),
    })
    sae.to(device=device, dtype=dtype)
    sae.cfg.architecture = "topk"
    sae.cfg.dtype = "bfloat16"
    print(f"✅ Loaded SAE: {dict_size} features, {activation_dim} dimensions, k={k}")
    return sae

# ============================================================================
# Main evaluation function
# ============================================================================

def main():
    """Run AutoInterp evaluation for top finance features."""
    print("\n" + "="*70)
    print("🚀 AutoInterp Evaluation - Top Finance Features")
    print("="*70)
    
    # Setup
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check vLLM server
    if not check_vllm_server():
        print("\n⚠️  vLLM server is not running. Please start it first.")
        return None
    
    # Create Results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load API key
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(f"API key not found at: {API_KEY_PATH}")
    with open(API_KEY_PATH) as f:
        api_key = f.read().strip()
    
    # Get features to evaluate
    if FEATURES_TO_EVALUATE is not None:
        # Use directly provided feature IDs
        features_to_evaluate = FEATURES_TO_EVALUATE
        print(f"\n📖 Using {len(features_to_evaluate)} directly provided features: {features_to_evaluate[:10]}...")
    else:
        # Extract top K features from summary file (finance-specific features)
        print(f"\n📖 Extracting top {TOP_K_FEATURES} finance features from summary file...")
        print(f"   Summary file: {FEATURES_SUMMARY_PATH}")
        features_to_evaluate = extract_top_features(FEATURES_SUMMARY_PATH, TOP_K_FEATURES)
        print(f"✅ Extracted {len(features_to_evaluate)} finance features: {features_to_evaluate[:10]}...")
        print(f"   (These are the top {TOP_K_FEATURES} highest-scoring finance features from the summary)")
    
    # Load SAE
    sae = load_nemotron_sae(
        SAE_CHECKPOINT_PATH, SAE_CONFIG_PATH, MODEL_NAME,
        torch.device(device), torch.bfloat16, LAYER
    )
    d_sae, d_in = sae.W_dec.data.shape
    k_value = int(sae.k.item()) if hasattr(sae, 'k') else 64
    
    # Configure SAE
    hook_name = f"blocks.{LAYER}.hook_resid_post"
    sae.cfg = custom_sae_config.CustomSAEConfig(
        MODEL_NAME, d_in, d_sae, LAYER, hook_name, context_size=CONTEXT_SIZE
    )
    sae.cfg.dtype = LLM_DTYPE
    
    # Validate features
    if any(f < 0 or f >= d_sae for f in features_to_evaluate):
        raise ValueError(f"Features out of range (0-{d_sae-1})")
    
    # Setup
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    sae_id = f"{model_short}_layer{LAYER}_k{k_value}_latents{d_sae}"
    selected_saes = [(sae_id, sae)]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_features = len(features_to_evaluate)
    features_str = f"{num_features}features" if FEATURES_TO_EVALUATE is not None else f"top{TOP_K_FEATURES}_finance"
    log_filename = f"autointerp_{model_short}_layer{LAYER}_{features_str}_{timestamp}.txt"
    log_path = os.path.join(RESULTS_DIR, log_filename)
    
    # Print configuration
    print(f"\n📋 Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Layer: {LAYER}")
    print(f"  Features: {len(features_to_evaluate)} features")
    print(f"  Dataset: {DATASET_NAME} ⚠️  (Financial dataset)")
    print(f"  Total tokens: {TOTAL_TOKENS:,}")
    print(f"  Context size: {CONTEXT_SIZE}")
    print(f"  Results folder: {RESULTS_DIR}")
    print(f"  Log file: {log_filename}\n")
    
    # Create evaluation config
    config = autointerp_config.AutoInterpEvalConfig(
        model_name=MODEL_NAME,
        n_latents=None,
        override_latents=features_to_evaluate,
        random_seed=42,
        llm_batch_size=LLM_BATCH_SIZE,
        llm_dtype=LLM_DTYPE,
        llm_context_size=CONTEXT_SIZE,
        total_tokens=TOTAL_TOKENS,
        dataset_name=DATASET_NAME,  # ⚠️ CRITICAL: Financial dataset!
        scoring=True,
    )
    
    # Setup artifacts path
    num_features = len(features_to_evaluate)
    features_str_artifacts = f"{num_features}features" if FEATURES_TO_EVALUATE is not None else f"top{TOP_K_FEATURES}_finance"
    run_artifact_dir = f"artifacts_{model_short}_layer{LAYER}_{features_str_artifacts}_{timestamp}"
    artifacts_path = os.path.join(RESULTS_DIR, run_artifact_dir)
    os.makedirs(artifacts_path, exist_ok=True)
    
    # Run evaluation
    print(f"🚀 Running evaluation...")
    results_dict = autointerp_main.run_eval(
        config=config,
        selected_saes=selected_saes,
        device=device,
        api_key=api_key,
        output_path=RESULTS_DIR,
        force_rerun=FORCE_RERUN,
        save_logs_path=log_path,
        artifacts_path=artifacts_path,
        provider=PROVIDER,
        api_base_url=EXPLAINER_API_BASE_URL,
        explainer_model=EXPLAINER_MODEL,
    )
    
    # Generate CSV summary
    print(f"\n📊 Generating CSV summary...")
    csv_data = []
    for sae_key, result in results_dict.items():
        if isinstance(result, dict) and "eval_result_unstructured" in result:
            unstructured = result["eval_result_unstructured"]
            if isinstance(unstructured, dict):
                for latent_id, latent_data in unstructured.items():
                    if isinstance(latent_data, dict):
                        csv_data.append({
                            "feature": latent_id,
                            "label": latent_data.get("explanation", ""),
                            "autointerp_score": f"{latent_data.get('score', 0.0):.4f}",
                        })
    
    csv_data.sort(key=lambda x: int(x["feature"]) if str(x["feature"]).isdigit() else 999)
    
    csv_filename = f"nemotron_layer{LAYER}_features_summary_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "label", "autointerp_score"])
        writer.writeheader()
        if csv_data:
            writer.writerows(csv_data)
    
    print(f"   ✅ CSV saved to: {csv_path}")
    print(f"   📊 Generated summary for {len(csv_data)} features")
    
    # Print summary
    print(f"\n✅ Complete! All outputs saved to: {RESULTS_DIR}")
    print(f"  - CSV Summary: {csv_path}")
    print(f"  - Logs: {log_path}")
    print(f"  - Artifacts: {artifacts_path}/")
    
    for sae_key, results in results_dict.items():
        if isinstance(results, dict) and "eval_result_metrics" in results:
            m = results["eval_result_metrics"].get("autointerp", {})
            score = m.get('autointerp_score', 'N/A')
            std = m.get('autointerp_std_dev', 'N/A')
            if score != 'N/A':
                print(f"\n  📊 Score: {score:.4f} ± {std:.4f}")
    
    return results_dict

if __name__ == "__main__":
    main()

