"""Run AutoInterp evaluation for Nemotron features using vLLM.
Processes features from top_finance_features_summary.txt and top_reasoning_features_summary.txt
"""

import csv
import json
import os
import re
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

# Still need sae_bench utilities for SAE loading
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.sae_bench_utils.dataset_utils as dataset_utils

# Patch dataset loading for financial-news dataset
_original_load_and_tokenize = dataset_utils.load_and_tokenize_dataset

def patched_load_and_tokenize_dataset(dataset_name, ctx_len, num_tokens, tokenizer, column_name="text", add_bos=True):
    """Patched version that uses 'headline' column for financial-news dataset."""
    if "financial-news" in dataset_name:
        column_name = "headline"
    return _original_load_and_tokenize(dataset_name, ctx_len, num_tokens, tokenizer, column_name, add_bos)

dataset_utils.load_and_tokenize_dataset = patched_load_and_tokenize_dataset

# Configuration (needed before patches)
MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
BASE_DIR = Path("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Nemotron_EndToEnd")
SAE_PATH = Path("/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_sae_converted")
FINANCE_FEATURES_FILE = BASE_DIR / "feature_discovery" / "nemotron_finance_features" / "top_finance_features_summary.txt"
REASONING_FEATURES_FILE = BASE_DIR / "feature_discovery" / "nemotron_reasoning_features" / "top_reasoning_features_summary.txt"
LAYER = 28
RESULTS_DIR = str(SCRIPT_DIR / "Results")
TOTAL_TOKENS = 500_000  # 500k tokens for evaluation
CONTEXT_SIZE = 1024
LLM_BATCH_SIZE = 16
LLM_DTYPE = "bfloat16"
TORCH_DTYPE = torch.bfloat16
FORCE_RERUN = True

# Provider configuration - using vLLM
PROVIDER = "vllm"
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"
API_KEY_PATH = Path(__file__).parent / "openai_api_key.txt"

# Patch transformer_lens for Nemotron support
if "nemotron" in MODEL_NAME.lower():
    print("🔧 Patching transformer_lens for Nemotron support...")
    
    # Patch get_official_model_name to allow Nemotron (treat as Llama-like)
    import transformer_lens.loading_from_pretrained as loading_from_pretrained
    _original_get_official_model_name = loading_from_pretrained.get_official_model_name
    
    def patched_get_official_model_name(model_name):
        if "nemotron" in model_name.lower():
            # Nemotron is Llama-based, use Llama-3.1-8B-Instruct as base
            return "meta-llama/Llama-3.1-8B-Instruct"
        return _original_get_official_model_name(model_name)
    
    loading_from_pretrained.get_official_model_name = patched_get_official_model_name
    
    # Patch HookedTransformer to load Nemotron directly (similar to FinBERT approach)
    from transformer_lens import HookedTransformer
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
    _original_from_pretrained = HookedTransformer.from_pretrained_no_processing
    
    @classmethod
    def patched_from_pretrained(cls, model_name, **kwargs):
        if "nemotron" in model_name.lower() or ("nvidia" in model_name.lower() and "nemotron" in model_name.lower()):
            print(f"🔧 Creating minimal HookedTransformer for Nemotron ({model_name})...")
            print("   Note: Using raw Nemotron model for activations, HookedTransformer is just a placeholder")
            # Load only config (not the full model to save memory)
            nemotron_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            
            # Create HookedTransformer config matching Nemotron architecture
            from transformers import AutoTokenizer
            tokenizer = kwargs.get("tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            # Create config - use hidden_size for d_model, but we'll need to handle mixer separately
            nemotron_cfg_dict = {
                "d_model": nemotron_config.hidden_size,  # 4096
                "d_head": nemotron_config.hidden_size // nemotron_config.num_attention_heads,
                "n_heads": nemotron_config.num_attention_heads,
                "n_layers": nemotron_config.num_hidden_layers,
                "n_ctx": nemotron_config.max_position_embeddings,
                "d_vocab": nemotron_config.vocab_size,
                "tokenizer_name": model_name,
                "model_name": model_name,
                "act_fn": "silu",  # Nemotron uses SiLU
                "normalization_type": "RMS",  # Nemotron uses RMSNorm (use "RMS" for transformer_lens)
                "d_mlp": 14336,  # Standard MLP dimension in transformer_lens
                # Note: The actual mixer dimension (4480) will be handled via custom hook
            }
            nemotron_cfg = HookedTransformerConfig(**nemotron_cfg_dict)
            
            # Create model structure on CPU (we won't use it for actual computation)
            # The raw model will be loaded separately in the activation collection patch
            model = HookedTransformer(nemotron_cfg, tokenizer=tokenizer,
                                      move_to_device=False,  # Keep on CPU to save GPU memory
                                      default_padding_side=kwargs.get("default_padding_side", "right"))
            
            print("   ✅ Minimal HookedTransformer created (raw model will be used for activations)")
            return model
        
        return _original_from_pretrained(model_name, **kwargs)
    
    HookedTransformer.from_pretrained_no_processing = patched_from_pretrained
    
    # Patch activation collection to use raw Nemotron model for mixer activations
    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    from transformers import AutoModelForCausalLM as TransformersAutoModel
    from transformers import AutoTokenizer as TransformersAutoTokenizer
    
    _original_collect_sae_activations = activation_collection.collect_sae_activations
    _original_get_feature_activation_sparsity = activation_collection.get_feature_activation_sparsity
    
    # Store raw model for activation collection
    _nemotron_raw_model = None
    _nemotron_raw_tokenizer = None
    _nemotron_layer = None
    
    def patched_collect_sae_activations(
        tokens, model, sae, batch_size, layer, hook_name, 
        mask_bos_pad_eos_tokens=False, selected_latents=None, activation_dtype=None
    ):
        """Patched version that uses raw Nemotron model for mixer activations (4480 dim)."""
        global _nemotron_raw_model, _nemotron_raw_tokenizer, _nemotron_layer
        
        # Check if this is Nemotron with dimension 4480
        if (hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480 and 
            "nemotron" in str(model.cfg.model_name).lower()):
            
            print(f"   Using raw Nemotron model for mixer activations (dim 4480)...")
            
            # Load raw model if not already loaded
            if _nemotron_raw_model is None or _nemotron_layer != layer:
                print(f"   Loading raw Nemotron model for layer {layer}...")
                # Load on GPU (CUDA) since Nemotron model requires CUDA device
                _nemotron_raw_model = TransformersAutoModel.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, 
                    device_map="cuda",  # Load on GPU
                    torch_dtype=torch.float16  # Use float16 to save memory
                )
                _nemotron_raw_tokenizer = TransformersAutoTokenizer.from_pretrained(
                    MODEL_NAME, trust_remote_code=True
                )
                _nemotron_layer = layer
                _nemotron_raw_model.eval()
            
            sae_acts = []
            device = next(_nemotron_raw_model.parameters()).device
            
            from tqdm import tqdm
            for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Collecting mixer activations"):
                tokens_BL = tokens[i : i + batch_size].to(device)
                
                # Get mixer activations from raw model
                with torch.no_grad():
                    # Hook the mixer output
                    mixer_activations = []
                    def hook_fn(module, input, output):
                        mixer_activations.append(output.detach())
                    
                    handle = _nemotron_raw_model.backbone.layers[layer].mixer.register_forward_hook(hook_fn)
                    
                    # Forward pass
                    outputs = _nemotron_raw_model(input_ids=tokens_BL)
                    
                    handle.remove()
                    
                    if mixer_activations:
                        mixer_act = mixer_activations[0]  # [batch, seq_len, 4480]
                    else:
                        # Fallback: manually compute mixer output
                        hidden_states = outputs.hidden_states[layer] if hasattr(outputs, 'hidden_states') else None
                        if hidden_states is None:
                            # Get from model's intermediate outputs
                            with _nemotron_raw_model.backbone.layers[layer].mixer.register_forward_hook(hook_fn):
                                _ = _nemotron_raw_model(input_ids=tokens_BL)
                            mixer_act = mixer_activations[0] if mixer_activations else None
                        
                        if mixer_act is None:
                            raise RuntimeError("Could not extract mixer activations from Nemotron model")
                
                # Convert to SAE dtype before encoding
                mixer_act = mixer_act.to(dtype=sae.dtype, device=sae.device)
                
                # Apply SAE encoding
                sae_act_BLF = sae.encode(mixer_act)  # [batch, seq_len, d_sae]
                
                if selected_latents is not None:
                    sae_act_BLF = sae_act_BLF[:, :, selected_latents]
                
                if mask_bos_pad_eos_tokens:
                    from sae_bench.sae_bench_utils.activation_collection import get_bos_pad_eos_mask
                    attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, _nemotron_raw_tokenizer)
                else:
                    attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)
                
                attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)
                sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]
                
                if activation_dtype is not None:
                    sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)
                
                sae_acts.append(sae_act_BLF.cpu())
            
            all_sae_acts_BLF = torch.cat(sae_acts, dim=0)
            return all_sae_acts_BLF
        
        # For non-Nemotron or standard dimensions, use original function
        return _original_collect_sae_activations(
            tokens, model, sae, batch_size, layer, hook_name,
            mask_bos_pad_eos_tokens, selected_latents, activation_dtype
        )
    
    def patched_get_feature_activation_sparsity(
        tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens=False
    ):
        """Patched version for sparsity calculation."""
        global _nemotron_raw_model, _nemotron_raw_tokenizer
        
        if (hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480 and 
            "nemotron" in str(model.cfg.model_name).lower()):
            
            # Use patched collect_sae_activations and compute sparsity
            sae_acts = patched_collect_sae_activations(
                tokens, model, sae, batch_size, layer, hook_name,
                mask_bos_pad_eos_tokens, selected_latents=None, activation_dtype=None
            )
            
            device = sae.device
            running_sum_F = (sae_acts > 0).to(dtype=torch.float32).sum(dim=(0, 1))
            total_tokens = sae_acts.shape[0] * sae_acts.shape[1]
            
            sparsity_F = running_sum_F / total_tokens if total_tokens > 0 else torch.zeros_like(running_sum_F)
            return sparsity_F.to(device)
        
        return _original_get_feature_activation_sparsity(
            tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens
        )
    
    activation_collection.collect_sae_activations = patched_collect_sae_activations
    activation_collection.get_feature_activation_sparsity = patched_get_feature_activation_sparsity
    
    print("✅ Nemotron patches applied successfully")


def extract_features_from_summary(summary_path: Path, top_k: int = 100) -> list[int]:
    """Extract feature indices from summary file."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    features = []
    pattern = re.compile(r'^\s*\d+\.\s+Feature\s+(\d+):')
    
    with open(summary_path, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                features.append(int(match.group(1)))
                if len(features) >= top_k:
                    break
    
    return features


def setup_environment():
    """Setup CUDA environment and return device."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    return device


def load_nemotron_sae(sae_path: Path, model_name: str, device: torch.device, dtype: torch.dtype, layer: int) -> topk_sae.TopKSAE:
    """Load Nemotron SAE from converted format."""
    layer_dir = sae_path / f"layers.{layer}"
    cfg_path = layer_dir / "cfg.json"
    
    if not cfg_path.exists():
        raise FileNotFoundError(f"SAE config not found: {cfg_path}")
    
    with open(cfg_path) as f:
        config = json.load(f)
    
    sae_file = layer_dir / "sae.safetensors"
    if not sae_file.exists():
        raise FileNotFoundError(f"SAE file not found: {sae_file}")
    
    state_dict = load_file(str(sae_file))
    
    # Config might use different keys - check both formats
    d_in = config.get("d_in", config.get("activation_dim", None))
    d_sae = config.get("d_sae", config.get("num_latents", config.get("dict_size", None)))
    k = config.get("k", 64)  # Default to 64 if not found
    
    if d_in is None or d_sae is None:
        raise ValueError(f"Could not determine SAE dimensions from config: {config}")
    
    # Convert to SAEBench format
    renamed_params = {
        "W_enc": state_dict["encoder.weight"].T,  # [d_sae, d_in] -> [d_in, d_sae]
        "b_enc": state_dict["encoder.bias"],
        "W_dec": state_dict["W_dec"],  # Already [d_sae, d_in]
        "b_dec": state_dict["b_dec"],
        "k": torch.tensor(k, dtype=torch.int),
    }
    
    # For Nemotron, SAE was trained on backbone.layers[28].mixer (dimension 4480)
    # However, transformer_lens loads Nemotron as Llama which has different dimensions:
    # - MLP output: 14336 (not 4480)
    # - Residual stream: 4096 (not 4480)
    # This is a known limitation - Nemotron's architecture doesn't map cleanly to transformer_lens
    # 
    # We'll use hook_resid_post as a fallback, but this will cause dimension mismatches
    # The proper solution would be to use autointerp_full with nnsight instead of transformer_lens
    hook_name = config.get("hook_name", f"blocks.{layer}.hook_resid_post")
    
    if d_in == 4480:
        print(f"⚠️  WARNING: SAE expects dimension 4480 (Nemotron mixer), but transformer_lens")
        print(f"   doesn't support this dimension. The SAE was trained on Nemotron's actual")
        print(f"   architecture, but transformer_lens maps it to Llama (4096/14336 dimensions).")
        print(f"   This will likely cause dimension mismatch errors.")
        print(f"   Consider using autointerp_full with nnsight for Nemotron models.")
        # Try to use resid_post anyway - might need manual dimension adjustment
        hook_name = f"blocks.{layer}.hook_resid_post"
    elif d_in == 4096:
        hook_name = f"blocks.{layer}.hook_resid_post"  # Residual stream
    
    sae = topk_sae.TopKSAE(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        model_name=model_name,
        hook_layer=layer,
        device=device,
        dtype=dtype,
        hook_name=hook_name,
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


def run_evaluation(features: list[int], feature_type: str, sae: topk_sae.TopKSAE, device: str, api_key: str):
    """Run AutoInterp evaluation for a set of features."""
    print(f"\n{'='*70}")
    print(f"Running AutoInterp for {feature_type} features: {len(features)} features")
    print(f"{'='*70}")
    
    # Create unique log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    features_str = f"{len(features)}features"
    log_filename = f"autointerp_{model_short}_layer{LAYER}_{feature_type}_{features_str}_{timestamp}.txt"
    log_path = os.path.join(RESULTS_DIR, log_filename)
    
    print(f"  Model: {MODEL_NAME}")
    print(f"  Features: {features[:10]}..." if len(features) > 10 else f"  Features: {features}")
    print(f"  Tokens: {TOTAL_TOKENS:,}")
    print(f"  Log file: {log_filename}\n")
    
    # Setup SAE
    sae_id = f"nemotron_layer{sae.cfg.hook_layer}_features{sae.W_dec.shape[0]}_k{sae.k}"
    selected_saes = [(sae_id, sae)]
    
    # Run evaluation
    config = autointerp_config.AutoInterpEvalConfig(
        model_name=MODEL_NAME,
        n_latents=None,
        override_latents=features,
        random_seed=42,
        llm_batch_size=LLM_BATCH_SIZE,
        llm_dtype=LLM_DTYPE,
        llm_context_size=CONTEXT_SIZE,
        total_tokens=TOTAL_TOKENS,
        scoring=True,
        dataset_name="ashraq/financial-news",  # Use financial-news dataset
        dead_latent_threshold=-1.0,
        act_threshold_frac=0.00001,
        max_tokens_in_explanation=40,
        use_demos_in_explanation=True,
    )
    
    # Save artifacts
    run_artifact_dir = f"artifacts_{model_short}_layer{LAYER}_{feature_type}_{timestamp}"
    artifacts_path = os.path.join(RESULTS_DIR, run_artifact_dir)
    os.makedirs(artifacts_path, exist_ok=True)
    
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
    
    # Print summary
    for sae_key, results in results_dict.items():
        if isinstance(results, dict) and "eval_result_metrics" in results:
            m = results["eval_result_metrics"].get("autointerp", {})
            score = m.get('autointerp_score', 'N/A')
            std = m.get('autointerp_std_dev', 'N/A')
            if score != 'N/A':
                print(f"\n  {feature_type.capitalize()} Features Score: {score:.4f} ± {std:.4f}")
    
    return results_dict


def main():
    """Run AutoInterp evaluation for both finance and reasoning features."""
    device = setup_environment()
    
    # Create Results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check vLLM server
    print(f"🔍 Checking vLLM server at {EXPLAINER_API_BASE_URL}...")
    if not check_vllm_server():
        print(f"❌ vLLM server is not running at {EXPLAINER_API_BASE_URL}")
        print(f"   Please start the vLLM server first using:")
        print(f"   bash start_vllm_server_72b.sh")
        raise RuntimeError("vLLM server is not running")
    print(f"✅ vLLM server is running")
    
    # Load API key (optional for vLLM)
    if not API_KEY_PATH.exists():
        api_key = ""
        print(f"ℹ️  API key file not found - vLLM doesn't require authentication")
    else:
        with open(API_KEY_PATH) as f:
            api_key = f.read().strip()
        if not api_key:
            api_key = ""
            print(f"ℹ️  API key file is empty - vLLM doesn't require authentication")
    
    print(f"Using vLLM provider with API base URL: {EXPLAINER_API_BASE_URL}")
    print(f"Explainer model: {EXPLAINER_MODEL}\n")
    
    # Extract features from summary files (5 features total)
    print("📊 Extracting features from summary files...")
    finance_features = extract_features_from_summary(FINANCE_FEATURES_FILE, top_k=5)
    
    print(f"✅ Extracted {len(finance_features)} finance features")
    
    # Load SAE
    print(f"\n📥 Loading Nemotron SAE from {SAE_PATH}...")
    sae = load_nemotron_sae(SAE_PATH, MODEL_NAME, torch.device(device), TORCH_DTYPE, LAYER)
    d_sae, d_in = sae.W_dec.data.shape
    print(f"✅ SAE loaded: {d_sae} features, K={sae.k}, activation_dim={d_in}")
    
    # Configure SAE
    layer_dir = SAE_PATH / f"layers.{LAYER}"
    with open(layer_dir / "cfg.json") as f:
        config = json.load(f)
        context_size = config.get("context_size", CONTEXT_SIZE)
    
    sae.cfg = custom_sae_config.CustomSAEConfig(
        MODEL_NAME, d_in, d_sae, sae.cfg.hook_layer, sae.cfg.hook_name, context_size=context_size
    )
    sae.cfg.dtype = LLM_DTYPE
    
    # Validate features
    if any(f < 0 or f >= d_sae for f in finance_features):
        invalid = [f for f in finance_features if f < 0 or f >= d_sae]
        raise ValueError(f"Invalid features: {invalid} (valid range: 0-{d_sae-1})")
    
    # Run evaluation for finance features
    finance_results = run_evaluation(finance_features, "finance", sae, device, api_key)
    
    # Generate CSV summary - similar to finbert script
    print(f"\n📊 Generating CSV summary...")
    csv_data = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for sae_key, result in finance_results.items():
        if isinstance(result, dict) and "eval_result_unstructured" in result:
            unstructured = result["eval_result_unstructured"]
            if isinstance(unstructured, dict):
                for latent_id, latent_data in unstructured.items():
                    if isinstance(latent_data, dict):
                        explanation = latent_data.get("explanation", "")
                        score = latent_data.get("score", 0.0)
                        
                        csv_data.append({
                            "layer": LAYER,
                            "feature": latent_id,
                            "label": explanation,  # Full explanation, no truncation
                            "autointerp_score": f"{score:.4f}",
                        })
    
    # Sort by feature ID (convert to int for sorting)
    csv_data.sort(key=lambda x: int(x["feature"]) if str(x["feature"]).isdigit() else 999)
    
    # Write CSV to Results folder
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    csv_filename = f"nemotron_layer{LAYER}_features_summary_{timestamp}.csv"
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
    print(f"\n{'='*70}")
    print(f"✓ Complete! All outputs saved to: {RESULTS_DIR}")
    print(f"{'='*70}")
    print(f"  - Finance features: {len(finance_features)} features evaluated")
    print(f"  - Results JSON: {RESULTS_DIR}/*_eval_results.json")
    print(f"  - CSV Summary: {csv_path}")
    print(f"  - Logs: {RESULTS_DIR}/*.txt")
    print(f"  - Artifacts: {RESULTS_DIR}/artifacts_*/")
    
    return {
        "finance": finance_results
    }


if __name__ == "__main__":
    main()

