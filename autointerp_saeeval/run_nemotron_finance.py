"""Run AutoInterp evaluation for Nemotron finance features.
Simplified version that saves examples used for labeling.
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import requests

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from autointerp import eval_config as autointerp_config
from autointerp import main as autointerp_main
import sae_bench.custom_saes.custom_sae_config as custom_sae_config
import sae_bench.custom_saes.topk_sae as topk_sae

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
SAE_CHECKPOINT_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/nemotron_nano_layer28_features35840_k64/trainer_0/ae.pt"
SAE_CONFIG_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/train/trained_models/nemotron_nano_layer28_features35840_k64/trainer_0/config.json"
FEATURES_SUMMARY_PATH = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/EndtoEnd/Autointerp/nemotron_finance_features/top_finance_features_summary.txt"

LAYER = 28
TOP_K_FEATURES = 100
RESULTS_DIR = str(SCRIPT_DIR / "Results")

# Data collection parameters
TOTAL_TOKENS = 500_000  # Total tokens to sample from dataset
DATASET_NAME = "ashraq/financial-news"  # ⚠️ IMPORTANT: Must be financial dataset!
CONTEXT_SIZE = 128
LLM_BATCH_SIZE = 32
LLM_DTYPE = "bfloat16"

# Example selection parameters (see README.md for details)
N_TOP_EX_FOR_GENERATION = 10  # Top examples for explanation
N_IW_SAMPLED_EX_FOR_GENERATION = 5  # Importance-weighted examples
ACT_THRESHOLD_FRAC = 0.01  # Activation threshold fraction

# vLLM configuration
PROVIDER = "vllm"
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"
API_KEY_PATH = SCRIPT_DIR / "openai_api_key.txt"

FORCE_RERUN = True

# ============================================================================
# Nemotron-specific patches (handles 4480-dim mixer)
# ============================================================================

def setup_nemotron_patches():
    """Setup patches for Nemotron architecture compatibility."""
    import transformer_lens.loading_from_pretrained as loading_from_pretrained
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sae_bench.sae_bench_utils.activation_collection as activation_collection
    
    # Patch model name mapping
    _original_get_official_model_name = loading_from_pretrained.get_official_model_name
    def patched_get_official_model_name(model_name: str):
        if "nemotron" in model_name.lower():
            return "meta-llama/Llama-3.1-8B-Instruct"  # Map to Llama
        return _original_get_official_model_name(model_name)
    loading_from_pretrained.get_official_model_name = patched_get_official_model_name
    
    # Patch HookedTransformer
    from transformer_lens import HookedTransformer
    _original_from_pretrained = HookedTransformer.from_pretrained_no_processing
    @classmethod
    def patched_from_pretrained(cls, model_name, **kwargs):
        if "nemotron" in model_name.lower():
            kwargs.setdefault("trust_remote_code", True)
        return _original_from_pretrained(model_name, **kwargs)
    HookedTransformer.from_pretrained_no_processing = patched_from_pretrained
    
    # Patch activation collection for 4480-dim mixer
    _original_collect = activation_collection.collect_sae_activations
    _original_sparsity = activation_collection.get_feature_activation_sparsity
    _nemotron_model = None
    _nemotron_tokenizer = None
    _nemotron_layer = None
    
    def patched_collect(tokens, model, sae, batch_size, layer, hook_name, 
                       mask_bos_pad_eos_tokens=False, selected_latents=None, activation_dtype=None):
        global _nemotron_model, _nemotron_tokenizer, _nemotron_layer
        if hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480:
            if _nemotron_model is None or _nemotron_layer != layer:
                print(f"   Loading raw Nemotron model for layer {layer}...")
                _nemotron_model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME, trust_remote_code=True, device_map="cuda", torch_dtype=torch.float16
                )
                _nemotron_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
                _nemotron_layer = layer
                _nemotron_model.eval()
            
            sae_acts = []
            device = next(_nemotron_model.parameters()).device
            from tqdm import tqdm
            for i in tqdm(range(0, tokens.shape[0], batch_size), desc="Collecting mixer activations"):
                tokens_BL = tokens[i:i+batch_size].to(device)
                with torch.no_grad():
                    mixer_acts = []
                    def hook_fn(module, input, output): mixer_acts.append(output.detach())
                    handle = _nemotron_model.backbone.layers[layer].mixer.register_forward_hook(hook_fn)
                    _ = _nemotron_model(input_ids=tokens_BL)
                    handle.remove()
                    mixer_act = mixer_acts[0].to(dtype=sae.dtype, device=sae.device)
                
                sae_act = sae.encode(mixer_act)
                if selected_latents is not None:
                    sae_act = sae_act[:, :, selected_latents]
                if mask_bos_pad_eos_tokens:
                    from sae_bench.sae_bench_utils.activation_collection import get_bos_pad_eos_mask
                    mask = get_bos_pad_eos_mask(tokens_BL, _nemotron_tokenizer).to(device=sae_act.device)
                    sae_act = sae_act * mask[:, :, None]
                if activation_dtype:
                    sae_act = sae_act.to(dtype=activation_dtype)
                sae_acts.append(sae_act.cpu())
            return torch.cat(sae_acts, dim=0)
        return _original_collect(tokens, model, sae, batch_size, layer, hook_name,
                                mask_bos_pad_eos_tokens, selected_latents, activation_dtype)
    
    def patched_sparsity(tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens=False):
        if hasattr(sae, 'W_enc') and sae.W_enc.shape[0] == 4480:
            sae_acts = patched_collect(tokens, model, sae, batch_size, layer, hook_name,
                                       mask_bos_pad_eos_tokens, None, None)
            running_sum = (sae_acts > 0).float().sum(dim=(0, 1))
            total = sae_acts.shape[0] * sae_acts.shape[1]
            return (running_sum / total if total > 0 else torch.zeros_like(running_sum)).to(sae.device)
        return _original_sparsity(tokens, model, sae, batch_size, layer, hook_name, mask_bos_pad_eos_tokens)
    
    activation_collection.collect_sae_activations = patched_collect
    activation_collection.get_feature_activation_sparsity = patched_sparsity
    print("✅ Nemotron patches applied")

# ============================================================================
# Helper functions
# ============================================================================

def check_vllm_server():
    """Check if vLLM server is running."""
    try:
        response = requests.get(f"{EXPLAINER_API_BASE_URL.rstrip('/v1')}/v1/models", timeout=5)
        if response.status_code == 200:
            print(f"✅ vLLM server running at {EXPLAINER_API_BASE_URL}")
            return True
    except:
        pass
    print(f"❌ vLLM server not running. Start with: python -m vllm.entrypoints.openai.api_server --model {EXPLAINER_MODEL} --port 8002")
    return False

def extract_features(summary_path: str, top_k: int) -> list[int]:
    """Extract feature indices from summary file."""
    import re
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

def load_nemotron_sae(checkpoint_path: str, config_path: str, model_name: str, 
                     device: torch.device, dtype: torch.dtype, layer: int) -> topk_sae.TopKSAE:
    """Load Nemotron SAE from checkpoint."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    trainer_config = config["trainer"]
    dict_size = trainer_config["dict_size"]
    k = trainer_config["k"]
    activation_dim = trainer_config["activation_dim"]
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    W_enc = checkpoint["encoder.weight"].T
    b_enc = checkpoint["encoder.bias"]
    W_dec = checkpoint["decoder.weight"].T
    b_dec = checkpoint.get("b_dec", torch.zeros(activation_dim))
    
    sae = topk_sae.TopKSAE(
        d_in=activation_dim, d_sae=dict_size, k=k, model_name=model_name,
        hook_layer=layer, device=device, dtype=dtype,
        hook_name=f"blocks.{layer}.hook_resid_post", use_threshold=False,
    )
    sae.load_state_dict({
        "W_enc": W_enc.to(dtype=dtype),
        "b_enc": b_enc.to(dtype=dtype),
        "W_dec": W_dec.to(dtype=dtype),
        "b_dec": b_dec.to(dtype=dtype),
        "k": torch.tensor(k, dtype=torch.int),
    })
    sae.to(device=device, dtype=dtype)
    sae.cfg.architecture = "topk"
    return sae

# ============================================================================
# Main evaluation function
# ============================================================================

def main():
    """Run AutoInterp evaluation with example visibility."""
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not check_vllm_server():
        return None
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load API key
    api_key = ""
    if API_KEY_PATH.exists():
        api_key = API_KEY_PATH.read_text().strip()
    
    # Setup Nemotron patches
    setup_nemotron_patches()
    
    # Extract features
    print(f"\n📖 Extracting top {TOP_K_FEATURES} features...")
    features = extract_features(FEATURES_SUMMARY_PATH, TOP_K_FEATURES)
    print(f"✅ Extracted {len(features)} features")
    
    # Load SAE
    print(f"\n📂 Loading SAE...")
    sae = load_nemotron_sae(SAE_CHECKPOINT_PATH, SAE_CONFIG_PATH, MODEL_NAME,
                            torch.device(device), torch.bfloat16, LAYER)
    d_sae, d_in = sae.W_dec.data.shape
    k_value = int(sae.k.item()) if hasattr(sae, 'k') else 64
    print(f"✅ SAE: {d_sae} features, {d_in} dims, k={k_value}")
    
    sae.cfg = custom_sae_config.CustomSAEConfig(
        MODEL_NAME, d_in, d_sae, LAYER, f"blocks.{LAYER}.hook_resid_post", context_size=CONTEXT_SIZE
    )
    sae.cfg.dtype = LLM_DTYPE
    
    # Create config with ALL parameters visible
    config = autointerp_config.AutoInterpEvalConfig(
        model_name=MODEL_NAME,
        override_latents=features,
        random_seed=42,
        llm_batch_size=LLM_BATCH_SIZE,
        llm_dtype=LLM_DTYPE,
        llm_context_size=CONTEXT_SIZE,
        total_tokens=TOTAL_TOKENS,
        dataset_name=DATASET_NAME,  # ⚠️ CRITICAL: Financial dataset!
        scoring=True,
        n_top_ex_for_generation=N_TOP_EX_FOR_GENERATION,
        n_iw_sampled_ex_for_generation=N_IW_SAMPLED_EX_FOR_GENERATION,
        act_threshold_frac=ACT_THRESHOLD_FRAC,
    )
    
    # Print ALL configuration parameters
    print(f"\n📋 Configuration Parameters:")
    print(f"  Dataset: {DATASET_NAME} ⚠️  (Must be financial dataset for finance features!)")
    print(f"  Total tokens: {TOTAL_TOKENS:,}")
    print(f"  Context size: {CONTEXT_SIZE}")
    print(f"  LLM batch size: {LLM_BATCH_SIZE}")
    print(f"  Top examples for generation: {N_TOP_EX_FOR_GENERATION}")
    print(f"  IW examples for generation: {N_IW_SAMPLED_EX_FOR_GENERATION}")
    print(f"  Activation threshold fraction: {ACT_THRESHOLD_FRAC}")
    print(f"  Explainer model: {EXPLAINER_MODEL}")
    print(f"  vLLM URL: {EXPLAINER_API_BASE_URL}")
    
    # Run evaluation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    sae_id = f"{model_short}_layer{LAYER}_k{k_value}_latents{d_sae}"
    artifacts_path = os.path.join(RESULTS_DIR, f"artifacts_{model_short}_layer{LAYER}_top{TOP_K_FEATURES}_{timestamp}")
    os.makedirs(artifacts_path, exist_ok=True)
    
    log_path = os.path.join(RESULTS_DIR, f"autointerp_{model_short}_layer{LAYER}_top{TOP_K_FEATURES}_{timestamp}.txt")
    
    print(f"\n🚀 Running evaluation...")
    results_dict = autointerp_main.run_eval(
        config=config,
        selected_saes=[(sae_id, sae)],
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
    
    # Extract examples used for each feature and save them
    print(f"\n📊 Generating CSV and saving examples...")
    csv_data = []
    examples_data = {}
    
    for sae_key, result in results_dict.items():
        if isinstance(result, dict) and "eval_result_unstructured" in result:
            unstructured = result["eval_result_unstructured"]
            if isinstance(unstructured, dict):
                for latent_id, latent_data in unstructured.items():
                    if isinstance(latent_data, dict):
                        explanation = latent_data.get("explanation", "")
                        score = latent_data.get("score", 0.0)
                        csv_data.append({
                            "feature": latent_id,
                            "label": explanation,
                            "autointerp_score": f"{score:.4f}",
                        })
                        
                        # Extract examples from logs
                        logs = latent_data.get("logs", "")
                        examples_text = ""
                        if logs and "Generation phase" in logs:
                            # Extract the examples table (between "Generation phase" and "Scoring phase" or end)
                            gen_section = logs.split("Generation phase")[1]
                            if "Scoring phase" in gen_section:
                                gen_section = gen_section.split("Scoring phase")[0]
                            # Get the examples table (starts with "Top act")
                            if "Top act" in gen_section:
                                table_start = gen_section.find("Top act")
                                examples_text = gen_section[table_start:table_start+2000]  # First 2000 chars of table
                        
                        examples_data[latent_id] = {
                            "explanation": explanation,
                            "score": float(score),
                            "generation_examples": examples_text if examples_text else "Not available in logs",
                        }
    
    # Sort and save CSV
    csv_data.sort(key=lambda x: int(x["feature"]) if str(x["feature"]).isdigit() else 999)
    csv_path = os.path.join(RESULTS_DIR, f"nemotron_layer{LAYER}_features_summary_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "label", "autointerp_score"])
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Save examples data
    examples_path = os.path.join(RESULTS_DIR, f"nemotron_layer{LAYER}_examples_{timestamp}.json")
    with open(examples_path, 'w') as f:
        json.dump(examples_data, f, indent=2)
    
    print(f"\n✅ Output Files:")
    print(f"  📄 CSV Summary: {csv_path}")
    print(f"  📄 Examples Data: {examples_path}")
    print(f"  📄 Full Logs: {log_path}")
    print(f"  📄 JSON Results: {RESULTS_DIR}/*_eval_results.json")
    print(f"\n💡 To view examples used for a feature, check the 'generation_examples' field in:")
    print(f"   {examples_path}")
    
    # Print summary
    for sae_key, results in results_dict.items():
        if isinstance(results, dict) and "eval_result_metrics" in results:
            m = results["eval_result_metrics"].get("autointerp", {})
            score = m.get('autointerp_score', 'N/A')
            std = m.get('autointerp_std_dev', 'N/A')
            if score != 'N/A':
                print(f"\n📊 Overall Score: {score:.4f} ± {std:.4f}")
    
    return results_dict

if __name__ == "__main__":
    main()

