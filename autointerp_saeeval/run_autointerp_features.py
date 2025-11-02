"""Run AutoInterp evaluation for specific SAE features (3 and 6).
Uses local autointerp module from autointerp/ folder and saves all results to Results/ folder.
"""

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

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SAE_PATH = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama3.1_8b_layer19_k32_latents400_lmsys_chat1m_multiGPU"
RESULTS_DIR = str(SCRIPT_DIR / "Results")  # Save directly to Results folder
FEATURES_TO_EVALUATE = [3, 6]
TOTAL_TOKENS = 100_000  # Reduced for testing (increase to 2_000_000 for full run)
CONTEXT_SIZE = 128
LLM_BATCH_SIZE = 32
LLM_DTYPE = "bfloat16"
TORCH_DTYPE = torch.bfloat16
FORCE_RERUN = True  # Set to True to rerun even if results exist (generates artifacts)

# API key path - only look locally for complete independence
API_KEY_PATH = Path(__file__).parent / "openai_api_key.txt"


def load_local_topk_sae(sae_path: str, model_name: str, device: torch.device, dtype: torch.dtype) -> topk_sae.TopKSAE:
    """Load a TopK SAE from local safetensors file."""
    cfg_path = os.path.join(sae_path, "layers.19", "cfg.json")
    with open(cfg_path) as f:
        config = json.load(f)

    sae_file = os.path.join(sae_path, "layers.19", "sae.safetensors")
    state_dict = load_file(sae_file)

    renamed_params = {
        "W_enc": state_dict["encoder.weight"].T,
        "b_enc": state_dict["encoder.bias"],
        "W_dec": state_dict["W_dec"],
        "b_dec": state_dict["b_dec"],
        "k": torch.tensor(config["k"], dtype=torch.int),
    }

    sae = topk_sae.TopKSAE(
        d_in=config["d_in"],
        d_sae=config["d_sae"],
        k=config["k"],
        model_name=model_name,
        hook_layer=config["hook_layer"],
        device=device,
        dtype=dtype,
        hook_name=config["hook_name"],
        use_threshold=False,
    )

    sae.load_state_dict(renamed_params)
    sae.to(device=device, dtype=dtype)
    sae.cfg.architecture = "topk"
    sae.cfg.dtype = LLM_DTYPE
    return sae


def main():
    """Run AutoInterp evaluation for specified features.
    All results, logs, and artifacts are saved to Results/ folder.
    Completely independent from SAEBench (except for required SAE utilities).
    """
    device = setup_environment()
    
    # Create Results directory - all outputs go here
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load API key from local file (self-contained)
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(
            f"OpenAI API key not found at: {API_KEY_PATH}\n"
            f"Please copy your openai_api_key.txt to this folder for complete independence."
        )
    
    with open(API_KEY_PATH) as f:
        api_key = f.read().strip()
    if not api_key:
        raise ValueError(f"API key file is empty: {API_KEY_PATH}")

    # Load SAE
    sae = load_local_topk_sae(SAE_PATH, MODEL_NAME, torch.device(device), TORCH_DTYPE)
    d_sae, d_in = sae.W_dec.data.shape

    # Configure SAE
    with open(os.path.join(SAE_PATH, "layers.19", "cfg.json")) as f:
        context_size = json.load(f).get("context_size", CONTEXT_SIZE)
    
    sae.cfg = custom_sae_config.CustomSAEConfig(
        MODEL_NAME, d_in, d_sae, sae.cfg.hook_layer, sae.cfg.hook_name, context_size=context_size
    )
    sae.cfg.dtype = LLM_DTYPE

    # Validate features
    if any(f < 0 or f >= d_sae for f in FEATURES_TO_EVALUATE):
        raise ValueError(f"Features {FEATURES_TO_EVALUATE} out of range (0-{d_sae-1})")

    # Setup
    sae_id = f"llama3.1_8b_layer{sae.cfg.hook_layer}_k32_latents400"
    selected_saes = [(sae_id, sae)]

    # Create unique log filename based on model, SAE, features, and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = MODEL_NAME.split("/")[-1].replace("-", "_").lower()
    features_str = "_".join(map(str, FEATURES_TO_EVALUATE))
    log_filename = f"autointerp_{model_short}_layer{sae.cfg.hook_layer}_features{features_str}_{timestamp}.txt"
    log_path = os.path.join(RESULTS_DIR, log_filename)

    print(f"\nAutoInterp Evaluation")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Features: {FEATURES_TO_EVALUATE}")
    print(f"  Tokens: {TOTAL_TOKENS:,}")
    print(f"  Results folder: {RESULTS_DIR}")
    print(f"  Log file: {log_filename}\n")

    # Run evaluation
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
    )

    # Check artifacts location (autointerp creates subfolder "autointerp/")
    autointerp_artifacts_dir = os.path.join(artifacts_path, "autointerp")
    artifact_files = []
    if os.path.exists(autointerp_artifacts_dir):
        artifact_files = [f for f in os.listdir(autointerp_artifacts_dir) if os.path.isfile(os.path.join(autointerp_artifacts_dir, f))]

    # Summary
    print(f"\n✓ Complete! All outputs saved to: {RESULTS_DIR}")
    print(f"  - Results JSON: {RESULTS_DIR}/*_eval_results.json")
    print(f"  - Logs: {log_path}")
    print(f"  - Artifacts: {autointerp_artifacts_dir}/")
    if artifact_files:
        print(f"    ({len(artifact_files)} artifact file(s) generated)")
    else:
        print(f"    (No artifacts - evaluation may have been skipped)")
    
    for sae_key, results in results_dict.items():
        if isinstance(results, dict) and "eval_result_metrics" in results:
            m = results["eval_result_metrics"].get("autointerp", {})
            score = m.get('autointerp_score', 'N/A')
            std = m.get('autointerp_std_dev', 'N/A')
            if score != 'N/A':
                print(f"\n  Score: {score:.4f} ± {std:.4f}")
    
    return results_dict


if __name__ == "__main__":
    main()

