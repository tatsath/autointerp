#!/usr/bin/env python3
"""
Small test script to validate steering pipeline fixes before running full pipeline.
Tests with minimal data: 2 prompts, 2 features, 1 batch.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sae_lens import HookedSAETransformer
from sae_pipeline.steering import run_steering_experiment
from sae_pipeline.steering_utils import load_local_sae

def test_pipeline():
    print("="*70)
    print("🧪 Testing Steering Pipeline Fixes")
    print("="*70)
    print()
    
    # Configuration
    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    LARGE_SAE_MODEL = "/home/nvidia/work/autointerp/converted_safetensors"
    LAYER = 19
    DEVICE = "cuda:0"
    DATASET_REPO = "jyanimaulik/yahoo_finance_stockmarket_news"
    
    # Test with minimal data
    TEST_FEATURES = [0, 1]  # Just 2 features
    TEST_PROMPTS = [
        "The stock market is",
        "Technology companies are"
    ]
    OUTPUT_FOLDER = "test_steering_outputs"
    
    print(f"📋 Configuration:")
    print(f"   Model: {BASE_MODEL}")
    print(f"   SAE: {LARGE_SAE_MODEL}")
    print(f"   Layer: {LAYER}")
    print(f"   Device: {DEVICE}")
    print(f"   Dataset: {DATASET_REPO}")
    print(f"   Test Features: {TEST_FEATURES}")
    print(f"   Test Prompts: {len(TEST_PROMPTS)}")
    print()
    
    # Test 1: Model Loading
    print("🔬 Test 1: Loading model...")
    try:
        model = HookedSAETransformer.from_pretrained(BASE_MODEL, device=DEVICE)
        print(f"   ✅ Model loaded on {model.cfg.device}")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    print()
    
    # Test 2: SAE Loading
    print("🔬 Test 2: Loading SAE...")
    try:
        sae = load_local_sae(LARGE_SAE_MODEL, LAYER, DEVICE)
        print(f"   ✅ SAE loaded")
        print(f"   ✅ SAE device: {sae.cfg.device if hasattr(sae.cfg, 'device') else 'unknown'}")
        print(f"   ✅ Hook name: {getattr(sae.cfg, 'hook_name', 'N/A')}")
        print(f"   ✅ Hook layer: {getattr(sae.cfg, 'hook_layer', 'N/A')}")
    except Exception as e:
        print(f"   ❌ SAE loading failed: {e}")
        return False
    print()
    
    # Test 3: ActivationsStore Creation
    print("🔬 Test 3: Creating ActivationsStore...")
    try:
        from sae_lens import ActivationsStore
        
        # Use provided dataset
        activation_store = ActivationsStore.from_sae(
            model=model,
            sae=sae,
            dataset=DATASET_REPO,
            streaming=True,
            store_batch_size_prompts=1,
            train_batch_size_tokens=512,
            n_batches_in_buffer=2,
            device=DEVICE,
        )
        print(f"   ✅ ActivationsStore created")
    except Exception as e:
        print(f"   ❌ ActivationsStore creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 4: Finding Max Activation
    print("🔬 Test 4: Finding max activation...")
    try:
        from sae_pipeline.steering import find_max_activation
        
        # Test with first feature
        max_act = find_max_activation(model, sae, activation_store, TEST_FEATURES[0], layer=LAYER, num_batches=1)
        print(f"   ✅ Max activation found: {max_act:.4f}")
    except Exception as e:
        print(f"   ❌ Finding max activation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 5: Full Steering Experiment (minimal)
    print("🔬 Test 5: Running minimal steering experiment...")
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Format features for steering
        top_features_per_layer = {
            LAYER: [(fid, 1.0) for fid in TEST_FEATURES]
        }
        
        run_steering_experiment(
            model=model,
            prompts=TEST_PROMPTS,
            top_features_per_layer=top_features_per_layer,
            layers=[LAYER],
            output_folder=OUTPUT_FOLDER,
            device=DEVICE,
            sae_path=LARGE_SAE_MODEL,
            dataset=DATASET_REPO
        )
        print(f"   ✅ Steering experiment completed!")
        print(f"   📁 Output folder: {OUTPUT_FOLDER}/")
        
        # Check if output files were created
        output_files = list(Path(OUTPUT_FOLDER).glob("*.json"))
        print(f"   📄 Created {len(output_files)} output files")
        
    except Exception as e:
        print(f"   ❌ Steering experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Test 6: vLLM Server Check
    print("🔬 Test 6: Checking vLLM server...")
    try:
        import requests
        api_url = "http://localhost:8002/v1/models"
        response = requests.get(api_url, timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"   ✅ vLLM server is running")
            print(f"   ✅ Available models: {[m.get('id', 'N/A') for m in models.get('data', [])]}")
        else:
            print(f"   ⚠️  vLLM server returned status {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Could not connect to vLLM server: {e}")
        print(f"   ⚠️  This is okay if you're only testing steering (not interpretation)")
    print()
    
    print("="*70)
    print("✅ All tests passed! The pipeline should work correctly.")
    print("="*70)
    return True

if __name__ == "__main__":
    # Set environment variables (should match the bash script)
    os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"
    
    success = test_pipeline()
    sys.exit(0 if success else 1)

