#!/usr/bin/env python3
"""
Command-line script for interpreting SAE features using LLM analysis of steering outputs.

Usage examples:
    # Using OpenRouter (GPT-4o)
    python scripts/run_interpretation.py \
        --steering_outputs steering_outputs \
        --output interpretations.json \
        --explainer_provider openrouter \
        --explainer_model openai/gpt-4o \
        --api_key YOUR_KEY
    
    # Using vLLM
    python scripts/run_interpretation.py \
        --steering_outputs steering_outputs \
        --output interpretations.json \
        --explainer_provider vllm \
        --explainer_model Qwen/Qwen2.5-7B-Instruct \
        --explainer_api_base_url http://localhost:8002/v1
    
    # Using offline transformers
    python scripts/run_interpretation.py \
        --steering_outputs steering_outputs \
        --output interpretations.json \
        --explainer_provider offline \
        --explainer_model meta-llama/Llama-2-7b-chat-hf
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae_pipeline.feature_interpreter import (
    load_steering_outputs,
    interpret_all_features
)

# Import clients - add autointerp_full to path
_autointerp_full_path = Path(__file__).parent.parent.parent / "autointerp_full"
if _autointerp_full_path.exists():
    sys.path.insert(0, str(_autointerp_full_path))
else:
    # Try alternative path
    _alt_path = Path(__file__).parent.parent.parent.parent / "autointerp_full"
    if _alt_path.exists():
        sys.path.insert(0, str(_alt_path))

try:
    from autointerp_full.clients import OpenRouter, VLLMClient, TransformersClient, TransformersFastClient
    try:
        from autointerp_full.clients import Offline
        OFFLINE_AVAILABLE = True
    except ImportError:
        OFFLINE_AVAILABLE = False
except ImportError as e:
    print("Error: Could not import LLM clients from autointerp_full")
    print(f"Tried path: {_autointerp_full_path}")
    print(f"Make sure autointerp_full is accessible from autointerp_steer")
    print(f"Import error: {e}")
    sys.exit(1)


def create_client(args):
    """Create LLM client based on provider."""
    provider = args.explainer_provider.lower()
    
    if provider == "openrouter":
        if not args.api_key:
            raise ValueError("--api_key required for OpenRouter provider")
        return OpenRouter(
            model=args.explainer_model,
            api_key=args.api_key,
            max_tokens=2000,
            temperature=0.7
        )
    
    elif provider == "vllm":
        from autointerp_full.clients.vllm import VLLMClient
        api_base = args.explainer_api_base_url or "http://localhost:8000/v1"
        return VLLMClient(
            model=args.explainer_model,
            base_url=api_base,  # Changed from api_base_url to base_url
            max_tokens=500,  # Reduced from 2000 to avoid context length issues
            temperature=0.7
        )
    
    elif provider == "offline":
        if not OFFLINE_AVAILABLE:
            raise ImportError("Offline client not available. Install dependencies.")
        return Offline(
            model=args.explainer_model,
            max_tokens=2000,
            temperature=0.7
        )
    
    elif provider == "transformers":
        return TransformersClient(
            model=args.explainer_model,
            max_tokens=2000,
            temperature=0.7
        )
    
    elif provider == "transformers_fast":
        return TransformersFastClient(
            model=args.explainer_model,
            max_tokens=2000,
            temperature=0.7
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            "Supported: openrouter, vllm, offline, transformers, transformers_fast"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Interpret SAE features using LLM analysis of steering outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--steering_outputs',
        type=str,
        required=True,
        help='Directory containing steering output JSON files (from run_steering.py)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSON file path for interpretations'
    )
    
    # LLM provider arguments
    parser.add_argument(
        '--explainer_provider',
        type=str,
        default='openrouter',
        choices=['openrouter', 'vllm', 'offline', 'transformers', 'transformers_fast'],
        help='LLM provider to use for interpretation (default: openrouter)'
    )
    parser.add_argument(
        '--explainer_model',
        type=str,
        default='openai/gpt-4o',
        help='Model name for interpretation (default: openai/gpt-4o)'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=None,
        help='API key (required for openrouter, optional for others)'
    )
    parser.add_argument(
        '--explainer_api_base_url',
        type=str,
        default=None,
        help='API base URL (for vllm provider, default: http://localhost:8000/v1)'
    )
    
    # Optional filtering
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Specific layers to interpret (default: all layers in steering outputs)'
    )
    
    args = parser.parse_args()
    
    # Get API key from environment if not provided
    if args.explainer_provider == 'openrouter' and not args.api_key:
        args.api_key = os.getenv('OPENROUTER_API_KEY')
        if not args.api_key:
            print("Warning: No API key provided. Set --api_key or OPENROUTER_API_KEY env var.")
    
    # Load steering outputs
    print(f"Loading steering outputs from {args.steering_outputs}...")
    try:
        steering_outputs = load_steering_outputs(args.steering_outputs)
        print(f"✓ Loaded steering outputs for {len(steering_outputs)} layer(s)")
    except Exception as e:
        print(f"Error loading steering outputs: {e}")
        sys.exit(1)
    
    # Create LLM client
    print(f"\nInitializing {args.explainer_provider} client with model {args.explainer_model}...")
    try:
        client = create_client(args)
        print("✓ Client initialized")
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)
    
    # Run interpretation
    print(f"\nStarting feature interpretation...")
    print(f"Output will be saved to: {args.output}")
    
    try:
        results = asyncio.run(interpret_all_features(
            client,
            steering_outputs,
            args.output,
            layers=args.layers
        ))
        
        # Print summary
        total_features = sum(len(layer_data) for layer_data in results.values())
        successful = sum(
            sum(1 for feat_data in layer_data.values() if feat_data.get('status') == 'success')
            for layer_data in results.values()
        )
        
        print(f"\n{'='*60}")
        print(f"Interpretation complete!")
        print(f"Total features processed: {total_features}")
        print(f"Successful: {successful}")
        print(f"Failed: {total_features - successful}")
        print(f"Results saved to: {args.output}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during interpretation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

