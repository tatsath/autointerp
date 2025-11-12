#!/usr/bin/env python3
"""
Command-line script for interpreting SAE features using LLM analysis of steering outputs.

Usage examples:
    # Using vLLM HTTP API
    python scripts/run_interpretation.py \
        --steering_output_dir steering_outputs \
        --output_dir interpretation_outputs \
        --explainer_api_base http://127.0.0.1:8002/v1 \
        --explainer_model Qwen/Qwen2.5-72B-Instruct \
        --explainer_max_tokens 256 \
        --explainer_temperature 0.0 \
        --max_features 50
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_clients import VLLMHTTPClient, VLLMHTTPConfig
from sae_pipeline.feature_interpreter import (
    load_steering_outputs,
    interpret_all_features,
)


def main():
    parser = argparse.ArgumentParser(
        description="Interpret SAE features using LLM analysis of steering outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--steering_output_dir',
        type=str,
        required=True,
        help='Directory containing steering output JSON files (from run_steering.py)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where to write interpretation files'
    )
    
    # vLLM HTTP settings
    parser.add_argument(
        '--explainer_api_base',
        type=str,
        required=True,
        help='Base URL of vLLM OpenAI-compatible API, e.g. http://localhost:8002/v1'
    )
    parser.add_argument(
        '--explainer_model',
        type=str,
        required=True,
        help='Model name served by vLLM (string you pass to "model" in the API)'
    )
    parser.add_argument(
        '--explainer_api_key',
        type=str,
        default=None,
        help='Optional API key if your vLLM server checks auth'
    )
    parser.add_argument(
        '--explainer_max_tokens',
        type=int,
        default=256,
        help='Maximum tokens for explanation generation'
    )
    parser.add_argument(
        '--explainer_temperature',
        type=float,
        default=0.0,
        help='Temperature for explanation generation'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='HTTP timeout in seconds per request'
    )
    
    # Optional filtering
    parser.add_argument(
        '--max_features',
        type=int,
        default=None,
        help='Optionally limit how many features to interpret'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=None,
        help='Specific layers to interpret (default: all layers in steering outputs)'
    )
    
    args = parser.parse_args()
    
    # Load steering outputs
    print(f"Loading steering outputs from {args.steering_output_dir}...")
    try:
        steering_outputs = load_steering_outputs(args.steering_output_dir)
        print(f"✓ Loaded steering outputs for {len(steering_outputs)} layer(s)")
    except Exception as e:
        print(f"Error loading steering outputs: {e}")
        sys.exit(1)
    
    # Create LLM client
    print(f"\nInitializing vLLM HTTP client...")
    print(f"  API Base: {args.explainer_api_base}")
    print(f"  Model: {args.explainer_model}")
    try:
        cfg = VLLMHTTPConfig(
            base_url=args.explainer_api_base,
            model=args.explainer_model,
            api_key=args.explainer_api_key,
            max_tokens=args.explainer_max_tokens,
            temperature=args.explainer_temperature,
            timeout=args.timeout,
        )
        client = VLLMHTTPClient(cfg)
        print("✓ Client initialized")
    except Exception as e:
        print(f"Error initializing client: {e}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "interpretations.json")
    
    # Run interpretation
    print(f"\nStarting feature interpretation...")
    print(f"Output will be saved to: {output_file}")
    if args.max_features:
        print(f"Limiting to {args.max_features} features per layer")
    
    try:
        results = asyncio.run(interpret_all_features(
            steering_outputs=steering_outputs,
            client=client,
            output_dir=args.output_dir,
            max_features=args.max_features,
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
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Error during interpretation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
