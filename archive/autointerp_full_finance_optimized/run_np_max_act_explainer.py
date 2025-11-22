#!/usr/bin/env python3
"""
Standalone script to run Neuronpedia-style max-activation explainer.

This script uses cached activations from Delphi and generates labels using
the Neuronpedia max-activation approach without modifying Delphi code.
"""

import argparse
import asyncio
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from autointerp_full.clients.vllm import VLLMClient
from autointerp_full.config import ConstructorConfig, SamplerConfig
from autointerp_full.explainers.np_max_act_explainer import NPMaxActExplainer
from autointerp_full.latents.loader import LatentDataset


async def generate_labels(
    latents_path: str,
    explanations_path: str,
    hookpoints: list[str],
    feature_nums: list[int] | None,
    explainer_model: str,
    explainer_api_base_url: str,
    tokenizer,
    k_max_act: int = 24,
    window: int = 12,
    verbose: bool = False,
):
    """
    Generate labels using Neuronpedia max-activation explainer.

    Args:
        latents_path: Path to cached latent activations.
        explanations_path: Path to save generated explanations.
        hookpoints: List of hookpoints to process.
        feature_nums: List of feature indices to process (None for all).
        explainer_model: Model name for explainer LLM.
        explainer_api_base_url: API base URL for explainer.
        tokenizer: Tokenizer instance.
        k_max_act: Number of top max-activation examples to use.
        window: Context window size around max-act token.
        verbose: Whether to print verbose output.
    """
    explanations_path = Path(explanations_path)
    explanations_path.mkdir(parents=True, exist_ok=True)

    # Create latent dataset from cached activations
    latent_dict = None
    if feature_nums:
        latent_dict = {hook: torch.tensor(feature_nums) for hook in hookpoints}

    sampler_cfg = SamplerConfig()
    constructor_cfg = ConstructorConfig()

    dataset = LatentDataset(
        raw_dir=latents_path,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        modules=hookpoints,
        latents=latent_dict,
        tokenizer=tokenizer,
    )

    # Create explainer client
    client = VLLMClient(
        model=explainer_model,
        base_url=explainer_api_base_url,
    )

    # Create Neuronpedia explainer
    explainer = NPMaxActExplainer(
        client=client,
        tokenizer=tokenizer,
        k_max_act=k_max_act,
        window=window,
        use_logits=False,
        verbose=verbose,
    )

    # Process each latent using async iteration
    print(f"Processing latents...")

    idx = 0
    async for record in dataset:
        idx += 1
        print(f"\n[{idx}] Processing {record.latent}...")

        try:
            # Generate explanation
            result = await explainer(record)

            # Save explanation
            explanation_file = explanations_path / f"{record.latent}.txt"
            with open(explanation_file, "w") as f:
                f.write(result.explanation)

            if verbose:
                print(f"  ✓ Generated: {result.explanation[:100]}...")
            else:
                print(f"  ✓ Generated explanation")

        except Exception as e:
            print(f"  ✗ Error: {repr(e)}")
            # Save error message
            explanation_file = explanations_path / f"{record.latent}.txt"
            with open(explanation_file, "w") as f:
                f.write(f"Error: {repr(e)}")

    print(f"\n✅ Completed! Explanations saved to: {explanations_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Neuronpedia-style max-activation explainer on cached activations"
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        required=True,
        help="Path to cached latent activations directory",
    )
    parser.add_argument(
        "--explanations_path",
        type=str,
        required=True,
        help="Path to save generated explanations",
    )
    parser.add_argument(
        "--hookpoints",
        type=str,
        nargs="+",
        required=True,
        help="Hookpoints to process (e.g., layers.28)",
    )
    parser.add_argument(
        "--feature_num",
        type=int,
        nargs="+",
        default=None,
        help="Feature indices to process (default: all)",
    )
    parser.add_argument(
        "--explainer_model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Model name for explainer LLM",
    )
    parser.add_argument(
        "--explainer_api_base_url",
        type=str,
        default="http://localhost:8002/v1",
        help="API base URL for explainer",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Base model name (for tokenizer)",
    )
    parser.add_argument(
        "--k_max_act",
        type=int,
        default=24,
        help="Number of top max-activation examples to use (default: 24)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Context window size around max-act token (default: 12)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token for private models",
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)

    # Run async function
    asyncio.run(
        generate_labels(
            latents_path=args.latents_path,
            explanations_path=args.explanations_path,
            hookpoints=args.hookpoints,
            feature_nums=args.feature_num,
            explainer_model=args.explainer_model,
            explainer_api_base_url=args.explainer_api_base_url,
            tokenizer=tokenizer,
            k_max_act=args.k_max_act,
            window=args.window,
            verbose=args.verbose,
        )
    )


if __name__ == "__main__":
    main()

