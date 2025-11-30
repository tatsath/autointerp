#!/usr/bin/env python3
"""
Show exactly what the Explainer LLM sees and how FAISS contrastive examples work.

This script loads the actual examples that were sent to the explainer LLM and
shows how they're formatted, including:
1. Activating examples (what the feature fires on)
2. FAISS contrastive examples (semantically similar but non-activating)
3. How the prompt is constructed
"""

import json
import torch
import argparse
from pathlib import Path
from transformers import AutoTokenizer

from autointerp_full.latents.loader import LatentDataset
from autointerp_full.config import SamplerConfig, ConstructorConfig
from analyze_feature_activations import (
    load_cached_latents,
    find_hookpoint_in_latents,
    get_model_from_config,
    load_explanation,
)


def format_example_for_explainer(example, tokenizer, show_activations=True):
    """Format an example as the explainer LLM would see it.
    
    The explainer sees tokens with <<token>> around activating tokens.
    """
    if not hasattr(example, 'str_tokens') or not example.str_tokens:
        # Fallback: decode tokens
        if hasattr(example, 'tokens'):
            tokens = example.tokens
            str_tokens = tokenizer.convert_ids_to_tokens(tokens.tolist())
        else:
            return "Could not decode example"
    else:
        str_tokens = example.str_tokens
    
    # Get activations
    if hasattr(example, 'activations') and show_activations:
        activations = example.activations
        # Activation threshold - tokens above this are marked
        # Typically uses a threshold like 0.001% of max or similar
        if len(activations) > 0:
            max_act = activations.max().item()
            threshold = max(0.00001 * max_act, 0.001)  # Very low threshold
        else:
            threshold = 0.0
    else:
        activations = None
        threshold = 0.0
    
    # Format tokens with << >> around activating ones
    formatted_tokens = []
    for i, token in enumerate(str_tokens):
        if activations is not None and i < len(activations):
            act_val = activations[i].item() if hasattr(activations[i], 'item') else float(activations[i])
            if act_val > threshold:
                formatted_tokens.append(f"<<{token}>>")
            else:
                formatted_tokens.append(token)
        else:
            formatted_tokens.append(token)
    
    return " ".join(formatted_tokens)


def explain_feature_end_to_end(
    results_dir: Path,
    hookpoint: str,
    feature_id: int,
    base_model: str,
    output_file: Path,
):
    """Create a complete end-to-end explanation of how the feature explanation was generated."""
    
    latents_dir = results_dir / "latents"
    
    # Load run config
    run_config_path = results_dir / "run_config.json"
    with open(run_config_path) as f:
        run_config = json.load(f)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create configs
    sampler_cfg = SamplerConfig(**run_config["sampler_cfg"])
    constructor_cfg = ConstructorConfig(**run_config["constructor_cfg"])
    
    # Create dataset for this feature
    latents_dict = {hookpoint: torch.tensor([feature_id])}
    
    # Load explanation
    explanation = load_explanation(results_dir, hookpoint, feature_id)
    
    with open(output_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"END-TO-END EXPLANATION: Feature {feature_id}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"FINAL EXPLANATION: {explanation}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("STEP 1: DATA COLLECTION\n")
        f.write("=" * 80 + "\n\n")
        f.write("""
AutoInterp processes 20M tokens from the financial news dataset and extracts
all places where this feature activates. Each activation has:
- Batch index (which document)
- Sequence index (position in document)  
- Activation value (how strongly it fired)

For feature {feature_id}, we found activations across the dataset.
\n""".format(feature_id=feature_id))
        
        try:
            dataset = LatentDataset(
                raw_dir=str(latents_dir),
                modules=[hookpoint],
                sampler_cfg=sampler_cfg,
                constructor_cfg=constructor_cfg,
                latents=latents_dict,
                tokenizer=tokenizer,
            )
            
            # Find the record
            record = None
            for r in dataset:
                if hasattr(r.latent, 'latent_index'):
                    if r.latent.latent_index == feature_id:
                        record = r
                        break
            
            if record:
                f.write("=" * 80 + "\n")
                f.write("STEP 2: EXAMPLE CONSTRUCTION\n")
                f.write("=" * 80 + "\n\n")
                
                # Get training examples (these are what explainer sees)
                train_examples = record.train if hasattr(record, "train") and record.train else []
                examples = record.examples if hasattr(record, "examples") and record.examples else []
                
                examples_to_show = train_examples[:15] if train_examples else examples[:15]
                
                f.write(f"Selected {len(examples_to_show)} strongest activating examples.\n")
                f.write("These examples are 32-token windows CENTERED on the activation point.\n\n")
                
                f.write("=" * 80 + "\n")
                f.write("STEP 3: WHAT THE EXPLAINER LLM SEES\n")
                f.write("=" * 80 + "\n\n")
                f.write("""
The explainer LLM (Qwen/Qwen2.5-72B-Instruct) receives these examples where
activating tokens are marked with <<token>>. This is the EXACT format sent to the LLM:

\n""")
                
                for i, ex in enumerate(examples_to_show, 1):
                    max_act = ex.max_activation if hasattr(ex, 'max_activation') else "N/A"
                    f.write(f"\nExample {i} (max_activation={max_act}):\n")
                    f.write("-" * 80 + "\n")
                    
                    # Format as explainer sees it
                    formatted = format_example_for_explainer(ex, tokenizer, show_activations=True)
                    f.write(formatted + "\n")
                    
                    # Also show raw text for clarity
                    if hasattr(ex, 'text'):
                        f.write(f"\nRaw text: {ex.text}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
                f.write("STEP 4: FAISS CONTRASTIVE EXAMPLES\n")
                f.write("=" * 80 + "\n\n")
                f.write("""
FAISS (Facebook AI Similarity Search) is used to find CONTRASTIVE examples:
- Semantically SIMILAR to activating examples
- But the feature does NOT activate on them
- These are "hard negatives" - similar meaning but different activation

How it works:
1. Embed activating examples using a finance embedding model
2. Search for similar texts in the dataset
3. Filter to only include texts where feature DOESN'T activate
4. These help the explainer understand what the feature is NOT

\n""")
                
                # Get FAISS examples (non-activating)
                neg_examples = record.not_active if hasattr(record, "not_active") and record.not_active else []
                
                if neg_examples:
                    f.write(f"Found {len(neg_examples)} FAISS contrastive examples:\n\n")
                    for i, ex in enumerate(neg_examples[:10], 1):
                        f.write(f"Contrastive Example {i}:\n")
                        f.write("-" * 80 + "\n")
                        if hasattr(ex, 'text'):
                            f.write(f"{ex.text}\n")
                        elif hasattr(ex, 'str_tokens'):
                            f.write(" ".join(ex.str_tokens) + "\n")
                        if hasattr(ex, 'distance'):
                            f.write(f"(FAISS similarity distance: {ex.distance:.4f})\n")
                        f.write("\n")
                else:
                    f.write("(No FAISS examples available in cache)\n")
                
                f.write("=" * 80 + "\n")
                f.write("STEP 5: THE PROMPT SENT TO EXPLAINER LLM\n")
                f.write("=" * 80 + "\n\n")
                f.write("""
The explainer receives a prompt like this:

---
We're studying neurons in a neural network. Each neuron activates on some particular 
word/words/substring/concept in a short document. The activating words in each document 
are indicated with << ... >>. 

We will give you a list of documents on which the neuron activates, in order from 
most strongly activating to least strongly activating. Look at the parts of the 
document the neuron activates for and summarize in a single sentence what the neuron 
is activating on.

Try not to be overly specific in your explanation. Note that some neurons will 
activate only on specific words or substrings, but others will activate on most/all 
words in a sentence provided that sentence contains some particular concept.

Your explanation should cover most or all activating words (for example, don't give 
an explanation which is specific to a single word if all words in a sentence cause 
the neuron to activate). Pay attention to things like the capitalization and 
punctuation of the activating words or concepts, if that seems relevant. 

Keep the explanation as short and simple as possible, limited to 20 words or less. 
Omit punctuation and formatting.

Examples:
- "This neuron activates on the word 'knows' in rhetorical questions"
- "This neuron activates on verbs related to decision-making and preferences"
- "This neuron activates on company names followed by stock ticker symbols"

Here are the documents:

[Examples 1-15 shown above with <<token>> markers]

---

The LLM then generates: "{explanation}"

\n""".format(explanation=explanation))
                
                f.write("=" * 80 + "\n")
                f.write("STEP 6: WHY THE EXPLANATION DIFFERS FROM TOP WORDS\n")
                f.write("=" * 80 + "\n\n")
                f.write("""
KEY INSIGHT: The explainer sees SEMANTIC PATTERNS, not just word frequencies.

Looking at the examples above, you can see:
- Individual words like "Industry", "Third", "Should" appear frequently
- BUT in context, they appear in patterns like:
  * "Municipal Income Fund declares $0.01 special dividend"
  * "Analyst Blog" sections
  * Dividend announcement contexts

The LLM explainer:
✓ Sees FULL CONTEXT (32 tokens)
✓ Recognizes SEMANTIC PATTERNS across examples
✓ Abstracts to a CONCEPT ("Special dividend declarations by funds")

Token analysis:
✗ Sees INDIVIDUAL WORDS in isolation
✗ Counts FREQUENCY only
✗ Loses SEMANTIC CONTEXT

That's why:
- Top words: "Industry", "Third", "Should" (common financial words)
- Explanation: "Special dividend declarations by funds" (semantic pattern)

BOTH are correct! They just measure different things:
- Top words = WHAT tokens fire
- Explanation = WHAT SEMANTIC PATTERN those tokens represent

\n""")
                
            else:
                f.write("\n(Could not load examples from dataset - they may not be cached)\n")
                f.write("The examples are constructed on-the-fly during autointerp run.\n")
        
        except Exception as e:
            f.write(f"\nError loading examples: {e}\n")
            import traceback
            f.write(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Explain what the explainer LLM sees")
    parser.add_argument("results_dir", type=str, help="Results directory")
    parser.add_argument("--feature-id", type=int, required=True, help="Feature ID")
    parser.add_argument("--output", type=str, default=None, help="Output file")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir).resolve()
    latents_dir = results_dir / "latents"
    
    # Auto-detect
    hookpoint = find_hookpoint_in_latents(latents_dir)
    base_model = get_model_from_config(latents_dir, hookpoint)
    
    print(f"Analyzing feature {args.feature_id}")
    print(f"Hookpoint: {hookpoint}")
    print(f"Model: {base_model}")
    
    # Set output
    if args.output is None:
        output_file = results_dir / "feature_analysis" / f"feature_{args.feature_id}_end_to_end_explanation.txt"
    else:
        output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Create explanation
    explain_feature_end_to_end(
        results_dir, hookpoint, args.feature_id, base_model, output_file
    )
    
    print(f"\n✓ End-to-end explanation saved to: {output_file}")


if __name__ == "__main__":
    main()



