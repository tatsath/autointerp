#!/usr/bin/env python3
"""Label reasoning features using vLLM API based on activating and non-activating sentences"""

import json
import math
import requests
from typing import List, Dict, Tuple

############################
# CONFIG
############################

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default paths - can be overridden via environment variables
FEATURE_LIST_JSON = os.getenv("FEATURE_LIST_JSON", os.path.join(BASE_DIR, "results", "1_search", "feature_list.json"))
ACTIVATING_CONTEXTS_JSON = os.getenv("ACTIVATING_CONTEXTS_JSON", os.path.join(BASE_DIR, "results", "2_labeling_lite", "activating_sentences.json"))
OUTPUT_JSON = os.getenv("OUTPUT_JSON", os.path.join(BASE_DIR, "results", "2_labeling_lite", "feature_labels.json"))

# vLLM API config (from run_llama_features_10.sh)
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"

# Labeling parameters
TOP_K_POS = 20  # Use top K positive snippets
TOP_K_NEG = 20  # Use top K negative snippets

############################
# REASONING VOCAB
############################

# Reasoning-related keywords for coverage calculation
REASONING_KEYWORDS = {
    "therefore", "however", "because", "since", "implies", "follows",
    "thus", "hence", "consequently", "alternatively", "moreover",
    "furthermore", "nevertheless", "nonetheless", "accordingly",
    "reasoning", "inference", "deduction", "conclusion", "premise",
    "logical", "causal", "contradiction", "hypothesis", "evidence"
}

def is_reasoning(text: str) -> bool:
    """Check if text contains any reasoning-related keywords"""
    lower = text.lower()
    return any(keyword in lower for keyword in REASONING_KEYWORDS)

############################
# TEXT CLEANING
############################

def clean_snippet(text: str) -> str:
    """Remove dataset prefixes and clean up text snippets"""
    import re
    # Remove common prefixes from reasoning datasets
    text = re.sub(r"^This is a reasoning example[^:]*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Continued from[^:]*:\s*", "", text)
    return text.strip()

############################
# vLLM API CALL
############################

def call_vllm_api(prompt: str, model: str, api_base: str) -> str:
    """Call vLLM API (OpenAI-compatible format)"""
    url = f"{api_base}/chat/completions"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling vLLM API: {e}")
        return "LABEL: unknown\nEXPLANATION: Error generating label."

############################
# PROMPT BUILDING
############################

def build_prompt(pos_contexts: List[str], neg_contexts: List[str]) -> str:
    """Build prompt for labeling using token-level context windows"""
    # Context windows are already cleaned (they're token windows, not full sentences)
    pos_text = "\n".join(f"- {s[:200]}..." if len(s) > 200 else f"- {s}" for s in pos_contexts[:TOP_K_POS])
    
    if neg_contexts:
        neg_text = "\n".join(f"- {s[:200]}..." if len(s) > 200 else f"- {s}" for s in neg_contexts[:TOP_K_NEG])
    else:
        neg_text = "(No negative examples provided)"
    
    prompt = f"""You are labeling a sparse autoencoder (SAE) feature of a REASONING language model.

You are given context windows where the feature is ACTIVE:

POSITIVE_CONTEXTS:
{pos_text}

And context windows where the feature is INACTIVE:

NEGATIVE_CONTEXTS:
{neg_text}

Your tasks:

1) Identify the MAIN BROAD REASONING CONCEPT represented in the POSITIVE_CONTEXTS.
   Examples: logical connectors (therefore, because, since), causal relationships,
   inference patterns, step-by-step reasoning, contradiction detection, hypothesis
   formation, evidence evaluation, deductive reasoning, inductive reasoning,
   analogical reasoning, counterfactual reasoning, etc.

2) Detect any SPECIFIC REFERENCE that appears MULTIPLE times in the POSITIVE_CONTEXTS
   and is largely absent in the NEGATIVE_CONTEXTS. This can be:
   - a specific logical pattern or reasoning structure,
   - a particular type of inference (e.g., modus ponens, reductio ad absurdum),
   - a domain-specific reasoning concept,
   - a specific logical operator or connective pattern.
   Use it ONLY IF:
   - it appears at least 2–3 times in the positive contexts.

3) Produce a CONCISE LABEL that:
   - Starts with the broad reasoning concept.
   - If a repeated specific reference exists, append it at the END of the label
     in the format: ", specific reference: <Name>".
     Examples:
       "Causal reasoning patterns, specific reference: modus ponens"
       "Logical connectors and inference, specific reference: counterfactuals"
       "Step-by-step reasoning, specific reference: mathematical proofs"
   - If NO such repeated reference exists, DO NOT mention any reference or
     placeholder in the label.
   - Prefer labels with at most 10 words.

4) Assign a GRANULARITY category, choosing EXACTLY ONE from:
   ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL

   Use these guidelines:
   - ENTITY: dominated by a specific named reasoning pattern, logical structure,
     or formal system (e.g., specific theorem, logical rule, reasoning framework).
   - SECTOR: focused on broad reasoning categories or domains (e.g., mathematical
     reasoning, scientific reasoning, ethical reasoning, causal reasoning).
   - EVENT: focused on discrete reasoning events or steps (e.g., making an inference,
     drawing a conclusion, identifying a contradiction, forming a hypothesis).
   - MACRO: focused on high-level reasoning processes (e.g., problem-solving strategies,
     decision-making processes, abstract reasoning patterns).
   - STRUCTURAL: focused on stable reasoning structures (e.g., logical operators,
     inference rules, reasoning templates, argument structures).
   - LEXICAL: keyed mainly to particular recurring phrases/wording rather than
     a clear reasoning concept (e.g., specific connector words, phrasing patterns).

5) Provide ONE sentence explaining what reasoning pattern the feature detects.

Return output EXACTLY as:

LABEL: <label>
GRANULARITY: <one of ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL>
EXPLANATION: <one sentence explanation>
"""
    
    return prompt.strip()

def parse_label_output(text: str) -> Tuple[str, str, str]:
    """Parse LLM output to extract label, granularity, and explanation"""
    label = "unknown"
    granularity = "STRUCTURAL"  # Default
    explanation = text.strip()
    
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("LABEL:"):
            label = line.replace("LABEL:", "").strip()
        elif line.startswith("GRANULARITY:"):
            granularity = line.replace("GRANULARITY:", "").strip()
            # Validate granularity
            valid_granularities = ["ENTITY", "SECTOR", "EVENT", "MACRO", "STRUCTURAL", "LEXICAL"]
            if granularity not in valid_granularities:
                granularity = "STRUCTURAL"  # Default if invalid
        elif line.startswith("EXPLANATION:"):
            explanation = line.replace("EXPLANATION:", "").strip()
    
    # Clean up label
    label = label.strip('"').strip("'").strip()
    if len(label) > 100:
        label = label[:100].rsplit(' ', 1)[0]
    
    return label, granularity, explanation

############################
# MAIN
############################

def main():
    print("\n" + "=" * 80)
    print("🏷️  Generating Reasoning Feature Labels")
    print("=" * 80)
    
    # Load feature list with reason scores
    print("\n📋 Step 1: Loading feature list...")
    print(f">>> Loading from: {FEATURE_LIST_JSON}")
    with open(FEATURE_LIST_JSON, "r", encoding="utf-8") as f:
        feature_data = json.load(f)
    
    feature_indices = feature_data["feature_indices"]
    reason_scores = feature_data["scores"]
    print(f">>> Found {len(feature_indices)} features")
    
    # Create mapping from feature_id to reason_score
    reason_score_map = {fid: score for fid, score in zip(feature_indices, reason_scores)}
    
    # Load activating context windows
    print("\n📋 Step 2: Loading activating context windows...")
    print(f">>> Loading from: {ACTIVATING_CONTEXTS_JSON}")
    with open(ACTIVATING_CONTEXTS_JSON, "r", encoding="utf-8") as f:
        activating_data = json.load(f)
    print(f">>> Loaded context windows for {len(activating_data)} features")
    
    # Normalize reason scores to [0, 1]
    min_reason = min(reason_scores)
    max_reason = max(reason_scores)
    eps = 1e-8
    
    print("\n📋 Step 3: Generating labels using LLM...")
    print(f">>> vLLM API: {EXPLAINER_API_BASE_URL}")
    print(f">>> Model: {EXPLAINER_MODEL}")
    print(f">>> Processing {len(feature_indices)} features...")
    print(">>> This may take several minutes...")
    print()
    
    results = []
    
    for idx, fid in enumerate(feature_indices):
        print(f"[{idx+1}/{len(feature_indices)}] Processing feature {fid}...")
        
        # Get reason score
        rs = reason_score_map[fid]
        reason_norm = (rs - min_reason) / (max_reason - min_reason + eps)
        
        # Get pos/neg snippets
        if str(fid) not in activating_data:
            print(f"  ⚠️  No data found for feature {fid}, skipping...")
            continue
        
        feat_data = activating_data[str(fid)]
        
        # Handle both old format (top_sentences) and new format (pos_contexts/neg_contexts or pos_snippets/neg_snippets)
        if "pos_contexts" in feat_data and "neg_contexts" in feat_data:
            # New format: token-level context windows
            pos_contexts = feat_data["pos_contexts"]
            neg_contexts = feat_data["neg_contexts"]
        elif "pos_snippets" in feat_data and "neg_snippets" in feat_data:
            # Intermediate format: still using snippets name
            pos_contexts = feat_data["pos_snippets"]
            neg_contexts = feat_data["neg_snippets"]
        elif "top_sentences" in feat_data:
            # Old format - convert to new format
            pos_contexts = [s["sentence"] for s in feat_data["top_sentences"]]
            neg_contexts = []  # No negative examples in old format
        else:
            print(f"  ⚠️  Unknown data format for feature {fid}, skipping...")
            continue
        
        if not pos_contexts:
            print(f"  ⚠️  No positive context windows for feature {fid}, skipping...")
            continue
        
        # Context windows are already token-level, no need to clean further
        # Build prompt and call LLM
        prompt = build_prompt(pos_contexts, neg_contexts)
        
        # Show prompt for first feature for debugging
        if idx == 0:
            print(f"\n--- PROMPT FOR FEATURE {fid} (first 500 chars) ---")
            print(prompt[:500] + "...")
            print("--- END PROMPT PREVIEW ---\n")
        raw_out = call_vllm_api(prompt, EXPLAINER_MODEL, EXPLAINER_API_BASE_URL)
        label, granularity, explanation = parse_label_output(raw_out)
        
        # Compute coverage: fraction of pos_contexts containing reasoning keywords
        num_reasoning = sum(is_reasoning(s) for s in pos_contexts)
        coverage = num_reasoning / len(pos_contexts) if pos_contexts else 0.0
        
        # Compute label_score
        label_score = 0.5 * reason_norm + 0.5 * coverage
        
        # Store result (context windows are already token-level)
        result = {
            "feature_id": fid,
            "reason_score": rs,
            "reason_norm": reason_norm,
            "coverage": coverage,
            "label_score": label_score,
            "label": label,
            "granularity": granularity,
            "explanation": explanation,
            "pos_contexts": pos_contexts[:TOP_K_POS],  # Token-level context windows
            "neg_contexts": neg_contexts[:TOP_K_NEG] if neg_contexts else []  # Token-level context windows
        }
        
        results.append(result)
        
        print(f"  ✅ Label: {label!r}")
        print(f"     Granularity: {granularity}")
        print(f"     reason_norm={reason_norm:.3f}, coverage={coverage:.3f}, label_score={label_score:.3f}")
        print()
    
    # Save results
    output_data = {
        "features": results,
        "num_features": len(results),
        "model": EXPLAINER_MODEL,
        "api_base": EXPLAINER_API_BASE_URL
    }
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Saved labeled features to {OUTPUT_JSON}")
    print(f"   Total features labeled: {len(results)}")

if __name__ == "__main__":
    main()



