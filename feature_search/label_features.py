#!/usr/bin/env python3
"""Label features using vLLM API based on activating and non-activating sentences"""

import json
import math
import requests
from typing import List, Dict, Tuple

############################
# CONFIG
############################

FEATURE_LIST_JSON = "test_results/feature_list.json"
ACTIVATING_CONTEXTS_JSON = "test_results/activating_sentences.json"  # Contains context windows
FINANCE_VOCAB_FILE = "finance_vocab.txt"
OUTPUT_JSON = "test_results/feature_labels.json"

# vLLM API config (from run_llama_features_10.sh)
EXPLAINER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
EXPLAINER_API_BASE_URL = "http://localhost:8002/v1"

# Labeling parameters
TOP_K_POS = 20  # Use top K positive snippets
TOP_K_NEG = 20  # Use top K negative snippets

############################
# LOAD FINANCE VOCAB
############################

def load_finance_vocab(vocab_file: str) -> set:
    """Load finance vocabulary words"""
    vocab_words = set()
    try:
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip()
                if word:
                    vocab_words.add(word.lower())
    except FileNotFoundError:
        print(f"Warning: {vocab_file} not found. Coverage will be 0.")
    return vocab_words

vocab_words_set = load_finance_vocab(FINANCE_VOCAB_FILE)

def is_financey(text: str) -> bool:
    """Check if text contains any finance vocabulary word"""
    lower = text.lower()
    return any(w in lower for w in vocab_words_set)

############################
# TEXT CLEANING
############################

def clean_snippet(text: str) -> str:
    """Remove dataset prefixes and clean up text snippets"""
    import re
    # Remove "This is a news article titled..." prefix
    text = re.sub(r"^This is a news article titled '[^']+', published on \d+ \w+ \d{4}\. It covers the following details:\s*", "", text)
    # Remove "Continued from the article titled..." prefix
    text = re.sub(r"^Continued from the article titled '[^']+':\s*", "", text)
    # Remove any remaining "This is a news article..." patterns
    text = re.sub(r"^This is a news article[^:]*:\s*", "", text, flags=re.IGNORECASE)
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
    
    prompt = f"""You are labeling a sparse autoencoder (SAE) feature of a FINANCIAL language model.

You are given context windows where the feature is ACTIVE:

POSITIVE_CONTEXTS:
{pos_text}

And context windows where the feature is INACTIVE:

NEGATIVE_CONTEXTS:
{neg_text}

Your tasks:

1) Identify the MAIN BROAD FINANCIAL CONCEPT represented in the POSITIVE_CONTEXTS.
   Examples: valuation ratios, dividend yields, liquidity, post-earnings reactions,
   IPOs, profit beats, analyst sentiment, sector themes, credit spreads, interest
   rates, Fed commentary, macro headlines, specific events, etc.

2) Detect any SPECIFIC REFERENCE that appears MULTIPLE times in the POSITIVE_CONTEXTS
   and is largely absent in the NEGATIVE_CONTEXTS. This can be:
   - a company or ticker (e.g. Meta, TSLA, ADBE),
   - an index (e.g. S&P 500),
   - a person (e.g. Jerome Powell),
   - an institution, sector, country, or location.
   Use it ONLY IF:
   - it appears at least 2–3 times in the positive contexts.

3) Produce a CONCISE LABEL that:
   - Starts with the broad concept.
   - If a repeated specific reference exists, append it at the END of the label
     in the format: ", specific reference: <Name>".
     Examples:
       "Dividend yield comparisons, specific reference: S&P 500"
       "Liquidity and valuation signals, specific reference: Meta"
       "Post-earnings reactions, specific reference: BNY Mellon"
   - If NO such repeated reference exists, DO NOT mention any reference or
     placeholder in the label.
   - Prefer labels with at most 10 words.

4) Assign a GRANULARITY category, choosing EXACTLY ONE from:
   ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL

   Use these guidelines:
   - ENTITY: dominated by a specific company, person, index or named institution.
   - SECTOR: focused on sectors, industries, asset classes, or broad themes
     (e.g. AI and big tech, energy stocks).
   - EVENT: focused on discrete events (earnings releases, IPOs, splits,
     profit warnings, M&A, big moves).
   - MACRO: focused on rates, inflation, Fed/central banks, GDP, broad markets.
   - STRUCTURAL: focused on stable financial notions like ratios (P/E, PEG),
     valuations, liquidity levels, balance-sheet/metric structure.
   - LEXICAL: keyed mainly to particular recurring phrases/wording rather than
     a clear financial concept.

5) Provide ONE sentence explaining what pattern the feature detects.

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
    # Load feature list with reason scores
    print(f"Loading feature list from {FEATURE_LIST_JSON}...")
    with open(FEATURE_LIST_JSON, "r", encoding="utf-8") as f:
        feature_data = json.load(f)
    
    feature_indices = feature_data["feature_indices"]
    reason_scores = feature_data["scores"]
    
    # Create mapping from feature_id to reason_score
    reason_score_map = {fid: score for fid, score in zip(feature_indices, reason_scores)}
    
    # Load activating context windows
    print(f"Loading activating context windows from {ACTIVATING_CONTEXTS_JSON}...")
    with open(ACTIVATING_CONTEXTS_JSON, "r", encoding="utf-8") as f:
        activating_data = json.load(f)
    
    # Normalize reason scores to [0, 1]
    min_reason = min(reason_scores)
    max_reason = max(reason_scores)
    eps = 1e-8
    
    print(f"Processing {len(feature_indices)} features...")
    print(f"vLLM API: {EXPLAINER_API_BASE_URL}")
    print(f"Model: {EXPLAINER_MODEL}")
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
        
        # Compute coverage: fraction of pos_contexts containing finance vocab
        num_financey = sum(is_financey(s) for s in pos_contexts)
        coverage = num_financey / len(pos_contexts) if pos_contexts else 0.0
        
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

