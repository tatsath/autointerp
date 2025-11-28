#!/usr/bin/env python3
"""
Create a simple, clear explanation document for features 1532 and 18529.
"""

import json
from pathlib import Path
from explain_explainer_input import explain_feature_end_to_end
from analyze_feature_activations import (
    find_hookpoint_in_latents,
    get_model_from_config,
)


def main():
    results_dir = Path("results/nemotron_finance_news_run").resolve()
    latents_dir = results_dir / "latents"
    
    hookpoint = find_hookpoint_in_latents(latents_dir)
    base_model = get_model_from_config(latents_dir, hookpoint)
    
    # Create explanations for both features
    for feature_id in [1532, 18529]:
        output_file = results_dir / "feature_analysis" / f"feature_{feature_id}_end_to_end_explanation.txt"
        explain_feature_end_to_end(
            results_dir, hookpoint, feature_id, base_model, output_file
        )
        print(f"✓ Created explanation for feature {feature_id}")
    
    # Create a combined simple summary
    summary_file = results_dir / "feature_analysis" / "SIMPLE_EXPLANATION_1532_18529.txt"
    
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SIMPLE EXPLANATION: How Feature Explanations Are Generated\n")
        f.write("Features 1532 and 18529\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
This document explains in simple terms how AutoInterp generates feature explanations
and why they might differ from top activating words.

================================================================================
THE PROCESS (Step by Step)
================================================================================

STEP 1: Collect All Activations
--------------------------------
- Process 20M tokens from financial news
- Find every place where the feature fires
- Record: which document, which position, how strongly

STEP 2: Create Example Windows
-------------------------------
- Take the 15-25 STRONGEST activations
- For each, create a 32-token window CENTERED on the activation
- This gives context around where the feature fired

STEP 3: Mark Activating Tokens
-------------------------------
- In each example, mark tokens that activated with <<token>>
- Non-activating tokens shown normally
- This shows the LLM exactly which parts fired

STEP 4: Find Contrastive Examples (FAISS)
------------------------------------------
- Use FAISS (semantic similarity search) to find similar texts
- BUT filter to only include texts where feature DOESN'T activate
- These are "hard negatives" - similar meaning, different activation
- Helps the explainer understand what the feature is NOT

STEP 5: Send to Explainer LLM
------------------------------
- Send the marked examples to Qwen/Qwen2.5-72B-Instruct
- Ask: "What semantic pattern do these activating tokens represent?"
- LLM analyzes the FULL CONTEXT and generates explanation

STEP 6: Get Explanation
-----------------------
- LLM returns a short phrase describing the semantic pattern
- This is the "explanation" you see

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("FEATURE 1532: 'Special dividend declarations by funds'\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
WHAT THE EXPLAINER SAW:
- Examples like: "<<Municipal>> <<Income>> <<Fund>> <<declares>> $0.01 <<special>> <<dividend>>"
- Pattern: Fund names + "declares" + "special dividend" + amounts
- Context: These appear in "Analyst Blog" sections

WHY IT'S NOT JUST "Industry" or "Third":
- Those words appear in MANY contexts (not just dividends)
- The explainer sees the FULL PATTERN: Fund + declares + special dividend
- It abstracts to the SEMANTIC CONCEPT, not individual words

FAISS CONTRASTIVE:
- Finds similar financial texts (about funds, dividends, etc.)
- But where feature DOESN'T activate
- Helps explainer distinguish: "special dividend declarations" vs other dividend news

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("FEATURE 18529: 'Smart Beta Equity ETPs globally increased'\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
WHAT THE EXPLAINER SAW:
- Examples like: "<<Assets>> <<In>> <<Smart>> <<Beta>> <<Equity>> <<ETP>>s <<Globally>> <<Increased>>"
- Pattern: Smart Beta + Equity + ETP/ETF + "increased"/"grew"
- Context: Investment product announcements

WHY IT'S NOT JUST INDIVIDUAL WORDS:
- Words like "Equity", "ETP" appear in many contexts
- The explainer sees the COMBINATION: Smart Beta + Equity ETPs + growth language
- It recognizes the specific financial product category

FAISS CONTRASTIVE:
- Finds similar investment/ETF texts
- But where feature DOESN'T activate
- Helps explainer distinguish: "Smart Beta Equity ETPs" vs other ETF news

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
1. EXPLANATIONS ARE SEMANTIC, NOT STATISTICAL
   ✓ They interpret MEANING in context
   ✓ They see PATTERNS across examples
   ✓ They abstract to CONCEPTS

2. TOP WORDS ARE STATISTICAL, NOT SEMANTIC
   ✓ They count FREQUENCY
   ✓ They lose CONTEXT
   ✓ They show WHAT fires, not WHAT IT MEANS

3. FAISS HELPS WITH CONTRAST
   ✓ Finds similar texts (semantic similarity)
   ✓ But where feature DOESN'T activate
   ✓ Helps explainer understand boundaries

4. BOTH ARE CORRECT AND COMPLEMENTARY
   ✓ Top words: "What tokens trigger the feature?"
   ✓ Explanation: "What semantic pattern do they represent?"
   ✓ Use both for complete understanding

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("WHY THE DISCREPANCY?\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
Example: Feature 1532

Top Words Show: "Industry", "Third", "Should", "Sector"
- These are COMMON words in financial texts
- They appear frequently when feature fires
- BUT they appear in MANY other contexts too

Explanation Says: "Special dividend declarations by funds"
- This captures the SEMANTIC PATTERN
- When you see the FULL CONTEXT, you see:
  * "Municipal Income Fund declares $0.01 special dividend"
  * "Analyst Blog" sections
  * Fund + declares + special dividend pattern

The LLM explainer:
- Sees 15-25 FULL examples with context
- Recognizes the PATTERN across examples
- Abstracts to the CONCEPT

It's like the difference between:
- Counting letters: "e" appears most often
- Reading words: "the" is the most common word
- Understanding meaning: "financial news about dividends"

ALL THREE are true, but they measure different things!

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("HOW FAISS WORKS (Contrastive Examples)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
FAISS = Facebook AI Similarity Search

1. EMBED ACTIVATING EXAMPLES
   - Convert text to embeddings using finance embedding model
   - Creates vector representations of meaning

2. SEARCH FOR SIMILAR TEXTS
   - Find texts with similar embeddings (semantic similarity)
   - These are texts that MEAN similar things

3. FILTER TO NON-ACTIVATING
   - Only keep texts where feature DOESN'T fire
   - These are "hard negatives"

4. WHY THIS HELPS
   - Shows explainer: "Similar meaning, but feature doesn't activate"
   - Helps distinguish boundaries
   - Example: "Regular dividend" vs "Special dividend"
     * Similar meaning (both about dividends)
     * But feature only fires on "special" ones
     * FAISS helps explainer understand this distinction

\n""")
        
        f.write("=" * 80 + "\n")
        f.write("CONCLUSION\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("""
The explanation is generated by an LLM that:
1. Sees FULL CONTEXT (32 tokens) not just individual words
2. Recognizes SEMANTIC PATTERNS across multiple examples
3. Uses CONTRASTIVE examples (FAISS) to understand boundaries
4. Abstracts to CONCEPTS, not just word frequencies

This is why explanations can differ from top words:
- Top words = Statistical frequency
- Explanation = Semantic interpretation

Both are valuable! Use them together to understand features.

\n""")
    
    print(f"\n✓ Simple explanation saved to: {summary_file}")


if __name__ == "__main__":
    main()

