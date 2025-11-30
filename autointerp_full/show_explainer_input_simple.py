#!/usr/bin/env python3
"""
Show EXACTLY what the explainer LLM sees for a given feature.
This shows the exact format based on the code structure.
"""

import json

# This is the EXACT system prompt from np_max_act_explainer.py
SYSTEM_PROMPT = """You are labeling ONE hidden feature from a language model. You will see the top-activating tokens and short surrounding text spans. Infer the single clearest description of what this feature detects.

Rules:
- Be SPECIFIC and CONCISE (≤ 18 words). No filler.
- Focus on SPECIFIC CONCEPTS with CONTEXT
- AVOID generic terms
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- If evidence shows the feature makes the model SAY a particular token/phrase, note it: "say: <TOKEN>"
- If the feature is structural/lexical (headers, tickers, boilerplate), specify the context

Required JSON output format:
{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "Entity/Sector/Event name or 'N/A'",
  "label": "≤18 words, HIGHLY SPECIFIC description with context",
  "say_token": "TOKEN if applicable else 'N/A'"
}"""


def show_format_example():
    """Show example of what the explainer sees based on the code."""
    
    print("=" * 100)
    print("EXACT INPUT TO EXPLAINER LLM")
    print("Based on code in: autointerp_full/explainers/np_max_act_explainer.py")
    print("=" * 100)
    print()
    
    print("=" * 100)
    print("STEP 1: SYSTEM PROMPT (sent first)")
    print("=" * 100)
    print(SYSTEM_PROMPT)
    print()
    
    print("=" * 100)
    print("STEP 2: USER PROMPT (JSON - sent second)")
    print("=" * 100)
    print()
    print("The explainer receives JSON with this structure:")
    print()
    
    # Example based on what we know from the code
    # The code does: left_context = "".join(str_tokens[start_idx:max_act_idx])
    # This means NO SPACES between tokens!
    
    example_prompt = {
        "feature_id": "latent_18529",
        "max_act_examples": [
            {
                "token": "Equity",
                "left_context": "comeEquityCEFs:10%YieldsWithSomeDefenseForYourPortfolio",
                "right_context": "StocksShowingImprovedRelativeStrength:BancoSantander",
                "activation": 9.75
            },
            {
                "token": "Smart",
                "left_context": "ysTJXInc,JD....AssetsIn",
                "right_context": "BetaEquityETPsGloballyIncreasedByARecord$134BillionDuring2",
                "activation": 7.125
            },
            {
                "token": "ETF",
                "left_context": "ConsumerE",
                "right_context": "sCrushedtheS&P5",
                "activation": 6.65625
            }
            # ... up to 24 examples total (k_max_act=24)
        ]
    }
    
    print(json.dumps(example_prompt, indent=2))
    print()
    
    print("=" * 100)
    print("KEY ISSUES - WHY LABELS ARE GENERIC")
    print("=" * 100)
    print()
    
    print("ISSUE 1: TOKENS JOINED WITHOUT SPACES")
    print("-" * 100)
    print("Code: left_context = ''.join(str_tokens[start_idx:max_act_idx])")
    print("Result: 'comeEquityCEFs:10%YieldsWithSomeDefenseForYourPortfolio'")
    print("Problem: No word boundaries - LLM can't parse properly")
    print("Should be: 'come Equity CEFs: 10% Yields With Some Defense For Your Portfolio'")
    print()
    
    print("ISSUE 2: ONLY ONE TOKEN HIGHLIGHTED PER EXAMPLE")
    print("-" * 100)
    print("Code: Only the token at max_act_idx is shown in 'token' field")
    print("Problem: Other activating tokens are NOT marked")
    print("Example: If 'Smart Beta Equity ETPs' all activate, only 'Smart' is highlighted")
    print("LLM can't see that multiple tokens activate together")
    print()
    
    print("ISSUE 3: CONTEXT IS SPLIT (LEFT + RIGHT)")
    print("-" * 100)
    print("Code: left_context (12 tokens) + token + right_context (12 tokens)")
    print("Problem: Breaks semantic coherence")
    print("Example: 'Smart Beta Equity ETPs' becomes:")
    print("  left_context: 'ysTJXInc,JD....AssetsIn'")
    print("  token: 'Smart'")
    print("  right_context: 'BetaEquityETPsGloballyIncreased...'")
    print("The phrase is broken across fields!")
    print()
    
    print("ISSUE 4: SMALL WINDOW SIZE")
    print("-" * 100)
    print("Code: window = 12 (default)")
    print("Problem: Only ±12 tokens around max activation")
    print("Full examples are 32 tokens, but explainer only sees ±12")
    print("May miss broader semantic patterns")
    print()
    
    print("ISSUE 5: SPECIAL TOKENS VISIBLE")
    print("-" * 100)
    print("Problem: Special tokens like <<SPECIAL_12>>> appear in text")
    print("May confuse LLM about what's actual content vs formatting")
    print()
    
    print("=" * 100)
    print("COMPARISON: What LLM SHOULD See vs What It DOES See")
    print("=" * 100)
    print()
    
    print("EXAMPLE FROM FEATURE 18529:")
    print()
    print("WHAT IT DOES SEE (current format):")
    print("  {")
    print('    "token": "Smart",')
    print('    "left_context": "ysTJXInc,JD....AssetsIn",')
    print('    "right_context": "BetaEquityETPsGloballyIncreasedByARecord$134BillionDuring2",')
    print('    "activation": 7.125')
    print("  }")
    print()
    
    print("WHAT IT SHOULD SEE (better format):")
    print("  {")
    print('    "text": "Assets In <<Smart>> <<Beta>> <<Equity>> <<ETPs>> Globally Increased By A Record $134 Billion During 2020",')
    print('    "activating_tokens": ["Smart", "Beta", "Equity", "ETPs"],')
    print('    "max_activation": 7.125')
    print("  }")
    print()
    print("Or even better - full context with all activating tokens marked:")
    print('  "text": "... TJX Inc, JD. ... Assets In <<Smart>> <<Beta>> <<Equity>> <<ETPs>> Globally Increased ..."')
    print()
    
    print("=" * 100)
    print("CODE LOCATION")
    print("=" * 100)
    print("File: autointerp_full/explainers/np_max_act_explainer.py")
    print("Method: _build_prompt()")
    print("Lines: 110-116 (the problematic join without spaces)")
    print()
    print("Current code:")
    print("  left_context = ''.join(str_tokens[start_idx:max_act_idx])")
    print("  right_context = ''.join(str_tokens[max_act_idx + 1 : end_idx])")
    print()
    print("Should be:")
    print("  left_context = ' '.join(str_tokens[start_idx:max_act_idx])")
    print("  right_context = ' '.join(str_tokens[max_act_idx + 1 : end_idx])")
    print("  # AND mark all activating tokens, not just max")


if __name__ == "__main__":
    show_format_example()

