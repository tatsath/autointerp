# Top Activating Sentences for AutoInterp Features

This document shows the top activating sentences/examples that were sent to the LLM for generating explanations. These examples help understand why explanations are generic.

---

## Feature 18529: "words related to energy materials and financial performance"

**AutoInterp Score:** 0.7857  
**Generated Explanation:** "words related to energy materials and financial performance"

### Top 15 Activating Examples (from log file):

1. **Activation 8.875:** `: AGCO, HSBC Holdings plc, The<< Wendy>>'s, Krispy Kreme Doughnuts and McDonald`
2. **Activation 8.750:** `. of Extension of Time for Compliance and That a<< Reverse>> Stock Split<< Would>> be AppropriateFutures, Dow`
3. **Activation 8.750:** `of Time for Compliance and That a<< Reverse>> Stock Split<< Would>> be AppropriateFutures, Dow Jones Today Edge`
4. **Activation 8.250:** `, SpartanNash Company, The Hain Cele<<stial>> Group, DHT, Sportsman's Warehouse and`
5. **Activation 8.062:** `featured highlights include: Cadence Design System, L<<PL>> Financial, CSX and U.S. Physical Therapy`
6. **Activation 7.750:** `Stock Jack in the Box (JACK), The<< Wendy>>'s Company (WEN) or Sonic Corporation (`
7. **Activation 7.625:** `And<< Oil>> Inventories (Not<< As>> Clear-Cut<< As>> You May Think It Is)California Resources and Ultra`
8. **Activation 7.500:** `Earnings Call TranscriptHollySys Automation Technologies,<< Ltd>>. (H<<OL>>I) CEO Baiqing Sh`
9. **Activation 7.375:** `A PauseEuropean Implosion Sends Panic Through Global Markets<< As>> George Soros Warns 'We May Be Heading For`
10. **Activation 7.375:** `Analyst BlogYour Daily Pharma Scoop: Dynavax<< Achie>>ves Major Milestone,<< GW>> Pharmaceuticals GWP420`
11. **Activation 7.281:** `<< As>> Questions Arise About Europe Approving Latest M<<erg>>ersHow Much Does It Cost To Produce One Barrel`
12. **Activation 5.656:** `ed This Week - That's The Downside Of<< Le>>verage Though There May Be Opportunity HereAnalysts Estimate`
13. **Activation 4.594:** `The Evolving Energy Business Model: A Transformational<< Change>> From 'Drill-Baby-Drill'`
14. **Activation 3.109:** `13: Mast Therapeutics' Potential,<< Johnson>> &<< Johnson>>'s Results, Investor RelationsRadar Signals: RR`
15. **Activation 2.438:** `ights.com Daily Round Up 6/24/<<15>>: Groupon, Allstate, La Jolla`

### Analysis:
- **Pattern observed:** Most examples show company names (AGCO, HSBC, Wendy's, etc.)
- **Energy-related:** Only example #13 mentions "Energy Business Model" and "Drill-Baby-Drill"
- **Issue:** The explanation "energy materials and financial performance" is too generic - most examples are just company names, not specifically about energy

---

## Feature 25313: "the number 13 in various contexts including dates and percentages"

**AutoInterp Score:** 1.0000 (Perfect!)  
**Generated Explanation:** "the number 13 in various contexts including dates and percentages"

### Top 15 Activating Examples (from log file):

1. **Activation 4.500:** `Pharma's special meeting of shareholders set for November <<13>> to approve additional shares for Depomed bidPremark`
2. **Activation 4.125:** `2Top 2 Trade Alert Ideas October <<13>>: Mast Therapeutics' Potential, Johnson & Johnson`
3. **Activation 3.922:** `Vs. Dividends Smack Down Between <<13>> Top Dividend Aristocrat Survivors Will Wake You`
4. **Activation 3.891:** `New York MellonBerkshire's Revealing <<13>>F: Buffett Didn't Buy The Dip - Didn`
5. **Activation 3.641:** `lung infectionTop 2 Trade Alert Ideas October <<13>>: Mast Therapeutics' Potential, Johnson & Johnson`
6. **Activation 3.641:** `Pharma despite positive early-state data; shares slump <<13>>% in early tradingInsiderInsights.com Daily`
7. **Activation 3.547:** `iderInsights.com Daily Round Up 10/<<13>>/16: Ruby Tuesday, Vishay Precision,`
8. **Activation 3.453:** `iderInsights.com Daily Round Up 1/<<13>>/16: Tuesday Morning, Conn's, Barnes`
9. **Activation 3.344:** `treatment of rare form of epilepsy; shares up <<13>>% premarketBiotech Forum Daily Digest: Another`
10. **Activation 3.312:** `iderInsights.com Daily Round Up 10/<<13>>/16: Ruby Tuesday, Vishay Precision,`
11. **Activation 3.297:** `Affimed: Positive Takeaways From AFM-<<13>> Focused R&D Day; Multiple Catalysts Ahead`
12. **Activation 3.016:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`
13. **Activation 2.891:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`
14. **Activation 2.484:** `orenal diseases in Japan; ARDX up <<13>>% premarketArdelyx prepares to launch`
15. **Activation 2.188:** `iderInsights.com Daily Round Up 7/<<13>>/15: Lawson Products, Atlas Energy, Tet`

### Analysis:
- **Pattern observed:** ALL examples contain the number "13" in dates (October 13, 10/13/16, 7/13/15) or percentages (13%)
- **Explanation quality:** This is a GOOD explanation - it's specific enough (the number 13) but general enough (various contexts)
- **Why it works:** The pattern is clear and consistent across all examples

---

## Feature 6105: "words related to business and finance terms"

**AutoInterp Score:** 0.5714  
**Generated Explanation:** "words related to business and finance terms"

### Analysis:
- **Issue:** This explanation is TOO GENERIC
- **Problem:** "business and finance terms" could describe almost any financial text
- **Need:** More specific pattern identification

---

## Key Observations

### Why Explanations Are Generic:

1. **Prompt says "Try not to be overly specific"** - This pushes LLM toward generic summaries
2. **20-word limit** - Forces very short, high-level descriptions
3. **No activation values shown** - LLM doesn't know which examples are most important
4. **Examples may be too diverse** - If examples show different patterns, LLM generalizes broadly

### Comparison with autointerp_full:

- **autointerp_full:** Asked for "MOST DISTINCTIVE" → too specific (picked rare phrases)
- **autointerp_saeeval:** Asks to "avoid being specific" → too generic (broad summaries)

### The Balance Needed:

The prompt should ask for:
- **"The MOST COMMON pattern that appears in AT LEAST 5-10 examples"**
- **Not "most distinctive"** (too specific)
- **Not "avoid being specific"** (too generic)
- **But "find the pattern that best represents MOST of the examples"**

---

## Current Prompt (Line 389 in main.py):

```
We're studying neurons in a neural network. Each neuron activates on some particular word/words/substring/concept in a short document. The activating words in each document are indicated with << ... >>. We will give you a list of documents on which the neuron activates, in order from most strongly activating to least strongly activating. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try not to be overly specific in your explanation. Note that some neurons will activate only on specific words or substrings, but others will activate on most/all words in a sentence provided that sentence contains some particular concept. Your explanation should cover most or all activating words (for example, don't give an explanation which is specific to a single word if all words in a sentence cause the neuron to activate). Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.
```

### Problematic Phrases:
- ❌ "Try not to be overly specific" → encourages generic explanations
- ❌ "limited to 20 words or less" → forces oversimplification
- ✅ "in order from most strongly activating" → good, tells LLM examples are sorted
- ✅ "Your explanation should cover most or all activating words" → good, requires coverage

---

## Suggested Prompt Modification:

Change from:
- "Try not to be overly specific in your explanation"

To:
- "Find the MOST COMMON pattern that appears in AT LEAST 5-10 of the examples. Your explanation should represent the majority of examples, not just 1-2 examples. If a pattern only appears in 1-2 examples, it's too specific - find a broader pattern that captures what MOST examples have in common."

This balances:
- Not being too specific (like autointerp_full's "most distinctive")
- Not being too generic (current "try not to be overly specific")
- Requiring the pattern to appear in multiple examples
