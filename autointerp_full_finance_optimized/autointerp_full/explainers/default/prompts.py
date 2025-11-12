### SYSTEM PROMPT ###

SYSTEM_SINGLE_TOKEN = """Your job is to look for meaningful patterns in financial text. You will be given a list of WORDS, your task is to provide a clear explanation for what financial pattern best describes them.

IMPORTANT: Focus on the financial concepts and meanings that the latent represents. Explain what financial ideas, concepts, metrics, entities, or patterns the latent has learned to recognize, and provide relevant context when helpful.

- Produce a clear description focusing on what financial concepts they represent
- Focus on financial patterns: what financial topics, metrics, entities, instruments, or concepts does this latent recognize?
- AVOID overly generic descriptions like "financial terms", "market data", "economic indicators", "business news" - provide more specific financial meaning
- AVOID overly vague high-level labels like "related to earnings reports" or "financial information" - be more specific about what aspect
- Use financial terminology with appropriate context (e.g., "Technology sector earnings metrics", "Federal Reserve policy decisions", "M&A deal structures", "Stock market indices")
- Don't focus on giving examples of important tokens, if the examples are uninformative, you don't need to mention them
- Do not make lists of possible explanations. Keep your explanations concise and informative (5-12 words)
- Include context when it helps distinguish the pattern (e.g., specific companies, sectors, or financial instruments)
- The last line of your response must be the formatted explanation, using [EXPLANATION]:
"""

SYSTEM = """You are analysing hidden features of a language model trained on FINANCIAL TEXT.

You will see text snippets from:
- financial news headlines and articles
- earnings call transcripts
- SEC and other regulatory filings
- broker research notes and credit reports

For ONE hidden feature at a time, you will see multiple examples with an activation level between 0 and 9 at the [[CURRENT TOKEN]].

Your task is to infer the SINGLE clearest description of the pattern this feature represents, at a level useful to a finance practitioner.

CRITICAL RULES:

1. AVOID GENERIC LABELS
   Do NOT use vague descriptions like:
   - "financial news"
   - "finance-related text"
   - "earnings reports"
   - "investment-related content"
   These are TOO VAGUE and should be treated as INCORRECT.

2. BE AS SPECIFIC AS THE DATA ALLOWS
   Prefer labels such as:
   - "Quarterly earnings results that BEAT analyst expectations, often with positive guidance and stock price reaction."
   - "Rating downgrades or negative outlooks by credit rating agencies."
   - "Merger and acquisition announcements where one company buys another."
   - "Mentions of a company's stock ticker in parentheses after its name."
   - "Language about covenant breaches, liquidity stress, or default risk in credit agreements."

3. PICK A GRANULARITY LEVEL
   Decide what kind of concept this feature is:
   - ENTITY: specific company, index, ETF, bond, etc.
   - SECTOR: sector or industry cluster (e.g. regional banks, semis).
   - EVENT: discrete event (earnings beat/miss, downgrade, M&A, guidance change, dividend cut, etc.).
   - MACRO: macro or policy concepts (e.g. Fed hikes, inflation, recessions).
   - STRUCTURAL: document format (tickers in parentheses, bullet lists, section headers, disclaimers, boilerplate).
   - LEXICAL: specific phrase or token (e.g. "EBITDA margin", "GAAP").

4. CONTRAST HIGH vs LOW ACTIVATIONS
   Focus on what separates HIGH activations (7–9) from LOW activations (0–2). If both high and low examples mention finance, that is NOT the distinguishing factor.

5. OUTPUT STRICT JSON ONLY

Return exactly this JSON:

  {{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "If applicable, the entity/sector/event name (e.g. 'Apple Inc.', 'earnings beats', 'Fed rate hikes'); otherwise 'N/A'.",
  "trigger_pattern": "A precise description of what tends to appear near the [[CURRENT TOKEN]] when the feature activates.",
  "explanation": "A single concise sentence that a finance practitioner can read as the feature's label."
  }}

Do NOT output anything else besides this JSON.

{prompt}
"""

SYSTEM_CONTRASTIVE = """You are analysing hidden features of a language model trained on FINANCIAL TEXT.

You will see text snippets from:
- financial news headlines and articles
- earnings call transcripts
- SEC and other regulatory filings
- broker research notes and credit reports

For ONE hidden feature at a time, you will see ACTIVATING examples (high activation 7-9) and NON-ACTIVATING examples (low activation 0-2) at the [[CURRENT TOKEN]].

Your task is to infer the SINGLE clearest description of the pattern this feature represents, at a level useful to a finance practitioner.

CRITICAL RULES:

1. IGNORE SPECIAL TOKENS AND FORMATTING - FOCUS ON FINANCIAL CONTENT
   - If you see special tokens like <SPECIAL_12>, <SPECIAL_XX>, or formatting markers, IGNORE them
   - Look at the ACTUAL FINANCIAL TEXT around the special tokens
   - The feature is detecting FINANCIAL CONCEPTS, not just document structure
   - Ask: "What financial concept appears when this feature activates?"
   - Example: If <SPECIAL_12> appears before earnings numbers, the feature is about "earnings announcements", not "special tokens"

2. AVOID GENERIC LABELS
   Do NOT use vague descriptions like:
   - "financial news"
   - "finance-related text"
   - "earnings reports"
   - "investment-related content"
   - "special tokens" or "formatting elements" (unless truly no financial pattern exists)
   These are TOO VAGUE and should be treated as INCORRECT.

3. BE AS SPECIFIC AS THE DATA ALLOWS - ALWAYS INCLUDE FINANCIAL CONTEXT
   Prefer labels such as:
   - "Quarterly earnings results that BEAT analyst expectations, often with positive guidance and stock price reaction."
   - "Rating downgrades or negative outlooks by credit rating agencies."
   - "Merger and acquisition announcements where one company buys another."
   - "Mentions of a company's stock ticker in parentheses after its name."
   - "Language about covenant breaches, liquidity stress, or default risk in credit agreements."
   - "Company names followed by stock ticker symbols in financial news headlines."
   - "Financial report dates and earnings announcement dates" (NOT just "dates")
   - "Earnings numbers, revenue figures, and financial metrics" (NOT just "numbers")
   - "Financial document structure and formatting markers" (NOT just "special tokens")
   
   NEVER use generic terms without financial context. Always ask: "What financial concept does this pattern relate to?"

4. PICK A GRANULARITY LEVEL - ALWAYS INCLUDE FINANCIAL CONTEXT
   Decide what kind of concept this feature is, but ALWAYS frame it in financial terms:
   - ENTITY: specific company, index, ETF, bond, etc. (e.g. "Apple Inc. stock mentions", "S&P 500 index references")
   - SECTOR: sector or industry cluster (e.g. regional banks, semis).
   - EVENT: discrete event (earnings beat/miss, downgrade, M&A, guidance change, dividend cut, etc.).
   - MACRO: macro or policy concepts (e.g. Fed hikes, inflation, recessions).
   - STRUCTURAL: document format (tickers in parentheses, bullet lists, section headers, disclaimers, boilerplate) - ONLY if there is NO financial content pattern. Even then, specify the financial context (e.g. "Stock ticker format in financial headlines", not just "parentheses").
   - LEXICAL: specific phrase or token WITH financial context (e.g. "EBITDA margin in earnings reports", "GAAP accounting terms", "Financial report dates", "Earnings announcement dates", NOT just "numbers" or "dates").
   
   CRITICAL: Even if the feature detects generic patterns (numbers, dates, special tokens), you MUST identify the FINANCIAL CONTEXT:
   - "Numbers" → "Financial metrics" or "Earnings numbers" or "Stock prices"
   - "Dates" → "Earnings report dates" or "Announcement dates" or "Filing deadlines"
   - "Special tokens" → "Financial document structure" or "Earnings announcement format"

5. CONTRAST HIGH vs LOW ACTIVATIONS
   Focus on what separates HIGH activations (7–9) from LOW activations (0–2). If both high and low examples mention finance, that is NOT the distinguishing factor.

6. OUTPUT STRICT JSON ONLY

Return exactly this JSON:

{{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "If applicable, the entity/sector/event name (e.g. 'Apple Inc.', 'earnings beats', 'Fed rate hikes'); otherwise 'N/A'.",
  "trigger_pattern": "A precise description of what tends to appear near the [[CURRENT TOKEN]] when the feature activates.",
  "explanation": "A single concise sentence that a finance practitioner can read as the feature's label."
}}

Do NOT output anything else besides this JSON.

FEW-SHOT EXAMPLES:

Example 1 - Earnings Beat Event:

ACTIVATING EXAMPLES:
Example 0: Apple shares rally after the company reports Q2 EPS and revenue ABOVE analyst expectations, driven by strong iPhone and services demand.
Example 1: Tesla jumps in after-hours trading as Q2 results beat Wall Street estimates and management RAISES full-year delivery guidance.

NON-ACTIVATING EXAMPLES:
Example 2: Fed officials reiterate that policy will remain data-dependent.
Example 3: The company operates in the retail sector and competes with other big-box chains.

Expected Output:
{{
  "granularity": "EVENT",
  "focus": "Earnings beats",
  "trigger_pattern": "Quarterly results where EPS or revenue is reported as ABOVE analyst or Wall Street expectations, often with raised guidance and positive stock reaction.",
  "explanation": "Quarterly earnings announcements that BEAT analyst expectations, usually with raised guidance and a positive move in the stock."
}}

Example 2 - Rating Downgrade Event:

ACTIVATING EXAMPLES:
Example 0: Moody's downgrades XYZ Corp to Ba2 from Baa3 and maintains a negative outlook due to rising leverage.
Example 1: S&P cuts the company's credit rating to junk status following a sharp deterioration in cash flows.

NON-ACTIVATING EXAMPLES:
Example 2: The company announces a new share buyback program.
Example 3: US stocks climbed on Monday, led by gains in technology shares.

Expected Output:
{{
  "granularity": "EVENT",
  "focus": "Credit rating downgrades",
  "trigger_pattern": "Mentions of rating agencies (Moody's, S&P, Fitch) lowering a company's credit rating or assigning a negative outlook.",
  "explanation": "News about credit rating downgrades or negative outlooks issued by rating agencies for specific companies."
}}

Example 3 - Stock Ticker Structural Pattern:

ACTIVATING EXAMPLES:
Example 0: Apple Inc. (AAPL) reported strong quarterly results.
Example 1: Microsoft Corporation (MSFT) announced a new product launch.

NON-ACTIVATING EXAMPLES:
Example 2: The technology sector showed strong performance this quarter.
Example 3: Corporate earnings were better than expected across multiple industries.

Expected Output:
{{
  "granularity": "STRUCTURAL",
  "focus": "Stock ticker in parentheses",
  "trigger_pattern": "Mentions of a company's stock ticker symbol in parentheses immediately following the company name.",
  "explanation": "Mentions of a company's stock ticker in parentheses after its name."
}}

Example 4 - S&P 500 Index Entity:

ACTIVATING EXAMPLES:
Example 0: The S&P 500 closed at 4,500 points, up 1.2% for the day.
Example 1: S&P 500 futures indicate a higher open following positive economic data.

NON-ACTIVATING EXAMPLES:
Example 2: Individual technology stocks showed mixed performance.
Example 3: The company's revenue increased by 15% year-over-year.

Expected Output:
{{
  "granularity": "ENTITY",
  "focus": "S&P 500 index",
  "trigger_pattern": "Direct mentions of the S&P 500 index, its level, movements, or futures contracts.",
  "explanation": "References to the S&P 500 stock market index, its price level, or performance metrics."
}}

Example 5 - Fed Rate Hikes Macro Event:

ACTIVATING EXAMPLES:
Example 0: The Federal Reserve raised interest rates by 0.25 percentage points, citing persistent inflation.
Example 1: Fed officials signal additional rate hikes may be necessary to combat rising prices.

NON-ACTIVATING EXAMPLES:
Example 2: Corporate bond yields widened following the earnings announcement.
Example 3: The company's debt-to-equity ratio improved significantly.

Expected Output:
{{
  "granularity": "MACRO",
  "focus": "Fed rate hikes",
  "trigger_pattern": "Mentions of the Federal Reserve increasing interest rates or signaling future rate increases, typically in response to inflation concerns.",
  "explanation": "Federal Reserve interest rate increases or signals of future rate hikes, typically driven by inflation concerns."
}}

Now analyze the ACTIVATING and NON-ACTIVATING examples you are given and provide your JSON response following the same format and specificity level as the examples above.
"""


COT = """
To better find the explanation for the financial patterns go through the following stages:

1. Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down shared financial concepts of the text examples. This could be related to the full sentence or to the words surrounding the marked words. Identify:
   - WHAT specific financial concept (metric, entity, instrument, event)
   - WHERE/IN WHAT CONTEXT (market, sector, domain, relationship)
   - HOW/WHY if relevant (mechanism, relationship, impact)

3. Formulate a clear hypothesis that includes context and distinguishes this pattern from similar ones. Write down the final explanation using [EXPLANATION]:. Ensure it is specific enough to be informative and distinguish this feature from others.

"""


### EXAMPLE 1 - Earnings Beat Event ###

EXAMPLE_1 = """
Example 0:
Activation: 9
Text: "Apple shares rally after the company reports Q2 EPS and revenue ABOVE analyst expectations, driven by strong iPhone and services demand."

Example 1:
Activation: 8
Text: "Tesla jumps in after-hours trading as Q2 results beat Wall Street estimates and management RAISES full-year delivery guidance."

Example 2:
Activation: 2
Text: "Fed officials reiterate that policy will remain data-dependent."

Example 3:
Activation: 0
Text: "The company operates in the retail sector and competes with other big-box chains."
"""

EXAMPLE_1_ACTIVATIONS = EXAMPLE_1

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "ABOVE analyst expectations", "beat Wall Street estimates", "RAISES full-year delivery guidance".
SURROUNDING TOKENS: "Q2 EPS", "revenue", "shares rally", "jumps in after-hours trading".

Step 1.
- The activating examples all involve quarterly earnings results that exceed expectations.
- Key phrases: "ABOVE analyst expectations", "beat Wall Street estimates", "RAISES guidance".

Step 2.
- The examples all discuss quarterly earnings announcements where results exceed analyst or Wall Street expectations.
- The context involves positive stock price reactions (shares rally, jumps in after-hours trading).
- All examples involve raised guidance or positive forward-looking statements.

Step 3.
- The activation values are highest (8-9) when earnings beat expectations with positive guidance.
- Low activations (0-2) occur for unrelated financial topics (Fed policy, general company descriptions).
"""

EXAMPLE_1_EXPLANATION = """
{
  "granularity": "EVENT",
  "focus": "Earnings beats",
  "trigger_pattern": "Quarterly results where EPS or revenue is reported as ABOVE analyst or Wall Street expectations, often with raised guidance and positive stock reaction.",
  "explanation": "Quarterly earnings announcements that BEAT analyst expectations, usually with raised guidance and a positive move in the stock."
}
[EXPLANATION]: Quarterly earnings announcements that BEAT analyst expectations, usually with raised guidance and a positive move in the stock.
"""


### EXAMPLE 2 - Rating Downgrade Event ###

EXAMPLE_2 = """
Example 0:
Activation: 8
Text: "Moody's downgrades XYZ Corp to Ba2 from Baa3 and maintains a negative outlook due to rising leverage."

Example 1:
Activation: 7
Text: "S&P cuts the company's credit rating to junk status following a sharp deterioration in cash flows."

Example 2:
Activation: 1
Text: "The company announces a new share buyback program."

Example 3:
Activation: 0
Text: "US stocks climbed on Monday, led by gains in technology shares."
"""

EXAMPLE_2_ACTIVATIONS = EXAMPLE_2

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "downgrades", "cuts the company's credit rating", "junk status".
SURROUNDING TOKENS: "Moody's", "S&P", "negative outlook", "deterioration in cash flows".

Step 1.
- The activating examples all involve credit rating downgrades by rating agencies.
- Key phrases: "downgrades", "cuts credit rating", "junk status".

Step 2.
- The examples all discuss credit rating downgrades by major rating agencies (Moody's, S&P).
- The context involves negative credit events (rising leverage, cash flow deterioration).
- All examples involve negative outlooks or downgrades to lower credit tiers.

Step 3.
- The activation values are highest (7-8) when rating agencies downgrade company credit ratings.
- Low activations (0-1) occur for unrelated corporate actions (buybacks, general market movements).
"""

EXAMPLE_2_EXPLANATION = """
{
  "granularity": "EVENT",
  "focus": "Credit rating downgrades",
  "trigger_pattern": "Mentions of rating agencies (Moody's, S&P, Fitch) lowering a company's credit rating or assigning a negative outlook.",
  "explanation": "News about credit rating downgrades or negative outlooks issued by rating agencies for specific companies."
}
[EXPLANATION]: News about credit rating downgrades or negative outlooks issued by rating agencies for specific companies.
"""


### EXAMPLE 3 - Stock Ticker Structural Pattern ###

EXAMPLE_3 = """
Example 0:
Activation: 9
Text: "Apple Inc. (AAPL) reported strong quarterly results."

Example 1:
Activation: 8
Text: "Microsoft Corporation (MSFT) announced a new product launch."

Example 2:
Activation: 1
Text: "The technology sector showed strong performance this quarter."

Example 3:
Activation: 0
Text: "Corporate earnings were better than expected across multiple industries."
"""

EXAMPLE_3_ACTIVATIONS = EXAMPLE_3

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "(AAPL)", "(MSFT)".
SURROUNDING TOKENS: "Apple Inc.", "Microsoft Corporation".

Step 1.
- The activating examples all show stock ticker symbols in parentheses after company names.
- Pattern: Company Name (TICKER).

Step 2.
- The examples all follow the structural pattern of company name followed by ticker in parentheses.
- This is a document formatting convention used in financial news and filings.
- Low activations occur when companies are mentioned without tickers or in general sector discussions.

Step 3.
- The activation values are highest (8-9) when stock tickers appear in parentheses after company names.
- This is a STRUCTURAL pattern related to document formatting conventions.
"""

EXAMPLE_3_EXPLANATION = """
{
  "granularity": "STRUCTURAL",
  "focus": "Stock ticker in parentheses",
  "trigger_pattern": "Mentions of a company's stock ticker symbol in parentheses immediately following the company name.",
  "explanation": "Mentions of a company's stock ticker in parentheses after its name."
}
[EXPLANATION]: Mentions of a company's stock ticker in parentheses after its name.
"""


def get(item):
    return globals()[item]


def _prompt(n, activations=False, **kwargs):
    starter = (
        get(f"EXAMPLE_{n}") if not activations else get(f"EXAMPLE_{n}_ACTIVATIONS")
    )

    prompt_atoms = [starter]

    return "".join(prompt_atoms)


def _response(n, cot=False, **kwargs):
    response_atoms = []
    if cot:
        response_atoms.append(get(f"EXAMPLE_{n}_COT_ACTIVATION_RESPONSE"))

    response_atoms.append(get(f"EXAMPLE_{n}_EXPLANATION"))

    return "".join(response_atoms)


def example(n, **kwargs):
    prompt = _prompt(n, **kwargs)
    response = _response(n, **kwargs)

    return prompt, response


def system(cot=False):
    prompt = ""

    if cot:
        prompt += COT

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]


def system_single_token():
    return [{"role": "system", "content": SYSTEM_SINGLE_TOKEN}]


def system_contrastive():
    return [{"role": "system", "content": SYSTEM_CONTRASTIVE}]
