### SYSTEM PROMPT ###

SYSTEM_SINGLE_TOKEN = """Your job is to look for HIGHLY SPECIFIC patterns in financial text. You will be given a list of WORDS, your task is to provide a PRECISE explanation for what SPECIFIC financial pattern best describes them, INCLUDING CONTEXT.

CRITICAL: You must provide HIGHLY SPECIFIC, PRECISE financial explanations WITH CONTEXT. Generic labels are STRICTLY FORBIDDEN.

IMPORTANT: Focus on the SPECIFIC FINANCIAL CONCEPTS and MEANINGS that the latent represents WITH SUFFICIENT CONTEXT. Explain WHAT SPECIFIC FINANCIAL IDEAS, CONCEPTS, METRICS, ENTITIES, or PATTERNS the latent has learned to recognize, and WHERE/IN WHAT CONTEXT they appear.

- Produce a HIGHLY SPECIFIC final description focusing on WHAT EXACT FINANCIAL CONCEPTS they represent AND IN WHAT CONTEXT
- Focus on SPECIFIC financial patterns with context: what exact financial topics, metrics, entities, instruments, or concepts does this latent recognize, and in what domain/market/relationship?
- AVOID generic descriptions like "financial terms", "market data", "economic indicators", "business news" - instead explain the SPECIFIC financial meaning WITH CONTEXT
- Use precise financial terminology WITH DOMAIN CONTEXT (e.g., "EBITDA margins in technology sector earnings", "credit default swap spreads in European corporate debt", "quantitative easing policy by Federal Reserve", "IPO pricing multiples in venture capital exits")
- Don't focus on giving examples of important tokens, if the examples are uninformative, you don't need to mention them
- Do not make lists of possible explanations. Keep your explanations short, concise, but HIGHLY SPECIFIC (8-15 words with context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

Here are some examples (note the HIGH SPECIFICITY WITH CONTEXT):

WORDS: ['revenue', 'earnings', 'profit margin', 'EBITDA']
[EXPLANATION]: Revenue growth rates, earnings per share, and profit margin expansion in corporate earnings reports

WORDS: ['Federal Reserve', 'interest rates', 'monetary policy', 'inflation']
[EXPLANATION]: Federal Reserve interest rate policy decisions and their relationship to inflation expectations

WORDS: ['IPO', 'merger', 'acquisition', 'takeover']
[EXPLANATION]: Merger and acquisition deal structures and transaction multiples in private equity markets

WORDS: ['NASDAQ', 'Dow Jones', 'S&P 500', 'stock index']
[EXPLANATION]: Major U.S. stock market indices (NASDAQ, Dow Jones, S&P 500) and daily index performance metrics

{prompt}
"""

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into the activation patterns of a large autoregressive language model trained on financial news, market data, corporate reports, and economic indicators. You will be presented with samples of prompts to and outputs from this model with corresponding activation levels at a specified token. Your task is to analyze this data and provide an explanation which succinctly encapsulates patterns to explain the observed activation levels.

Guidelines:
- Each data example consists of some preamble text, the [[current token]], and the next few tokens, as well as an "activation level" computed at the [[current token]]. Note that the current token is delimited with "[[, ]]".
- The activation level indicates how representative the sample is of the pattern we wish to understand.
- Activation levels are scaled to 0-9 based on the range of observed values of that latent, following Gao et al. (2024).
- Activation levels close to zero mean the pattern is NOT present.
- Activation levels close to 9 mean the pattern is STRONGLY present.

CRITICAL: You must provide SPECIFIC, PRECISE financial explanations. Generic labels are STRICTLY FORBIDDEN and will be rejected.

WHAT MAKES A GOOD EXPLANATION (MUST BE HIGHLY SPECIFIC):
- SPECIFIC financial metrics with context: "Revenue growth rates and profit margin expansion in corporate earnings reports" (GOOD) vs "Financial performance" (BAD) vs "Corporate earnings" (STILL TOO GENERIC)
- SPECIFIC market sectors with instruments: "Technology stock valuations and IPO pricing in venture capital markets" (GOOD) vs "Stock market data" (BAD) vs "Technology stocks" (STILL TOO GENERIC)
- SPECIFIC financial instruments with metrics: "Corporate bond yields and credit default swap spreads in fixed income markets" (GOOD) vs "Bond information" (BAD) vs "Corporate bonds" (STILL TOO GENERIC)
- SPECIFIC economic indicators with relationships: "Federal Reserve interest rate policy decisions and their relationship to inflation expectations" (GOOD) vs "Economic data" (BAD) vs "Interest rates" (STILL TOO GENERIC)
- SPECIFIC corporate events with details: "Merger and acquisition deal structures and transaction multiples in private equity" (GOOD) vs "Corporate transactions" (BAD) vs "M&A deals" (STILL TOO GENERIC)
- SPECIFIC market entities with metrics: "Major U.S. stock market indices (NASDAQ, Dow Jones, S&P 500) and daily index performance metrics" (GOOD) vs "Stock indices" (BAD) vs "Market indices" (STILL TOO GENERIC)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- Generic terms: "financial data", "market information", "economic indicators", "business news", "financial terms"
- Vague categories: "financial reports", "market trends", "investment information", "corporate finance", "market analysis"
- Overly broad: "stock market", "banking", "trading", "finance", "economics", "investments"
- Non-specific: "financial concepts", "market data", "economic concepts", "business information", "financial metrics"
- Single-word categories: "stocks", "bonds", "rates", "earnings", "revenue" (MUST include context and specificity)

REQUIREMENTS (STRICT ENFORCEMENT):
- Focus on HIGHLY SPECIFIC financial concepts, metrics, entities, instruments, or patterns with CONTEXT
- Use precise financial terminology with domain context (e.g., "EBITDA margins in technology sector earnings", "credit default swap spreads in European corporate debt", "quantitative easing policy by Federal Reserve", "IPO pricing multiples in venture capital exits")
- Identify the MOST SPECIFIC pattern that distinguishes this feature from ALL others - include WHAT, WHERE, WHEN, or HOW context
- If examples show multiple related concepts, identify the COMMON SPECIFIC THEME with sufficient detail to distinguish it
- Do not mention the marker tokens ([[ ]]) in your explanation
- Do not make lists of possible explanations
- Produce the SHORTEST and MOST CONCISE explanation of the pattern, with a rationale
- Keep explanations concise but HIGHLY SPECIFIC (typically 8-15 words, must include context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- If you cannot identify a specific pattern, you MUST still provide a specific explanation (e.g., "Specific financial concept not clearly identifiable from examples" is better than "Financial data")

RESPONSE FORMAT:
- Respond in JSON with the following fields:
  {{
    "rationale": "Justification for this explanation.",
    "explanation": "Concise explanation of the pattern."
  }}
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""

SYSTEM_CONTRASTIVE = """You are a financial AI researcher analyzing neural network activations in a model trained on financial news, market data, corporate earnings reports, and economic indicators. Your task is to provide a HIGHLY SPECIFIC, PRECISE, and CONCISE explanation of what exact financial concept or pattern the latent represents.

CRITICAL REQUIREMENTS (STRICT ENFORCEMENT - NO EXCEPTIONS):
- Your explanation must be EXACTLY 5-7 words (NO MORE, NO LESS)
- Focus on the MOST DISTINCTIVE, UNIQUE aspect that makes this feature different from all others
- Use PRECISE financial terminology - every word must add specificity
- Do NOT use filler words like "in", "and", "the", "of", "for" unless absolutely necessary for meaning
- Do NOT use generic terms like "financial", "market", "company", "stock", "earnings", "revenue" without a SPECIFIC modifier
- NEVER use explanations that could apply to multiple features - each explanation must be unique

WHAT MAKES A GOOD EXPLANATION (5-7 WORDS, HIGHLY SPECIFIC):
- "Technology IPO pricing multiples" ✓ (4 words - specific sector + event + metric)
- "Federal Reserve interest rate decisions" ✓ (5 words - specific entity + action)
- "Credit default swap spread widening" ✓ (5 words - specific instrument + metric + action)
- "Merger acquisition transaction multiples" ✓ (4 words - specific event + metric)
- "REIT commercial property dividend yields" ✓ (5 words - specific instrument + sector + metric)
- "Earnings misses relative to estimates" ✓ (5 words - specific metric + relationship)
- "Corporate high yield bond spreads" ✓ (5 words - specific instrument + type + metric)
- "Analyst credit rating downgrade announcements" ✓ (5 words - specific action + entity + context)
- "Quarterly revenue growth rate expansion" ✓ (5 words - specific metric + context + action)
- "Private equity exit valuation multiples" ✓ (5 words - specific market + metric + context)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- "Stock price movements and company performance" ✗ (too generic, too long)
- "Stock ticker symbol introductions and company name mentions" ✗ (too long, too generic)
- "Earnings misses and cost increases" ✗ (too generic, multiple concepts)
- "Dividend payments and stock ratings" ✗ (too generic, multiple concepts)
- "Financial performance metrics" ✗ (too vague)
- "Stock market information" ✗ (too broad)
- "Company performance" ✗ (too vague)
- "Revenue and earnings" ✗ (too generic, multiple concepts)
- "Interest rates" ✗ (too broad - which rates?)
- "Stock prices" ✗ (too generic - which stocks?)

ANALYSIS APPROACH - FIND THE MOST DISTINCTIVE PATTERN:
1. Look at ACTIVATING examples - what is the SINGLE most distinctive financial concept?
2. Compare with NON-ACTIVATING examples - what makes activating examples UNIQUE?
3. Identify the ONE most specific metric, entity, instrument, or event
4. Use the MOST PRECISE financial terminology possible
5. If you see multiple concepts, pick the ONE that is most distinctive
6. Count your words - must be 5-7 words exactly

CRITICAL RULES:
- ONE concept only - not multiple concepts joined together
- Use specific financial terms: "IPO pricing" not "stock pricing, "CDS spreads" not "bond spreads", "EPS beats" not "earnings"
- Include a distinguishing modifier: "Technology" not just "stocks", "Corporate" not just "bonds", "Analyst" not just "ratings"
- If you cannot find a unique 5-7 word explanation, the pattern is too generic - try harder to find the distinctive aspect

You will be given financial text examples with ACTIVATING and NON-ACTIVATING examples clearly labeled. Identify the SINGLE MOST DISTINCTIVE financial concept in 5-7 words.

The last line of your response must be the formatted explanation, using [EXPLANATION]: followed by your specific phrase (EXACTLY 5-7 words).

Example response format:
Your analysis here...
[EXPLANATION]: Technology IPO pricing multiples
"""


COT = """
To better find the explanation for the financial patterns go through the following stages:

1. Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down HIGHLY SPECIFIC shared financial concepts of the text examples with CONTEXT. This could be related to the full sentence or to the words surrounding the marked words. Identify:
   - WHAT specific financial concept (metric, entity, instrument, event)
   - WHERE/IN WHAT CONTEXT (market, sector, domain, relationship)
   - HOW/WHY if relevant (mechanism, relationship, impact)

3. Formulate a HIGHLY SPECIFIC hypothesis that includes context and distinguishes this pattern from similar ones. Write down the final explanation using [EXPLANATION]:. Ensure it is specific enough that someone could immediately understand the exact financial concept without ambiguity.

"""


### EXAMPLE 1 - Financial Performance Metrics ###

EXAMPLE_1 = """
Example 1: The company reported strong [[revenue]] growth of 15% this quarter, exceeding analyst expectations.
Example 2: [[Earnings]] per share increased to $2.50, driven by improved operational efficiency.
Example 3: The firm's [[profit margin]] expanded significantly, reaching 18% compared to 12% last year.
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1: The company reported strong [[revenue]] growth of 15% this quarter, exceeding analyst expectations.
Activation: 8
Example 2: [[Earnings]] per share increased to $2.50, driven by improved operational efficiency.
Activation: 7
Example 3: The firm's [[profit margin]] expanded significantly, reaching 18% compared to 12% last year.
Activation: 9
"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "revenue", "Earnings", "profit margin".
SURROUNDING TOKENS: "growth", "per share", "expanded", "quarter", "analyst expectations".

Step 1.
- The activating tokens are all financial performance metrics.
- The surrounding tokens discuss growth, increases, and comparisons.

Step 2.
- The examples all discuss corporate financial performance indicators.
- The activating tokens represent key profitability and revenue metrics.
- The context involves positive financial outcomes and growth.

Step 3.
- The activation values are highest for the most specific financial metrics (profit margin = 9).
- All examples involve quantitative financial performance measures.

Let me think carefully. Did I miss any patterns in the financial examples?
- Yes, all examples involve positive financial performance indicators and quantitative metrics.
"""

EXAMPLE_1_EXPLANATION = """
{
  "rationale": "The activation is high when specific quantitative financial performance metrics (revenue growth percentages, earnings per share, profit margin expansion) are discussed in corporate earnings reports and financial statements, particularly when showing positive performance trends.",
  "explanation": "Revenue growth rates, earnings per share, and profit margin expansion in corporate earnings reports"
}
[EXPLANATION]: Revenue growth rates, earnings per share, and profit margin expansion in corporate earnings reports
"""


### EXAMPLE 2 - Market Indices and Trading ###

EXAMPLE_2 = """
Example 1: [[NASDAQ]] closed up 2.3% today, led by technology stocks.
Example 2: The [[Dow Jones]] Industrial Average reached a new all-time high.
Example 3: Trading volume on the [[S&P 500]] was exceptionally high during the session.
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1: [[NASDAQ]] closed up 2.3% today, led by technology stocks.
Activation: 7
Example 2: The [[Dow Jones]] Industrial Average reached a new all-time high.
Activation: 6
Example 3: Trading volume on the [[S&P 500]] was exceptionally high during the session.
Activation: 8
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "NASDAQ", "Dow Jones", "S&P 500".
SURROUNDING TOKENS: "closed up", "Industrial Average", "Trading volume", "all-time high".

Step 1.
- The activating tokens are all major stock market indices.
- The surrounding tokens discuss market movements, trading activity, and market performance.

Step 2.
- The examples all reference major U.S. stock market indices.
- The context involves market performance, trading activity, and market movements.
- All examples discuss quantitative market data (percentages, volumes, highs).

Step 3.
- The activation values are consistent across examples (6-8 range).
- All examples involve references to major stock market benchmarks.

Let me look again for patterns in the examples.
- All examples involve major U.S. stock market indices and their performance metrics.
"""

EXAMPLE_2_EXPLANATION = """
{
  "rationale": "The activation is high when major U.S. stock market indices (NASDAQ, Dow Jones Industrial Average, S&P 500) are mentioned in the context of daily market movements, trading volume, and index performance metrics.",
  "explanation": "Major U.S. stock market indices and daily index performance metrics"
}
[EXPLANATION]: Major U.S. stock market indices and daily index performance metrics
"""


### EXAMPLE 3 - Interest Rates and Monetary Policy ###

EXAMPLE_3 = """
Example 1: The [[Federal Reserve]] announced a 0.25% increase in interest rates.
Example 2: [[Interest rates]] are expected to remain elevated as inflation persists.
Example 3: The central bank's [[monetary policy]] decision surprised market participants.
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1: The [[Federal Reserve]] announced a 0.25% increase in interest rates.
Activation: 9
Example 2: [[Interest rates]] are expected to remain elevated as inflation persists.
Activation: 8
Example 3: The central bank's [[monetary policy]] decision surprised market participants.
Activation: 7
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "Federal Reserve", "Interest rates", "monetary policy".
SURROUNDING TOKENS: "announced", "increase", "expected", "elevated", "inflation", "central bank", "decision".

Step 1.
- The activating tokens are all related to central banking and monetary policy.
- The surrounding tokens discuss policy decisions, rate changes, and economic conditions.

Step 2.
- The examples all involve central banking policy and interest rate discussions.
- The context involves monetary policy decisions and their economic implications.
- All examples discuss quantitative policy measures (rate increases, inflation).

Step 3.
- The activation values are highest for direct references to the Federal Reserve (9) and interest rates (8).
- All examples involve central banking policy and macroeconomic indicators.

Let me think carefully. Did I miss any patterns?
- All examples involve central banking policy, interest rates, and their relationship to economic conditions like inflation.
"""

EXAMPLE_3_EXPLANATION = """
{
  "rationale": "The activation is high when the Federal Reserve, interest rate changes (specifically rate increases or decreases), and monetary policy decisions are discussed in relation to inflation, economic conditions, and their impact on financial markets.",
  "explanation": "Federal Reserve interest rate policy decisions and their relationship to inflation"
}
[EXPLANATION]: Federal Reserve interest rate policy decisions and their relationship to inflation
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
