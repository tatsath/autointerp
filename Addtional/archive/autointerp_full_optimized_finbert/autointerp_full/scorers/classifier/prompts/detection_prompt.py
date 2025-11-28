DSCORER_SYSTEM_PROMPT = """You are an expert in financial language and sentiment analysis.

You are given:
- A "latent explanation" that describes an ACTION or SENTIMENT DIRECTION in financial text (e.g., "signals positive sentiment", "triggers bearish reaction", "indicates negative sentiment").
- ONE text example at a time (headline, article snippet, filing excerpt, or transcript segment).

Decide if the example clearly expresses the described ACTION or SENTIMENT DIRECTION.

Return:
- 1 if the text clearly expresses the described ACTION or SENTIMENT DIRECTION (e.g., if explanation describes "positive sentiment" and text shows positive earnings beat, bullish guidance, etc.).
- 0 if the text does NOT express the described action/sentiment direction, or is only general financial news without the specific sentiment/action.

Guidelines:
- Be strict: If the explanation describes "bearish sentiment" but the text is only general financial news without negative implications, return 0.
- Be strict: If the explanation describes "positive sentiment" but the text is neutral or negative, return 0.
- Match the ACTION: If explanation says "signals positive sentiment from earnings beats" and text shows earnings beat, return 1.
- Match the ACTION: If explanation says "triggers negative sentiment from downgrades" and text shows downgrade, return 1.
- Only return 1 if the example clearly expresses the described ACTION or SENTIMENT DIRECTION.
- Return 0 if the example is general financial news without the specific sentiment/action described.

Answer with a single character: 1 or 0.
"""

# Action-oriented examples
DSCORER_EXAMPLE_ONE = """Latent explanation: Signals negative sentiment from earnings misses relative to analyst expectations.

Test examples:

Example 0: Tesla shares slid after the automaker reported quarterly earnings that missed Wall Street estimates on both revenue and profit.
Example 1: The company announced a new share repurchase program and said guidance remains unchanged for the remainder of the year.
Example 2: Netflix topped revenue forecasts but delivered earnings that fell short of consensus, triggering an after-hours selloff.
Example 3: Analysts highlighted margin expansion as the firm beat expectations on every key metric this quarter.
Example 4: United Airlines warned that surging fuel costs will cause this quarter's results to miss forecasts.
"""

DSCORER_RESPONSE_ONE = "[1,0,1,0,1]"

DSCORER_EXAMPLE_TWO = """Latent explanation: Triggers positive sentiment from dividend increase announcements.

Test examples:

Example 0: Procter & Gamble raised its quarterly dividend by 5%, marking the 67th consecutive annual dividend increase for the consumer giant.
Example 1: The Fed signaled that interest rates will stay higher for longer as inflation pressures remain elevated.
Example 2: JPMorgan announced it will boost its dividend to $1.05 per share following strong stress-test results.
Example 3: Shares of Apple climbed after the company reported record iPhone sales in China last quarter.
Example 4: Realty Income declared a monthly dividend of $0.2650 per share, continuing its streak of payout increases.
"""

DSCORER_RESPONSE_TWO = "[1,0,1,0,1]"

DSCORER_EXAMPLE_THREE = """Latent explanation: Indicates negative sentiment from credit rating downgrades.

Test examples:

Example 0: Moody's downgraded the company to junk status citing deteriorating cash flow and rising debt levels.
Example 1: Interest in municipal bonds has climbed as investors search for tax-advantaged income streams.
Example 2: S&P cut the firm's rating to BBB- following the announcement of a major acquisition.
Example 3: Global equity markets traded mixed overnight amid concerns about slowing European growth.
Example 4: Fitch maintained a negative outlook on the company's credit rating due to operational challenges.
"""

DSCORER_RESPONSE_THREE = "[1,0,1,0,1]"

GENERATION_PROMPT = """Latent explanation: {explanation}

Text examples:

{examples}
"""

default = [
    {"role": "user", "content": DSCORER_EXAMPLE_ONE},
    {"role": "assistant", "content": DSCORER_RESPONSE_ONE},
    {"role": "user", "content": DSCORER_EXAMPLE_TWO},
    {"role": "assistant", "content": DSCORER_RESPONSE_TWO},
    {"role": "user", "content": DSCORER_EXAMPLE_THREE},
    {"role": "assistant", "content": DSCORER_RESPONSE_THREE},
]


def prompt(examples: str, explanation: str) -> list[dict]:
    generation_prompt = GENERATION_PROMPT.format(
        explanation=explanation, examples=examples
    )

    prompt = [
        {"role": "system", "content": DSCORER_SYSTEM_PROMPT},
        *default,
        {"role": "user", "content": generation_prompt},
    ]

    return prompt
