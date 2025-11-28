DSCORER_SYSTEM_PROMPT = """You are an expert in financial language.

You are given:
- A "latent explanation" that describes a pattern in financial text.
- ONE text example at a time (headline, article snippet, filing excerpt, or transcript segment).

Decide if the example exhibits the described pattern or a closely related financial concept.

Return:
- 1 if the text matches the described concept, contains related financial terminology, or discusses the same type of financial event/metric/entity.
- 0 if the concept is completely absent or unrelated to the financial domain described.

Guidelines:
- Be lenient: if the explanation mentions "earnings misses" and the example discusses earnings, quarterly results, or analyst expectations, return 1.
- Be lenient: if the explanation mentions "credit rating downgrades" and the example discusses ratings, credit agencies, or downgrades, return 1.
- Be lenient: if the explanation mentions "dividend announcements" and the example discusses dividends, payouts, or shareholder returns, return 1.
- Be lenient: if the explanation mentions "insider trading" and the example discusses trading activity, SEC filings, or corporate insiders, return 1.
- Only return 0 if the example is completely unrelated to the financial concept (e.g., general news, non-financial topics, or completely different financial domains).

Answer with a single character: 1 or 0.
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Latent explanation: Earnings misses relative to analyst expectations.

Test examples:

Example 0: Tesla shares slid after the automaker reported quarterly earnings that missed Wall Street estimates on both revenue and profit.
Example 1: The company announced a new share repurchase program and said guidance remains unchanged for the remainder of the year.
Example 2: Netflix topped revenue forecasts but delivered earnings that fell short of consensus, triggering an after-hours selloff.
Example 3: Analysts highlighted margin expansion as the firm beat expectations on every key metric this quarter.
Example 4: United Airlines warned that surging fuel costs will cause this quarter's results to miss forecasts.
"""

DSCORER_RESPONSE_ONE = "[1,0,1,0,1]"

DSCORER_EXAMPLE_TWO = """Latent explanation: Dividend increase announcements for public companies.

Test examples:

Example 0: Procter & Gamble raised its quarterly dividend by 5%, marking the 67th consecutive annual dividend increase for the consumer giant.
Example 1: The Fed signaled that interest rates will stay higher for longer as inflation pressures remain elevated.
Example 2: JPMorgan announced it will boost its dividend to $1.05 per share following strong stress-test results.
Example 3: Shares of Apple climbed after the company reported record iPhone sales in China last quarter.
Example 4: Realty Income declared a monthly dividend of $0.2650 per share, continuing its streak of payout increases.
"""

DSCORER_RESPONSE_TWO = "[1,0,1,0,1]"

DSCORER_EXAMPLE_THREE = """Latent explanation: Company name followed by ticker symbol in parentheses (e.g., Apple (NASDAQ:AAPL)).

Test examples:

Example 0: Alphabet (NASDAQ:GOOGL) unveiled new AI features for its cloud customers at the annual developer conference.
Example 1: Interest in municipal bonds has climbed as investors search for tax-advantaged income streams.
Example 2: Pfizer (NYSE:PFE) guided full-year revenue lower as COVID vaccine demand continues to fade.
Example 3: Global equity markets traded mixed overnight amid concerns about slowing European growth.
Example 4: Microsoft Corp. (MSFT) said its cloud unit Azure grew 28% year over year, beating expectations.
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
