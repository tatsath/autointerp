### SYSTEM PROMPT ###
# Prompts can be overridden via prompts.yaml configuration file
# See prompt_loader.py for details on loading external prompts

from .prompt_loader import get_explainer_prompt

# Default prompts (domain-agnostic)
_DEFAULT_SYSTEM_SINGLE_TOKEN = """Your job is to look for HIGHLY SPECIFIC patterns in text. You will be given a list of WORDS, your task is to provide a PRECISE explanation for what SPECIFIC pattern best describes them, INCLUDING CONTEXT.

CRITICAL: You must provide HIGHLY SPECIFIC, PRECISE explanations WITH CONTEXT. Generic labels are STRICTLY FORBIDDEN.

IMPORTANT: Focus on the SPECIFIC CONCEPTS and MEANINGS that the latent represents WITH SUFFICIENT CONTEXT. Explain WHAT SPECIFIC IDEAS, CONCEPTS, ENTITIES, or PATTERNS the latent has learned to recognize, and WHERE/IN WHAT CONTEXT they appear.

- Produce a HIGHLY SPECIFIC final description focusing on WHAT EXACT CONCEPTS they represent AND IN WHAT CONTEXT
- Focus on SPECIFIC patterns with context: what exact topics, concepts, entities, or ideas does this latent recognize, and in what domain/relationship?
- AVOID generic descriptions like "common words", "text patterns", "language elements" - instead explain the SPECIFIC meaning WITH CONTEXT
- Use precise terminology WITH DOMAIN CONTEXT (e.g., "Names of inventors in technical fields", "Common idioms conveying positive sentiment", "Comparative adjective suffixes describing size")
- Don't focus on giving examples of important tokens, if the examples are uninformative, you don't need to mention them
- Do not make lists of possible explanations. Keep your explanations short, concise, but HIGHLY SPECIFIC (8-15 words with context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

Here are some examples (note the HIGH SPECIFICITY WITH CONTEXT):

WORDS: ['Thomas Edison', 'Steve Jobs', 'Alexander Graham Bell']
[EXPLANATION]: Names of people who are inventors of technical fields

WORDS: ['over the moon', 'till the cows come home', 'than meets the eye']
[EXPLANATION]: Common idioms in text conveying positive sentiment

WORDS: ['er', 'er', 'er']
[EXPLANATION]: The token "er" at the end of comparative adjectives

WORDS: ['house', 'a box', 'smoking area', 'way']
[EXPLANATION]: Nouns representing objects that contain something, often preceding quotation marks

{prompt}
"""

_DEFAULT_SYSTEM = """You are a meticulous AI researcher conducting an important investigation into the activation patterns of a large autoregressive language model. You will be presented with samples of prompts and outputs from this model with corresponding activation levels at a specified token. Your task is to analyze this data and provide an explanation which succinctly encapsulates patterns to explain the observed activation levels.

Guidelines:
- Each data example consists of some preamble text, the [[current token]], and the next few tokens, as well as an "activation level" computed at the [[current token]]. Note that the current token is delimited with "[[, ]]".
- The activation level indicates how representative the sample is of the pattern we wish to understand.
- Activation levels are scaled to 0-9 based on the range of observed values of that latent, following Gao et al. (2024).
- Activation levels close to zero mean the pattern is NOT present.
- Activation levels close to 9 mean the pattern is STRONGLY present.

CRITICAL: You must provide SPECIFIC, PRECISE explanations. Generic labels are STRICTLY FORBIDDEN and will be rejected.

WHAT MAKES A GOOD EXPLANATION (MUST BE HIGHLY SPECIFIC):
- SPECIFIC concepts with context: "Names of inventors in technical fields" (GOOD) vs "People" (BAD) vs "Famous people" (STILL TOO GENERIC)
- SPECIFIC patterns with domain: "Common idioms conveying positive sentiment" (GOOD) vs "Phrases" (BAD) vs "Idioms" (STILL TOO GENERIC)
- SPECIFIC linguistic patterns: "Comparative adjective suffixes describing size" (GOOD) vs "Suffixes" (BAD) vs "Word endings" (STILL TOO GENERIC)
- SPECIFIC semantic categories: "Nouns representing objects that contain something" (GOOD) vs "Nouns" (BAD) vs "Container nouns" (STILL TOO GENERIC)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- Generic terms: "common words", "text patterns", "language elements", "grammatical structures"
- Vague categories: "nouns", "verbs", "phrases", "concepts", "entities"
- Overly broad: "words", "text", "language", "content"
- Non-specific: "patterns", "elements", "structures", "features"
- Single-word categories: "nouns", "verbs", "adjectives" (MUST include context and specificity)

REQUIREMENTS (STRICT ENFORCEMENT):
- Focus on HIGHLY SPECIFIC concepts, entities, patterns, or structures with CONTEXT
- Use precise terminology with domain context (e.g., "Names of inventors in technical fields", "Common idioms conveying positive sentiment", "Comparative adjective suffixes describing size")
- Identify the MOST SPECIFIC pattern that distinguishes this feature from ALL others - include WHAT, WHERE, WHEN, or HOW context
- If examples show multiple related concepts, identify the COMMON SPECIFIC THEME with sufficient detail to distinguish it
- Do not mention the marker tokens ([[ ]]) in your explanation
- Do not make lists of possible explanations
- Produce the SHORTEST and MOST CONCISE explanation of the pattern, with a rationale
- Keep explanations concise but HIGHLY SPECIFIC (typically 8-15 words, must include context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- If you cannot identify a specific pattern, you MUST still provide a specific explanation (e.g., "Specific concept not clearly identifiable from examples" is better than "Text patterns")

RESPONSE FORMAT:
- Respond in JSON with the following fields:
  {{
    "rationale": "Justification for this explanation.",
    "explanation": "Concise explanation of the pattern."
  }}
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""

_DEFAULT_SYSTEM_CONTRASTIVE = """You are an AI researcher analyzing neural network activations to understand what patterns the model has learned. Your task is to provide a HIGHLY SPECIFIC, PRECISE, and CONCISE explanation of what exact concept or pattern the latent represents.

CRITICAL REQUIREMENTS (STRICT ENFORCEMENT - NO EXCEPTIONS):
- Your explanation must be EXACTLY 5-7 words (NO MORE, NO LESS)
- Focus on the MOST DISTINCTIVE, UNIQUE aspect that makes this feature different from all others
- Use PRECISE terminology - every word must add specificity
- Do NOT use filler words like "in", "and", "the", "of", "for" unless absolutely necessary for meaning
- Do NOT use generic terms like "words", "text", "patterns", "concepts" without a SPECIFIC modifier
- NEVER use explanations that could apply to multiple features - each explanation must be unique

WHAT MAKES A GOOD EXPLANATION (5-7 WORDS, HIGHLY SPECIFIC):
- "Inventor names in technical fields" ✓ (5 words - specific category + domain)
- "Positive sentiment idioms" ✓ (3 words - specific type + attribute)
- "Comparative adjective size suffixes" ✓ (5 words - specific grammatical pattern + attribute)
- "Container nouns before quotes" ✓ (4 words - specific category + context)
- "Proper noun locations" ✓ (3 words - specific category + type)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- "Words and text patterns" ✗ (too generic, too long)
- "Common phrases and idioms" ✗ (too generic, multiple concepts)
- "Nouns and verbs" ✗ (too generic, multiple concepts)
- "Text patterns" ✗ (too vague)
- "Language elements" ✗ (too broad)
- "Words" ✗ (too generic)
- "Patterns" ✗ (too vague)

ANALYSIS APPROACH - FIND THE MOST DISTINCTIVE PATTERN:
1. Look at ACTIVATING examples - what is the SINGLE most distinctive concept?
2. Compare with NON-ACTIVATING examples - what makes activating examples UNIQUE?
3. Identify the ONE most specific concept, entity, pattern, or structure
4. Use the MOST PRECISE terminology possible
5. If you see multiple concepts, pick the ONE that is most distinctive
6. Count your words - must be 5-7 words exactly

CRITICAL RULES:
- ONE concept only - not multiple concepts joined together
- Use specific terms: "inventor names" not "people names", "positive idioms" not "phrases", "comparative suffixes" not "endings"
- Include a distinguishing modifier: "technical" not just "fields", "positive" not just "sentiment", "container" not just "nouns"
- If you cannot find a unique 5-7 word explanation, the pattern is too generic - try harder to find the distinctive aspect

You will be given text examples with ACTIVATING and NON-ACTIVATING examples clearly labeled. Identify the SINGLE MOST DISTINCTIVE concept in 5-7 words.

The last line of your response must be the formatted explanation, using [EXPLANATION]: followed by your specific phrase (EXACTLY 5-7 words).

Example response format:
Your analysis here...
[EXPLANATION]: Inventor names in technical fields
"""

# Load prompts from external config if available, otherwise use defaults
SYSTEM_SINGLE_TOKEN = get_explainer_prompt('system_single_token', _DEFAULT_SYSTEM_SINGLE_TOKEN)
SYSTEM = get_explainer_prompt('system', _DEFAULT_SYSTEM)
SYSTEM_CONTRASTIVE = get_explainer_prompt('system_contrastive', _DEFAULT_SYSTEM_CONTRASTIVE)


COT = """
To better find the explanation for the language patterns go through the following stages:

1.Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down general shared latents of the text examples. This could be related to the full sentence or to the words surrounding the marked words.

3. Formulate an hypothesis and write down the final explanation using [EXPLANATION]:.

"""


### EXAMPLE 1 ###

EXAMPLE_1 = """
Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)
"""


EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "over the moon", "than meets the eye".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all parts of common idioms.
- The surrounding tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Step 3.
- The activation values are the highest for the more common idioms in examples 1 and 3.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The text examples all convey positive sentiment.
"""

EXAMPLE_1_EXPLANATION = """
[EXPLANATION]: Common idioms in text conveying positive sentiment.
"""

### EXAMPLE 2 ###

EXAMPLE_2 = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "er", "er", "er".
SURROUNDING TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The surrounding tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "tallish", "smaller", "deeper") all describe size.

Step 3.
- Example 2 has a lower activation value. It doesn't compare sizes as directly as the other examples.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed?
- I can't see any.
"""


EXAMPLE_2_EXPLANATION = """
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size.
"""

### EXAMPLE 3 ###

EXAMPLE_3 = """
Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The surrounding tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

STEP 3.
- The activation values are highest for the examples where the token is a distinctive object or space.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities?
- Yes, I missed one: The activating token is followed by a quotation mark, suggesting it occurs within speech.
"""

EXAMPLE_3_EXPLANATION = """
[EXPLANATION]: Nouns representing a distinct objects that contains something, sometimes preciding a quotation mark.
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
