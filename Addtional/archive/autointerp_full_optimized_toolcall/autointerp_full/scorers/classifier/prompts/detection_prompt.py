DSCORER_SYSTEM_PROMPT = """You are an intelligent and meticulous linguistics researcher.

You will be given a certain latent explanation, such as "technical code and programming instructions" or "emotional language and sentiment expressions".

You will then be given several text examples. Your task is to determine which examples possess the latent pattern described.

For each example in turn, return 1 if the text matches the described pattern or contains the semantic content described, or 0 if it does not. You must return your response in a valid Python list. Do not return anything else besides a Python list.

Guidelines:
- Be lenient: if the explanation describes a semantic pattern, match examples that contain that type of content
- Focus on semantic meaning, not exact words
- Return 1 if the example exhibits the described pattern or closely related content
- Return 0 only if the example is completely unrelated to the described pattern
"""

# https://www.neuronpedia.org/gpt2-small/6-res-jb/6048
DSCORER_EXAMPLE_ONE = """Latent explanation: Technical code and programming instructions.

Test examples:

Example 0: def calculate_sum(a, b): return a + b
Example 1: The weather is nice today and I went for a walk.
Example 2: import numpy as np; arr = np.array([1,2,3])
Example 3: She was happy to see her friends at the party.
Example 4: class MyClass: def __init__(self): pass
"""

DSCORER_RESPONSE_ONE = "[1,0,1,0,1]"

DSCORER_EXAMPLE_TWO = """Latent explanation: Emotional language and sentiment expressions.

Test examples:

Example 0: I'm feeling really excited about this new project!
Example 1: The function takes two parameters and returns their sum.
Example 2: This makes me so happy to see progress on the work.
Example 3: Calculate the integral of x^2 from 0 to 1.
Example 4: I'm thrilled with the results we achieved together.
"""

DSCORER_RESPONSE_TWO = "[1,0,1,0,1]"

DSCORER_EXAMPLE_THREE = """Latent explanation: Structured data, lists, and formatted content.

Test examples:

Example 0: 1. First item\n2. Second item\n3. Third item
Example 1: This is a paragraph of regular text without formatting.
Example 2: - Bullet point\n- Another point\n- Final point
Example 3: The cat sat on the mat and looked around peacefully.
Example 4: Name: John\nAge: 30\nCity: New York
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
