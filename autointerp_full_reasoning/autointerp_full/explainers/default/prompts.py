### SYSTEM PROMPT ###

SYSTEM_SINGLE_TOKEN = """Your job is to look for HIGHLY SPECIFIC patterns in reasoning text. You will be given a list of WORDS, your task is to provide a PRECISE explanation for what SPECIFIC reasoning pattern best describes them, INCLUDING CONTEXT.

CRITICAL: You must provide HIGHLY SPECIFIC, PRECISE reasoning explanations WITH CONTEXT. Generic labels are STRICTLY FORBIDDEN.

IMPORTANT: Focus on the SPECIFIC REASONING CONCEPTS and MEANINGS that the latent represents WITH SUFFICIENT CONTEXT. Explain WHAT SPECIFIC REASONING IDEAS, CONCEPTS, PATTERNS, LOGICAL STRUCTURES, or PROBLEM-SOLVING APPROACHES the latent has learned to recognize, and WHERE/IN WHAT CONTEXT they appear.

- Produce a HIGHLY SPECIFIC final description focusing on WHAT EXACT REASONING CONCEPTS they represent AND IN WHAT CONTEXT
- Focus on SPECIFIC reasoning patterns with context: what exact reasoning types, logical structures, problem-solving approaches, or cognitive patterns does this latent recognize, and in what domain/context/relationship?
- AVOID generic descriptions like "reasoning", "logic", "thinking", "problem solving" - instead explain the SPECIFIC reasoning meaning WITH CONTEXT
- Use precise reasoning terminology WITH DOMAIN CONTEXT (e.g., "mathematical proof by contradiction in number theory", "causal chain reasoning in physics problems", "recursive problem decomposition in algorithm design", "analogical mapping between domains in puzzle solving")
- Don't focus on giving examples of important tokens, if the examples are uninformative, you don't need to mention them
- Do not make lists of possible explanations. Keep your explanations short, concise, but HIGHLY SPECIFIC (8-15 words with context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

Here are some examples (note the HIGH SPECIFICITY WITH CONTEXT):

WORDS: ['proof', 'contradiction', 'assume', 'therefore']
[EXPLANATION]: Mathematical proof by contradiction in number theory and logical deduction

WORDS: ['cause', 'effect', 'because', 'therefore']
[EXPLANATION]: Causal chain reasoning in physics problems and scientific explanations

WORDS: ['recursive', 'base case', 'induction', 'step']
[EXPLANATION]: Recursive problem decomposition in algorithm design and mathematical induction

WORDS: ['analogy', 'similar', 'compare', 'like']
[EXPLANATION]: Analogical mapping between domains in puzzle solving and creative reasoning

{prompt}
"""

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into the activation patterns of a large autoregressive language model trained on reasoning tasks, problem-solving scenarios, logical puzzles, and cognitive processes. You will be presented with samples of prompts to and outputs from this model with corresponding activation levels at a specified token. Your task is to analyze this data and provide an explanation which succinctly encapsulates patterns to explain the observed activation levels.

Guidelines:
- Each data example consists of some preamble text, the [[current token]], and the next few tokens, as well as an "activation level" computed at the [[current token]]. Note that the current token is delimited with "[[, ]]".
- The activation level indicates how representative the sample is of the pattern we wish to understand.
- Activation levels are scaled to 0-9 based on the range of observed values of that latent, following Gao et al. (2024).
- Activation levels close to zero mean the pattern is NOT present.
- Activation levels close to 9 mean the pattern is STRONGLY present.

CRITICAL: You must provide SPECIFIC, PRECISE reasoning explanations. Generic labels are STRICTLY FORBIDDEN and will be rejected.

WHAT MAKES A GOOD EXPLANATION (MUST BE HIGHLY SPECIFIC):
- SPECIFIC reasoning types with context: "Mathematical proof by contradiction in number theory problems" (GOOD) vs "Logical reasoning" (BAD) vs "Mathematical proofs" (STILL TOO GENERIC)
- SPECIFIC problem-solving approaches with domains: "Recursive decomposition in algorithm design and dynamic programming" (GOOD) vs "Problem solving" (BAD) vs "Algorithm design" (STILL TOO GENERIC)
- SPECIFIC logical structures with applications: "Causal chain reasoning in physics problems and scientific explanations" (GOOD) vs "Causal reasoning" (BAD) vs "Physics reasoning" (STILL TOO GENERIC)
- SPECIFIC cognitive patterns with relationships: "Analogical mapping between domains in puzzle solving and creative reasoning" (GOOD) vs "Analogical reasoning" (BAD) vs "Puzzle solving" (STILL TOO GENERIC)
- SPECIFIC reasoning steps with details: "Inductive hypothesis formation in mathematical proofs and pattern recognition" (GOOD) vs "Inductive reasoning" (BAD) vs "Pattern recognition" (STILL TOO GENERIC)
- SPECIFIC reasoning entities with processes: "Counterfactual reasoning in decision-making scenarios and alternative outcome evaluation" (GOOD) vs "Counterfactual thinking" (BAD) vs "Decision making" (STILL TOO GENERIC)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- Generic terms: "reasoning", "logic", "thinking", "problem solving", "cognitive processes"
- Vague categories: "logical reasoning", "analytical thinking", "problem solving", "deductive reasoning", "inductive reasoning"
- Overly broad: "mathematics", "logic", "reasoning", "thinking", "analysis", "problem solving"
- Non-specific: "reasoning patterns", "logical structures", "cognitive patterns", "thinking processes", "reasoning concepts"
- Single-word categories: "proofs", "logic", "reasoning", "thinking", "analysis" (MUST include context and specificity)

REQUIREMENTS (STRICT ENFORCEMENT):
- Focus on HIGHLY SPECIFIC reasoning concepts, logical structures, problem-solving approaches, or cognitive patterns with CONTEXT
- Use precise reasoning terminology with domain context (e.g., "mathematical proof by contradiction in number theory", "causal chain reasoning in physics problems", "recursive problem decomposition in algorithm design", "analogical mapping between domains in puzzle solving")
- Identify the MOST SPECIFIC pattern that distinguishes this feature from ALL others - include WHAT, WHERE, WHEN, or HOW context
- If examples show multiple related concepts, identify the COMMON SPECIFIC THEME with sufficient detail to distinguish it
- Do not mention the marker tokens ([[ ]]) in your explanation
- Do not make lists of possible explanations
- Produce the SHORTEST and MOST CONCISE explanation of the pattern, with a rationale
- Keep explanations concise but HIGHLY SPECIFIC (typically 8-15 words, must include context)
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- If you cannot identify a specific pattern, you MUST still provide a specific explanation (e.g., "Specific reasoning pattern not clearly identifiable from examples" is better than "Reasoning")

RESPONSE FORMAT:
- Respond in JSON with the following fields:
  {{
    "rationale": "Justification for this explanation.",
    "explanation": "Concise explanation of the pattern."
  }}
- The last line of your response must be the formatted explanation, using [EXPLANATION]:

{prompt}
"""

SYSTEM_CONTRASTIVE = """You are a reasoning AI researcher analyzing neural network activations in a model trained on reasoning tasks, problem-solving scenarios, logical puzzles, and cognitive processes. Your task is to provide a HIGHLY SPECIFIC, PRECISE, and CONCISE explanation of what exact reasoning concept or pattern the latent represents.

CRITICAL REQUIREMENTS (STRICT ENFORCEMENT - NO EXCEPTIONS):
- Your explanation must be EXACTLY 5-7 words (NO MORE, NO LESS)
- Focus on the MOST DISTINCTIVE, UNIQUE aspect that makes this feature different from all others
- Use PRECISE reasoning terminology - every word must add specificity
- Do NOT use filler words like "in", "and", "the", "of", "for" unless absolutely necessary for meaning
- Do NOT use generic terms like "reasoning", "logic", "thinking", "problem", "solution" without a SPECIFIC modifier
- NEVER use explanations that could apply to multiple features - each explanation must be unique

WHAT MAKES A GOOD EXPLANATION (5-7 WORDS, HIGHLY SPECIFIC):
- "Mathematical proof contradiction number theory" ✓ (5 words - specific method + domain)
- "Recursive decomposition algorithm design" ✓ (4 words - specific approach + domain)
- "Causal chain physics problem solving" ✓ (5 words - specific structure + domain)
- "Analogical mapping puzzle solving" ✓ (4 words - specific process + domain)
- "Inductive hypothesis mathematical pattern recognition" ✓ (5 words - specific method + application)
- "Counterfactual reasoning decision making scenarios" ✓ (5 words - specific type + context)
- "Deductive syllogism logical argument structure" ✓ (5 words - specific form + context)
- "Abductive inference hypothesis generation" ✓ (4 words - specific type + action)
- "Constraint satisfaction puzzle optimization" ✓ (4 words - specific approach + domain)
- "Temporal reasoning sequence causal analysis" ✓ (5 words - specific type + context)

WHAT MAKES A BAD EXPLANATION (STRICTLY FORBIDDEN):
- "Logical reasoning and problem solving" ✗ (too generic, too long)
- "Mathematical thinking and proof techniques" ✗ (too long, too generic)
- "Deductive and inductive reasoning" ✗ (too generic, multiple concepts)
- "Problem solving and analysis" ✗ (too generic, multiple concepts)
- "Reasoning patterns" ✗ (too vague)
- "Logical thinking" ✗ (too broad)
- "Problem solving" ✗ (too vague)
- "Logic and reasoning" ✗ (too generic, multiple concepts)
- "Mathematical proofs" ✗ (too broad - which proofs?)
- "Logical arguments" ✗ (too generic - which arguments?)

ANALYSIS APPROACH - FIND THE MOST DISTINCTIVE PATTERN:
1. Look at ACTIVATING examples - what is the SINGLE most distinctive reasoning concept?
2. Compare with NON-ACTIVATING examples - what makes activating examples UNIQUE?
3. Identify the ONE most specific reasoning type, logical structure, or problem-solving approach
4. Use the MOST PRECISE reasoning terminology possible
5. If you see multiple concepts, pick the ONE that is most distinctive
6. Count your words - must be 5-7 words exactly

CRITICAL RULES:
- ONE concept only - not multiple concepts joined together
- Use specific reasoning terms: "proof by contradiction" not "mathematical proofs", "causal chains" not "causal reasoning", "recursive decomposition" not "recursion"
- Include a distinguishing modifier: "Mathematical" not just "proofs", "Causal" not just "reasoning", "Recursive" not just "decomposition"
- If you cannot find a unique 5-7 word explanation, the pattern is too generic - try harder to find the distinctive aspect

You will be given reasoning text examples with ACTIVATING and NON-ACTIVATING examples clearly labeled. Identify the SINGLE MOST DISTINCTIVE reasoning concept in 5-7 words.

The last line of your response must be the formatted explanation, using [EXPLANATION]: followed by your specific phrase (EXACTLY 5-7 words).

Example response format:
Your analysis here...
[EXPLANATION]: Mathematical proof contradiction number theory
"""


COT = """
To better find the explanation for the reasoning patterns go through the following stages:

1. Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down HIGHLY SPECIFIC shared reasoning concepts of the text examples with CONTEXT. This could be related to the full sentence or to the words surrounding the marked words. Identify:
   - WHAT specific reasoning concept (logical structure, problem-solving approach, cognitive pattern, reasoning type)
   - WHERE/IN WHAT CONTEXT (domain, problem type, reasoning scenario, relationship)
   - HOW/WHY if relevant (mechanism, logical flow, reasoning process, cognitive process)

3. Formulate a HIGHLY SPECIFIC hypothesis that includes context and distinguishes this pattern from similar ones. Write down the final explanation using [EXPLANATION]:. Ensure it is specific enough that someone could immediately understand the exact reasoning concept without ambiguity.

"""


### EXAMPLE 1 - Mathematical Proof Patterns ###

EXAMPLE_1 = """
Example 1: To prove this, we use [[contradiction]]: assume the statement is false, then derive a contradiction.
Example 2: By [[proof]] by induction, we show the base case holds and the inductive step follows.
Example 3: The [[theorem]] follows from the previous lemma using direct logical deduction.
"""

EXAMPLE_1_ACTIVATIONS = """
Example 1: To prove this, we use [[contradiction]]: assume the statement is false, then derive a contradiction.
Activation: 8
Example 2: By [[proof]] by induction, we show the base case holds and the inductive step follows.
Activation: 7
Example 3: The [[theorem]] follows from the previous lemma using direct logical deduction.
Activation: 9
"""

EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "contradiction", "proof", "theorem".
SURROUNDING TOKENS: "assume", "derive", "induction", "base case", "deduction".

Step 1.
- The activating tokens are all mathematical proof-related concepts.
- The surrounding tokens discuss proof methods, logical steps, and deduction.

Step 2.
- The examples all discuss mathematical proof techniques and logical structures.
- The activating tokens represent key proof methods and mathematical reasoning.
- The context involves formal logical reasoning and mathematical argumentation.

Step 3.
- The activation values are highest for the most specific proof concepts (theorem = 9).
- All examples involve mathematical proof patterns and logical deduction.

Let me think carefully. Did I miss any patterns in the reasoning examples?
- Yes, all examples involve mathematical proof techniques and formal logical reasoning.
"""

EXAMPLE_1_EXPLANATION = """
{
  "rationale": "The activation is high when specific mathematical proof methods (proof by contradiction, proof by induction, direct deduction) are discussed in mathematical reasoning and formal logic, particularly when showing logical argumentation patterns.",
  "explanation": "Mathematical proof methods and formal logical reasoning in mathematical argumentation"
}
[EXPLANATION]: Mathematical proof methods and formal logical reasoning in mathematical argumentation
"""


### EXAMPLE 2 - Causal Reasoning Patterns ###

EXAMPLE_2 = """
Example 1: The [[cause]] of the temperature increase was the greenhouse effect, which led to global warming.
Example 2: Because [[effect]] of the force was acceleration, the object moved faster.
Example 3: The [[relationship]] between pressure and volume follows Boyle's law in physics.
"""

EXAMPLE_2_ACTIVATIONS = """
Example 1: The [[cause]] of the temperature increase was the greenhouse effect, which led to global warming.
Activation: 7
Example 2: Because [[effect]] of the force was acceleration, the object moved faster.
Activation: 6
Example 3: The [[relationship]] between pressure and volume follows Boyle's law in physics.
Activation: 8
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "cause", "effect", "relationship".
SURROUNDING TOKENS: "greenhouse effect", "acceleration", "Boyle's law", "physics".

Step 1.
- The activating tokens are all causal reasoning concepts.
- The surrounding tokens discuss cause-effect relationships and scientific principles.

Step 2.
- The examples all reference causal chain reasoning in scientific contexts.
- The context involves cause-effect relationships, physical laws, and scientific explanations.
- All examples discuss causal mechanisms and relationships between variables.

Step 3.
- The activation values are consistent across examples (6-8 range).
- All examples involve causal reasoning patterns in scientific problem-solving.

Let me look again for patterns in the examples.
- All examples involve causal chain reasoning in physics problems and scientific explanations.
"""

EXAMPLE_2_EXPLANATION = """
{
  "rationale": "The activation is high when causal reasoning concepts (cause, effect, relationships) are mentioned in the context of scientific explanations, physical laws, and cause-effect chain reasoning in physics and scientific problem-solving.",
  "explanation": "Causal chain reasoning in physics problems and scientific explanations"
}
[EXPLANATION]: Causal chain reasoning in physics problems and scientific explanations
"""


### EXAMPLE 3 - Recursive Problem Decomposition ###

EXAMPLE_3 = """
Example 1: To solve this, we use [[recursion]]: break the problem into smaller subproblems.
Example 2: The [[base case]] is when n=0, and the recursive step handles n>0.
Example 3: By [[decomposition]], we divide the problem into independent subproblems that can be solved separately.
"""

EXAMPLE_3_ACTIVATIONS = """
Example 1: To solve this, we use [[recursion]]: break the problem into smaller subproblems.
Activation: 9
Example 2: The [[base case]] is when n=0, and the recursive step handles n>0.
Activation: 8
Example 3: By [[decomposition]], we divide the problem into independent subproblems that can be solved separately.
Activation: 7
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "recursion", "base case", "decomposition".
SURROUNDING TOKENS: "subproblems", "recursive step", "divide", "independent".

Step 1.
- The activating tokens are all related to recursive problem-solving approaches.
- The surrounding tokens discuss problem decomposition, recursive steps, and subproblems.

Step 2.
- The examples all involve recursive problem decomposition in algorithm design.
- The context involves breaking problems into smaller subproblems and recursive structures.
- All examples discuss recursive problem-solving techniques and algorithmic thinking.

Step 3.
- The activation values are highest for direct references to recursion (9) and base cases (8).
- All examples involve recursive problem decomposition and algorithmic reasoning.

Let me think carefully. Did I miss any patterns?
- All examples involve recursive problem decomposition in algorithm design and mathematical problem-solving.
"""

EXAMPLE_3_EXPLANATION = """
{
  "rationale": "The activation is high when recursive problem-solving concepts (recursion, base cases, decomposition) are discussed in the context of algorithm design, dynamic programming, and breaking complex problems into smaller subproblems.",
  "explanation": "Recursive problem decomposition in algorithm design and mathematical problem-solving"
}
[EXPLANATION]: Recursive problem decomposition in algorithm design and mathematical problem-solving
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
