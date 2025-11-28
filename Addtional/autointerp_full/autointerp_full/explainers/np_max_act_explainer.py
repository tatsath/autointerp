"""
Neuronpedia-style Max-Activation Explainer

This explainer uses the max-activation approach from Neuronpedia to generate
concise, precise labels for latent features. It works with cached activations
from Delphi without modifying the Delphi codebase.

Based on: https://github.com/hijohnnylin/neuronpedia/tree/main/apps/autointerp
"""

import json
import re
from typing import Optional

from autointerp_full import logger
from autointerp_full.clients.client import Client, Response
from autointerp_full.latents.latents import ActivatingExample, LatentRecord

from .explainer import Explainer, ExplainerResult


# System prompt for Neuronpedia-style concise labeling (domain-agnostic)
# Can be overridden via prompts.yaml
from autointerp_full.explainers.default.prompt_loader import get_np_max_act_prompt

_DEFAULT_SYSTEM_CONCISE = """You are labeling ONE hidden feature from a language model. You will see examples with activating tokens marked with <<token>> and full context. Infer the single clearest description of what this feature detects.

Input format:
- Each example shows full text with activating tokens marked as <<token>>
- Multiple tokens may activate together, forming phrases or concepts
- The activating_tokens list shows all tokens that activated above threshold
- Full context is provided to understand semantic relationships

Rules:
- Be SPECIFIC and CONCISE (≤ 18 words). No filler.
- Focus on SPECIFIC CONCEPTS with CONTEXT
- AVOID generic terms
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- Pay attention to which tokens activate together - they often form meaningful phrases
- If evidence shows the feature makes the model SAY a particular token/phrase, note it: "say: <TOKEN>"
- If the feature is structural/lexical (headers, tickers, boilerplate), specify the context

Required JSON output format:
{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "Entity/Sector/Event name or 'N/A'",
  "label": "≤18 words, HIGHLY SPECIFIC description with context",
  "say_token": "TOKEN if applicable else 'N/A'"
}"""

SYSTEM_CONCISE = get_np_max_act_prompt('system_concise', _DEFAULT_SYSTEM_CONCISE)


class NPMaxActExplainer(Explainer):
    """
    Neuronpedia-style max-activation explainer.
    
    Uses top K max-activation examples with surrounding context to generate
    concise, precise labels for latent features.
    """

    def __init__(
        self,
        client: Client,
        tokenizer,
        k_max_act: int = 24,
        window: int = 12,
        use_logits: bool = False,
        verbose: bool = False,
        **generation_kwargs,
    ):
        """
        Initialize the NPMaxActExplainer.

        Args:
            client: LLM client for generating explanations.
            tokenizer: Tokenizer for decoding tokens.
            k_max_act: Number of top max-activation examples to use (default: 24).
            window: Context window size around max-act token (default: 12).
            use_logits: Whether to include top logits (requires extra forward pass).
            verbose: Whether to print verbose output.
            **generation_kwargs: Additional generation kwargs.
        """
        super().__init__(client=client, verbose=verbose, **generation_kwargs)
        self.tokenizer = tokenizer
        self.k_max_act = k_max_act
        self.window = window
        self.use_logits = use_logits

    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        """
        Build the prompt from max-activation examples.

        Args:
            examples: List of activating examples (should be sorted by max activation).

        Returns:
            List of message dicts for the LLM.
        """
        # Get top K examples sorted by max activation
        sorted_examples = sorted(
            examples, key=lambda e: e.max_activation, reverse=True
        )[: self.k_max_act]

        max_act_examples = []
        for example in sorted_examples:
            # Get max activation value
            max_activation = float(example.activations.max().item())

            # Get tokens and string tokens (use full window - all 32 tokens)
            tokens = example.tokens
            tokens_list = tokens.tolist()
            
            # Decode entire sequence at once to get proper text without fragmentation
            # This gives us clean, readable text instead of fragmented tokens
            full_decoded = self.tokenizer.decode(tokens_list, skip_special_tokens=False)
            
            # Decode each token individually to identify which tokens activated
            # Note: Individual decoding produces fragments, but we need it to map activations to tokens
            str_tokens = [
                self.tokenizer.decode([t], skip_special_tokens=False) for t in tokens_list
            ]
            all_activations = example.activations.tolist()

            # Calculate threshold (10% of max activation)
            threshold = max_activation * 0.1

            # Identify all activating tokens
            activating_tokens_list = []
            
            for i, (token_str, activation) in enumerate(zip(str_tokens, all_activations)):
                act_val = float(activation)
                if act_val > threshold:
                    token_clean = token_str.strip()
                    activating_tokens_list.append({
                        "token": token_clean,
                        "activation": act_val
                    })
            
            # Use the full decoded text (properly decoded, no fragmentation)
            # The activating_tokens list will help the LLM understand which parts activated
            # We could mark tokens in the text, but that's complex with fragmented token boundaries
            # For now, use clean text and let the LLM use the activating_tokens list
            full_text = full_decoded

            example_dict = {
                "text": full_text,
                "activating_tokens": activating_tokens_list,
                "max_activation": max_activation,
            }

            # Note: top_positive_logits would require an extra forward pass
            # and is optional for the basic implementation
            if self.use_logits:
                # This would require computing logits at max_act positions
                # For now, we'll skip this unless explicitly needed
                example_dict["top_positive_logits"] = None

            max_act_examples.append(example_dict)

        # Build the user prompt
        prompt_data = {
            "feature_id": f"latent_{len(examples)}",  # Placeholder, will be set from record
            "max_act_examples": max_act_examples,
        }

        user_content = json.dumps(prompt_data, indent=2)

        return [
            {"role": "system", "content": SYSTEM_CONCISE},
            {"role": "user", "content": user_content},
        ]

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        """
        Generate explanation for a latent record using max-activation approach.

        Args:
            record: LatentRecord with cached activations.

        Returns:
            ExplainerResult with generated explanation.
        """
        # Use train examples if available, otherwise use all examples
        examples_to_use = record.train if record.train else record.examples

        if not examples_to_use:
            logger.warning(f"No examples available for {record.latent}")
            return ExplainerResult(
                record=record, explanation="No examples available for explanation."
            )

        # Build prompt
        messages = self._build_prompt(examples_to_use)
        
        # Update feature_id in the prompt with actual latent info
        if isinstance(messages[-1]["content"], str):
            try:
                prompt_data = json.loads(messages[-1]["content"])
                prompt_data["feature_id"] = str(record.latent)
                messages[-1]["content"] = json.dumps(prompt_data, indent=2)
            except json.JSONDecodeError:
                pass  # Keep original if parsing fails

        # Generate explanation
        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        assert isinstance(response, Response)

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                logger.info(f"Explanation for {record.latent}: {explanation}")
                logger.info(f"Response: {response.text[:200]}...")

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed for {record.latent}: {repr(e)}")
            logger.error(f"Response text: {response.text}")
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )

    def parse_explanation(self, text: str) -> str:
        """
        Parse the explanation from LLM response.

        Expected format:
        {
          "granularity": "...",
          "focus": "...",
          "label": "...",
          "say_token": "..."
        }

        Returns:
            The label string (or full JSON if parsing fails).
        """
        try:
            # Try to extract JSON
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"label"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            json_match = re.search(json_pattern, text, re.DOTALL)
            
            if not json_match:
                # Try simpler pattern: find first { and matching }
                brace_start = text.find('{')
                if brace_start != -1:
                    brace_count = 0
                    brace_end = -1
                    for i in range(brace_start, len(text)):
                        if text[i] == '{':
                            brace_count += 1
                        elif text[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                brace_end = i + 1
                                break
                    if brace_end > brace_start:
                        json_str = text[brace_start:brace_end]
                        try:
                            json_data = json.loads(json_str)
                            label = json_data.get("label", "")
                            if label:
                                return label
                        except json.JSONDecodeError:
                            pass

            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    label = json_data.get("label", "")
                    if label:
                        return label
                    # If no label, return the full JSON as string
                    return json_str
                except (json.JSONDecodeError, KeyError):
                    pass

            # Fallback: return cleaned text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                return lines[-1]

            return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return text  # Return raw text as fallback

