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


# System prompt for Neuronpedia-style concise labeling (finance-specific)
SYSTEM_CONCISE = """You are labeling ONE hidden feature from a language model trained on FINANCIAL TEXT (financial news, market data, corporate reports, earnings calls, SEC filings, economic indicators). You will see the top-activating tokens and short surrounding text spans. Infer the single clearest description of what this feature detects.

CRITICAL: You must provide HIGHLY SPECIFIC, PRECISE financial explanations WITH CONTEXT. Generic labels are STRICTLY FORBIDDEN.

Rules:
- Be SPECIFIC and CONCISE (≤ 18 words). No filler.
- Focus on SPECIFIC FINANCIAL CONCEPTS with CONTEXT:
  * SPECIFIC financial metrics: "Revenue growth rates and profit margin expansion in corporate earnings reports" (GOOD) vs "Financial performance" (BAD)
  * SPECIFIC market sectors: "Technology stock valuations and IPO pricing in venture capital markets" (GOOD) vs "Stock market data" (BAD)
  * SPECIFIC financial instruments: "Corporate bond yields and credit default swap spreads in fixed income markets" (GOOD) vs "Bond information" (BAD)
  * SPECIFIC economic indicators: "Federal Reserve interest rate policy decisions and their relationship to inflation expectations" (GOOD) vs "Interest rates" (BAD)
  * SPECIFIC corporate events: "Merger and acquisition deal structures and transaction multiples in private equity" (GOOD) vs "Corporate transactions" (BAD)
- AVOID generic terms: "financial data", "market information", "economic indicators", "business news", "financial terms"
- AVOID vague categories: "financial reports", "market trends", "investment information", "corporate finance"
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- If evidence shows the feature makes the model SAY a particular token/phrase, note it: "say: <TOKEN>"
- If the feature is structural/lexical (headers, tickers, boilerplate), specify the financial context: "Stock ticker format in financial headlines" (GOOD) vs "parentheses" (BAD)
- Use precise financial terminology WITH DOMAIN CONTEXT (e.g., "EBITDA margins in technology sector earnings", "credit default swap spreads in European corporate debt")

Required JSON output format:
{
  "granularity": "ENTITY | SECTOR | EVENT | MACRO | STRUCTURAL | LEXICAL",
  "focus": "Entity/Sector/Event name or 'N/A'",
  "label": "≤18 words, HIGHLY SPECIFIC financial description with context",
  "say_token": "TOKEN if applicable else 'N/A'"
}"""


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
            # Find the index of max activation
            max_act_idx = int(example.activations.argmax().item())
            max_activation = float(example.activations[max_act_idx].item())

            # Get tokens and string tokens
            tokens = example.tokens
            str_tokens = example.str_tokens or [
                self.tokenizer.decode([t]) for t in tokens.tolist()
            ]

            # Extract context window
            start_idx = max(0, max_act_idx - self.window)
            end_idx = min(len(tokens), max_act_idx + self.window + 1)

            left_context = "".join(str_tokens[start_idx:max_act_idx])
            right_context = "".join(str_tokens[max_act_idx + 1 : end_idx])
            current_token = str_tokens[max_act_idx] if max_act_idx < len(str_tokens) else ""

            # Get next token if available
            next_token = None
            if max_act_idx + 1 < len(str_tokens):
                next_token = str_tokens[max_act_idx + 1]

            example_dict = {
                "token": current_token,
                "left_context": left_context,
                "right_context": right_context,
                "activation": max_activation,
            }

            if next_token:
                example_dict["next_token"] = next_token

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

