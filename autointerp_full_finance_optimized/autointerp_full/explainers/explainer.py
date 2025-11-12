import json
import os
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple

import aiofiles

from autointerp_full import logger

from ..clients.client import Client, Response
from ..latents.latents import ActivatingExample, LatentRecord


class ExplainerResult(NamedTuple):
    record: LatentRecord
    """Latent record passed through to scorer."""

    explanation: str
    """Generated explanation for latent."""


@dataclass
class Explainer(ABC):
    """
    Abstract base class for explainers.
    """

    client: Client
    """Client to use for explanation generation. """
    verbose: bool = False
    """Whether to print verbose output."""
    threshold: float = 0.3
    """The activation threshold to select tokens to highlight."""
    temperature: float = 0.0
    """The temperature for explanation generation."""
    generation_kwargs: dict = field(default_factory=dict)
    """Additional keyword arguments for the generation client."""

    async def __call__(self, record: LatentRecord) -> ExplainerResult:
        messages = self._build_prompt(record.train)

        response = await self.client.generate(
            messages, temperature=self.temperature, **self.generation_kwargs
        )
        assert isinstance(response, Response)

        try:
            explanation = self.parse_explanation(response.text)
            if self.verbose:
                logger.info(f"Explanation: {explanation}")
                logger.info(f"Messages: {messages[-1]['content']}")
                logger.info(f"Response: {response}")

            return ExplainerResult(record=record, explanation=explanation)
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return ExplainerResult(
                record=record, explanation="Explanation could not be parsed."
            )

    def parse_explanation(self, text: str) -> str:
        try:
            # First try to extract JSON with granularity fields (handle multi-line JSON)
            # Look for JSON object that contains "granularity" field
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"granularity"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
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
                        json_match = re.search(re.escape(text[brace_start:brace_end]), text)
            
            if json_match:
                try:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    # Build enhanced explanation string with granularity info
                    granularity = json_data.get("granularity", "N/A")
                    focus = json_data.get("focus", "N/A")
                    explanation_text = json_data.get("explanation", "")
                    
                    # Format: "GRANULARITY: [focus] - explanation"
                    if focus and focus != "N/A":
                        enhanced = f"{granularity}: {focus} - {explanation_text}"
                    else:
                        enhanced = f"{granularity} - {explanation_text}"
                    return enhanced
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall through to other parsing methods
            
            # Try to find [EXPLANATION]: pattern
            match = re.search(r"\[EXPLANATION\]:\s*(.*?)(?:\n|$)", text, re.DOTALL)
            if match:
                explanation = match.group(1).strip()
                # Clean up any remaining [EXPLANATION]: prefixes that might have been included
                explanation = re.sub(r"^\[EXPLANATION\]:\s*", "", explanation)
                return explanation
            
            # Fallback: look for any line that starts with [EXPLANATION]:
            lines = text.split('\n')
            for line in lines:
                if line.strip().startswith('[EXPLANATION]:'):
                    explanation = line.replace('[EXPLANATION]:', '').strip()
                    return explanation
            
            # If no [EXPLANATION]: found, return the last non-empty line
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                explanation = lines[-1]
                # Clean up the explanation
                explanation = self._clean_explanation(explanation)
                return explanation
            
            return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing regex failed: {repr(e)}")
            raise

    def _clean_explanation(self, explanation: str) -> str:
        """
        Clean up the explanation by removing square brackets, normalizing case,
        and improving readability.
        """
        # Remove square brackets and extract content
        explanation = re.sub(r'\[([^\]]+)\]', r'\1', explanation)
        
        # Remove any remaining [EXPLANATION]: prefixes
        explanation = re.sub(r'^\[EXPLANATION\]:\s*', '', explanation)
        
        # Normalize case: convert to title case for better readability
        explanation = explanation.strip()
        
        # Handle special cases for better formatting
        if explanation.isupper() and len(explanation) > 3:
            # Convert ALL CAPS to Title Case
            explanation = explanation.title()
        elif explanation.islower() and len(explanation) > 3:
            # Convert all lowercase to Title Case
            explanation = explanation.title()
        
        # Clean up common artifacts
        explanation = re.sub(r'\s+', ' ', explanation)  # Multiple spaces to single
        explanation = re.sub(r'^[:\-\s]+', '', explanation)  # Remove leading colons/dashes
        explanation = re.sub(r'[:\-\s]+$', '', explanation)  # Remove trailing colons/dashes
        
        return explanation.strip()

    def _highlight(self, str_toks: list[str], activations: list[float]) -> str:
        result = ""
        threshold = max(activations) * self.threshold

        def check(i):
            return activations[i] > threshold

        i = 0
        while i < len(str_toks):
            if check(i):
                result += "<<"

                while i < len(str_toks) and check(i):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)

    def _join_activations(
        self,
        str_toks: list[str],
        token_activations: list[float],
        normalized_activations: list[float],
    ) -> str:
        acts = ""
        activation_count = 0
        for str_tok, token_activation, normalized_activation in zip(
            str_toks, token_activations, normalized_activations
        ):
            if token_activation > max(token_activations) * self.threshold:
                # TODO: for each example, we only show the first 10 activations
                # decide on the best way to do this
                if activation_count > 10:
                    break
                acts += f'("{str_tok}" : {int(normalized_activation)}), '
                activation_count += 1

        return "Activations: " + acts

    @abstractmethod
    def _build_prompt(self, examples: list[ActivatingExample]) -> list[dict]:
        pass


async def explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    try:
        async with aiofiles.open(f"{explanation_dir}/{record.latent}.txt", "r") as f:
            explanation = json.loads(await f.read())
        return ExplainerResult(record=record, explanation=explanation)
    except FileNotFoundError:
        logger.info(f"No explanation found for {record.latent}")
        return ExplainerResult(record=record, explanation="No explanation found")


async def random_explanation_loader(
    record: LatentRecord, explanation_dir: str
) -> ExplainerResult:
    explanations = [f for f in os.listdir(explanation_dir) if f.endswith(".txt")]
    if str(record.latent) in explanations:
        explanations.remove(str(record.latent))
    random_explanation = random.choice(explanations)
    async with aiofiles.open(f"{explanation_dir}/{random_explanation}", "r") as f:
        explanation = json.loads(await f.read())

    return ExplainerResult(record=record, explanation=explanation)
