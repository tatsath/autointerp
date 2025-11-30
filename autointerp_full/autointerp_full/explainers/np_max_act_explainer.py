"""
Neuronpedia-style Max-Activation Explainer

This explainer uses the max-activation approach from Neuronpedia to generate
concise, precise labels for latent features. It works with cached activations
from Delphi without modifying the Delphi codebase.

Based on: https://github.com/hijohnnylin/neuronpedia/tree/main/apps/autointerp
"""

import json
import re
import traceback
from typing import Optional
import httpx

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
- Pay close attention to which tokens are marked with <<token>> - these are the key indicators

Rules:
- Be SPECIFIC and CONCISE (≤ 18 words). No filler.
- Focus on SPECIFIC CONCEPTS with CONTEXT
- AVOID generic terms like "declarations", "mentions", "references" - these are too vague
- NEVER use single-word or two-word explanations - always include domain, context, or relationship information
- Pay attention to which tokens activate together - they often form meaningful phrases
- Look for multi-token phrases marked with <<token>> - these represent complete concepts
- If evidence shows the feature makes the model SAY a particular token/phrase, note it: "say: <TOKEN>"
- If the feature is structural/lexical (headers, tickers, boilerplate), specify the context

BAD generic explanations to AVOID:
- "declarations by funds" (too generic - what kind of declarations?)
- "mentions of equity" (too vague - what about equity?)
- "references to financial terms" (not specific enough)
- "text about companies" (too broad)

GOOD specific explanations:
- "Smart Beta Equity ETPs experiencing global asset growth or increases in assets under management"
- "Special dividend declarations by municipal income funds with specific dollar amounts"
- "Consumer ETPs showing significant underperformance relative to S&P 500"

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
        window: int = 3,
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
            window: Context window size around max-act token (default: 3).
            use_logits: Whether to include top logits (requires extra forward pass).
            verbose: Whether to print verbose output.
            **generation_kwargs: Additional generation kwargs.
        """
        super().__init__(client=client, verbose=verbose, **generation_kwargs)
        self.tokenizer = tokenizer
        self.k_max_act = k_max_act
        self.window = window
        self.use_logits = use_logits

    def _mark_activating_tokens_in_text(
        self, 
        full_decoded: str, 
        str_tokens: list[str], 
        all_activations: list[float], 
        threshold: float
    ) -> str:
        """
        Mark activating tokens in the decoded text with <<token>> markers.
        
        This function reconstructs the text from tokens, marking activating ones.
        This ensures proper alignment between tokens and the decoded text.
        
        Args:
            full_decoded: The fully decoded text from tokenizer.decode()
            str_tokens: List of individual token strings
            all_activations: List of activation values for each token
            threshold: Activation threshold above which tokens are marked
            
        Returns:
            Text with activating tokens marked as <<token>>
        """
        # Identify which tokens activate
        activating_indices = set()
        for i, (token_str, activation) in enumerate(zip(str_tokens, all_activations)):
            if float(activation) > threshold:
                activating_indices.add(i)
        
        if not activating_indices:
            return full_decoded
        
        # Reconstruct text from tokens with markers
        # This approach ensures we mark the correct tokens even with BPE fragmentation
        marked_parts = []
        for i, token_str in enumerate(str_tokens):
            if i in activating_indices:
                # Mark activating tokens
                marked_parts.append(f"<<{token_str}>>")
            else:
                marked_parts.append(token_str)
        
        # Join tokens - most tokenizers handle spacing in decode()
        # But when joining individual decoded tokens, we need to be careful
        # The tokenizer's decode() method handles this, so let's decode the marked sequence
        # However, we can't decode with markers, so we'll concatenate and clean up
        
        # Build text by joining tokens (tokenizer.decode handles spacing for individual tokens)
        marked_text = "".join(marked_parts)
        
        # The issue is that full_decoded might have different spacing than concatenated tokens
        # So we'll use a more robust approach: find token positions in full_decoded
        
        # Alternative: Use the tokenizer to properly decode, but we need to mark tokens
        # Let's try a different strategy: build from tokens and use the tokenizer's
        # behavior to ensure proper spacing
        
        # Actually, the most reliable way is to use the tokenizer's decode on the
        # token IDs, but mark tokens as we go. Since we can't modify the decode process,
        # we'll reconstruct by finding positions.
        
        # For now, use a simpler heuristic: if the reconstructed text matches
        # full_decoded (after removing spaces), use the marked version
        # Otherwise, try to align tokens in full_decoded
        
        # Check if simple concatenation matches (after normalization)
        normalized_decoded = full_decoded.replace(" ", "")
        normalized_marked = marked_text.replace("<<", "").replace(">>", "").replace(" ", "")
        
        if normalized_decoded == normalized_marked:
            # Simple case: tokens align, use marked version
            return marked_text
        
        # Complex case: need to align tokens in full_decoded
        # Use a greedy matching approach
        result = full_decoded
        tokens_to_mark = [(i, str_tokens[i]) for i in sorted(activating_indices, reverse=True)]
        
        for i, token_str in tokens_to_mark:
            token_clean = token_str.strip()
            if not token_clean:
                continue
            
            # Try to find this token in the result (avoiding already marked tokens)
            # Use a simple string replacement, but be careful with overlapping tokens
            if f"<<{token_clean}>>" not in result:
                # Find the position of this token in the original decoded text
                # This is approximate but should work for most cases
                if token_clean in result:
                    # Replace the first occurrence from the right
                    # Split and find the right position
                    parts = result.rsplit(token_clean, 1)
                    if len(parts) == 2:
                        result = parts[0] + f"<<{token_clean}>>" + parts[1]
        
        return result

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
            # Get max activation value and position
            max_activation = float(example.activations.max().item())
            max_act_idx = int(example.activations.argmax().item())

            # Get tokens and string tokens
            tokens = example.tokens
            tokens_list = tokens.tolist()
            
            # Limit context window to ±window tokens around max activation
            start_idx = max(0, max_act_idx - self.window)
            end_idx = min(len(tokens_list), max_act_idx + self.window + 1)
            
            # Extract windowed tokens and activations
            windowed_tokens = tokens_list[start_idx:end_idx]
            windowed_activations = example.activations[start_idx:end_idx].tolist()
            
            # Decode windowed sequence at once to get proper text
            full_decoded = self.tokenizer.decode(windowed_tokens, skip_special_tokens=False)
            
            # Decode each token individually to identify which tokens activated
            str_tokens = [
                self.tokenizer.decode([t], skip_special_tokens=False) for t in windowed_tokens
            ]
            all_activations = windowed_activations

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
            
            # Mark activating tokens in the text with <<token>> markers
            # This makes it clear to the LLM which tokens are activating
            marked_text = self._mark_activating_tokens_in_text(
                full_decoded, str_tokens, all_activations, threshold
            )

            # Get clean text (without markers) for the main sentence list
            # The prompt expects "A list of multiple positively-activated sentences"
            # So we'll send clean sentences as the primary input
            clean_text = full_decoded.strip()

            example_dict = {
                "text": marked_text,  # Keep marked text for reference
                "text_clean": clean_text,  # Add clean text for the sentence list
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
        # The prompt expects "A list of multiple positively-activated sentences"
        # Format: Send clean sentences as a simple list, with full details in JSON
        clean_sentences = [ex["text_clean"] for ex in max_act_examples]
        
        prompt_data = {
            "feature_id": f"latent_{len(examples)}",  # Placeholder, will be set from record
            "positive_sentences": clean_sentences,  # Clean sentences as the main input
            "max_act_examples": max_act_examples,  # Full details with marked text for reference
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

        # Generate explanation with error handling to continue processing other features
        try:
            # Estimate prompt size
            prompt_text = json.dumps(messages)
            estimated_tokens = len(prompt_text) // 4  # Rough estimate: 1 token ≈ 4 chars
            logger.info(f"Generating explanation for {record.latent}: ~{estimated_tokens} input tokens, {len(examples_to_use)} examples")
            
            if estimated_tokens > 6000:
                logger.warning(f"Large prompt for {record.latent} (~{estimated_tokens} tokens). May cause timeout.")
            
            # Log the system prompt being used (first 500 chars)
            if self.verbose:
                logger.info(f"System prompt for {record.latent} (first 500 chars): {messages[0]['content'][:500]}")
            
            # Pass max_tokens explicitly to ensure enough tokens for response
            response = await self.client.generate(
                messages, 
                temperature=self.temperature,
                max_tokens=2000,  # Ensure enough tokens for JSON response
                **self.generation_kwargs
            )
            assert isinstance(response, Response)

            try:
                # Log full response for debugging
                if self.verbose:
                    logger.info(f"Full response for {record.latent}: {response.text}")
                
                explanation = self.parse_explanation(response.text)
                if self.verbose:
                    logger.info(f"Parsed explanation for {record.latent}: {explanation}")

                return ExplainerResult(record=record, explanation=explanation)
            except Exception as e:
                logger.error(f"Explanation parsing failed for {record.latent}: {repr(e)}")
                logger.error(f"Response text: {response.text if hasattr(response, 'text') else 'N/A'}")
                return ExplainerResult(
                    record=record, explanation="Explanation could not be parsed."
                )
        except Exception as e:
            # Catch all exceptions and log details
            error_type = type(e).__name__
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            
            logger.error(f"Failed to generate explanation for {record.latent}")
            logger.error(f"Error type: {error_type}")
            logger.error(f"Error message: {error_msg}")
            logger.error(f"Full traceback:\n{error_traceback}")
            
            # Check for specific error types
            if isinstance(e, RuntimeError):
                if "Failed to generate text after multiple attempts" in error_msg:
                    logger.warning(f"vLLM failed after retries for {record.latent} - check server logs")
                    # Check if it's a connection issue
                    if "connection" in error_msg.lower() or "server" in error_msg.lower():
                        error_type = "ConnectionError"
                        error_msg = "vLLM connection error - server may be down or unreachable"
                elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    logger.warning(f"vLLM timeout/connection error for {record.latent} - check server status")
                    error_type = "ConnectionError"
                    error_msg = "vLLM connection error"
            elif isinstance(e, httpx.HTTPStatusError):
                logger.error(f"HTTP error {e.response.status_code} for {record.latent}")
            elif isinstance(e, httpx.TimeoutException):
                logger.error(f"Request timeout for {record.latent} - prompt may be too large")
            elif isinstance(e, (httpx.ConnectError, httpx.RequestError)):
                logger.error(f"Connection error for {record.latent} - check network/server: {error_msg}")
                error_type = "ConnectionError"
                error_msg = "vLLM connection error - server may be down or unreachable"
            
            logger.warning(f"Continuing with fallback explanation for {record.latent}")
            return ExplainerResult(
                record=record, explanation=f"Explanation generation failed ({error_type}): {error_msg[:100]}"
            )

    def parse_explanation(self, text: str) -> str:
        """
        Parse the explanation from LLM response.

        Expected format:
        {
          "granularity": "...",
          "focus": "...",
          "label": "...",
          "reasoning": "...",
          "say_token": "..."
        }

        Returns:
            The label string (or full JSON if parsing fails).
        """
        # Log the raw response for debugging
        if self.verbose:
            logger.info(f"Raw LLM response (first 500 chars): {text[:500]}")
        
        try:
            # Try to extract JSON - look for JSON block
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

            # Fallback: Try to extract "Label:" pattern
            label_patterns = [
                r'"label"\s*:\s*"([^"]+)"',
                r'"label"\s*:\s*([^,\n}]+)',
                r'Label:\s*(.+)',
                r'label:\s*(.+)',
            ]
            for pattern in label_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    label = match.group(1).strip().strip('"').strip("'")
                    if label:
                        return label
            
            # Fallback: return cleaned text (last non-empty line)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if lines:
                return lines[-1]

            return "Explanation could not be parsed."
        except Exception as e:
            logger.error(f"Explanation parsing failed: {repr(e)}")
            return text  # Return raw text as fallback

