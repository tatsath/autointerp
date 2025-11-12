"""
Feature interpretation using LLM analysis of steering outputs.
Based on Kuznetsov et al. (2025) methodology.
"""
import json
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional


STEERING_PROMPT_TEMPLATE = """You are analyzing steering outputs from a Sparse Autoencoder (SAE) feature.

Your task is to analyze how steering a specific feature affects the generated text and determine what SPECIFIC, DISTINCT function this feature likely serves.

IMPORTANT OUTPUT FORMAT:
1. First provide a SHORT summary (one phrase, maximum 20 words) that is SPECIFIC and DISTINCTIVE
2. Focus on the most specific aspect that makes this feature unique - what specific semantic, stylistic, or structural element does it control?
3. Be concrete and specific, not generic. Avoid vague terms like "influences" or "affects" unless necessary.

The summary should be concise, descriptive, and specific, similar to examples like:
- "Financial and market-related terms and metrics"
- "Temporal references and historical context in narratives"
- "Medical terminology and clinical language patterns"
- "Logical connectors and argument structure"

Format your response as:
SUMMARY: [your specific short phrase here, max 20 words]

DETAILED ANALYSIS:
[Your detailed explanation here...]

---

Feature: {feature_number}
Layer: {layer}

Steering Examples:

{steering_examples}

Analyze this feature carefully. Identify the MOST SPECIFIC and DISTINCTIVE aspect of what this feature controls. Start with "SUMMARY: [specific short phrase]" then provide detailed analysis explaining why this label is accurate.
"""


def build_steering_prompt(
    feature_number: int,
    layer: int,
    steering_data: Dict,
    num_examples: int = 2  # Reduced from 5 to 2 for shorter prompts
) -> str:
    """
    Build prompt for LLM analysis of steering outputs.
    
    Args:
        feature_number: Feature ID to analyze
        layer: Layer number
        steering_data: Dictionary containing steering outputs for this feature
            Format: {prompt: {original: str, -4.0: str, -3.0: str, ..., 4.0: str}}
        num_examples: Number of prompt examples to include
    
    Returns:
        Formatted prompt string
    """
    # Select a few representative prompts
    prompts = list(steering_data.keys())[:num_examples]
    
    examples_text = []
    for i, prompt in enumerate(prompts, 1):
        prompt_data = steering_data[prompt]
        original = prompt_data.get('original', 'N/A')
        
        # Select key steering strengths to show pattern (using 4 levels: -2, -1, 1, 2)
        key_strengths = [-2.0, -1.0, 1.0, 2.0]
        
        # Truncate texts to first 200 chars to keep prompt manageable
        def truncate(text, max_len=200):
            if isinstance(text, str) and len(text) > max_len:
                return text[:max_len] + "..."
            return text
        
        example_lines = [f"\nExample {i}:"]
        example_lines.append(f"Original Prompt: {truncate(str(prompt), 150)}")
        example_lines.append(f"Original Text: {truncate(original, 200)}")
        
        for strength in key_strengths:
            steered_text = prompt_data.get(str(strength), 'N/A')
            example_lines.append(f"\nSteering Strength {strength:+1.1f}: {truncate(steered_text, 200)}")
        
        examples_text.append("\n".join(example_lines))
    
    steering_examples = "\n\n---\n".join(examples_text)
    
    return STEERING_PROMPT_TEMPLATE.format(
        feature_numbers=str([feature_number]),
        feature_number=feature_number,
        layer=layer,
        steering_examples=steering_examples
    )


def load_steering_outputs(output_folder: str) -> Dict:
    """
    Load all steering output JSON files.
    
    Args:
        output_folder: Directory containing steering output JSON files
    
    Returns:
        Dictionary organized as {layer: {feature_id: {prompt: {strength: text}}}}
    """
    results = {}
    output_path = Path(output_folder)
    
    if not output_path.exists():
        raise FileNotFoundError(f"Output folder not found: {output_folder}")
    
    # Find all JSON files
    json_files = list(output_path.glob("generated_texts_*.json"))
    
    if not json_files:
        raise FileNotFoundError(
            f"No steering output JSON files found in {output_folder}. "
            "Run steering experiments first using scripts/run_steering.py"
        )
    
    # Load and merge all JSON files
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Merge into results
            for layer, layer_data in data.items():
                if layer not in results:
                    results[layer] = {}
                for feature_id, feature_data in layer_data.items():
                    if feature_id not in results[layer]:
                        results[layer][feature_id] = {}
                    for prompt, prompt_data in feature_data.items():
                        if prompt not in results[layer][feature_id]:
                            results[layer][feature_id][prompt] = prompt_data
    
    return results


def extract_short_label(interpretation_text: str) -> str:
    """
    Extract a short label (max 20 words) from LLM interpretation.
    Looks for "SUMMARY:" prefix or extracts first sentence/phrase.
    
    Args:
        interpretation_text: Full interpretation text from LLM
    
    Returns:
        Short label string (max 20 words)
    """
    if not interpretation_text:
        return "No interpretation available"
    
    # Remove markdown code blocks
    text = interpretation_text
    text = text.replace("```markdown", "").replace("```plaintext", "").replace("```", "")
    
    # Look for "SUMMARY:" prefix (case insensitive)
    if "SUMMARY:" in text.upper():
        # Find SUMMARY: and take everything until DETAILED or end of first line/sentence
        summary_start_idx = text.upper().find("SUMMARY:")
        # Find the actual case-sensitive start
        for i in range(summary_start_idx, len(text)):
            if text[i:i+8].upper() == "SUMMARY:":
                summary_start_idx = i
                break
        
        summary_text = text[summary_start_idx + 8:].strip()  # Skip "SUMMARY:"
        
        # Stop at "DETAILED ANALYSIS" or double newline or end of first line
        if "DETAILED ANALYSIS" in summary_text.upper() or "\n\n" in summary_text:
            # Take first part before double newline or DETAILED
            if "\n\n" in summary_text:
                summary_text = summary_text.split("\n\n")[0].strip()
            elif "DETAILED" in summary_text.upper():
                summary_text = summary_text[:summary_text.upper().find("DETAILED")].strip()
        elif "\n" in summary_text:
            # Take first line only
            summary_text = summary_text.split("\n")[0].strip()
        
        # Remove any trailing periods that might be followed by newline
        summary_text = summary_text.strip().rstrip('.')
        
        # Clean up and take first 20 words
        words = summary_text.split()
        if len(words) > 20:
            return " ".join(words[:20])
        return summary_text
    
    # Extract first sentence or first meaningful phrase
    # Remove table formatting and markdown
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.startswith("|")]
    
    # Find first substantive line (not headers, not empty)
    for line in lines:
        # Skip markdown headers, code blocks, etc.
        if any(skip in line.lower() for skip in ["feature number", "possible function", "effect type", "observed behavior", "---", "#", "summary:", "detailed"]):
            continue
        if len(line) > 10:  # Substantive line
            words = line.split()
            if len(words) <= 20:
                return line
            else:
                return " ".join(words[:20])
    
    # Fallback: take first 20 words of cleaned text
    words = text.split()
    return " ".join(words[:20])


async def interpret_feature(
    client,
    feature_number: int,
    layer: int,
    steering_data: Dict,
    max_retries: int = 3
) -> Dict:
    """
    Use LLM to interpret a single feature based on steering outputs.
    
    Args:
        client: LLM client instance (duck-typed: just needs .generate(messages) method)
        feature_number: Feature ID to interpret
        layer: Layer number
        steering_data: Steering outputs for this feature
        max_retries: Maximum retry attempts
    
    Returns:
        Dictionary with interpretation results
    """
    prompt = build_steering_prompt(feature_number, layer, steering_data)
    
    # Format as chat messages
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(max_retries):
        try:
            response = await client.generate(messages, max_tokens=2000, temperature=0.7)
            
            if isinstance(response, str):
                interpretation_text = response
            else:
                interpretation_text = response.text
            
            return {
                "feature_number": feature_number,
                "layer": layer,
                "interpretation": interpretation_text,
                "status": "success"
            }
        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "feature_number": feature_number,
                    "layer": layer,
                    "interpretation": f"Error: {str(e)}",
                    "status": "error"
                }
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return {
        "feature_number": feature_number,
        "layer": layer,
        "interpretation": "Failed after all retries",
        "status": "error"
    }


async def interpret_all_features(
    steering_outputs: Dict,
    client,
    output_dir: str,
    max_features: Optional[int] = None,
    layers: Optional[List[int]] = None
):
    """
    Interpret all features in steering outputs.
    
    Args:
        steering_outputs: Loaded steering outputs from load_steering_outputs()
        client: LLM client instance (duck-typed: just needs .generate(messages) method)
        output_dir: Directory where to save interpretations JSON file
        max_features: Optional limit on how many features to interpret per layer
        layers: Optional list of layers to process (None = all layers)
    
    Returns:
        Dictionary of interpretations organized by layer and feature
    """
    all_interpretations = {}
    
    # Filter layers if specified
    # Convert steering_outputs keys to int for comparison (JSON keys are strings)
    steering_layers = {int(k): v for k, v in steering_outputs.items() if str(k).isdigit()}
    
    layers_to_process = layers if layers else list(steering_layers.keys())
    
    for layer in layers_to_process:
        if layer not in steering_layers:
            print(f"Warning: Layer {layer} not found in steering outputs")
            continue
        
        print(f"\n=== Interpreting Layer {layer} ===")
        all_interpretations[layer] = {}
        
        layer_data = steering_layers[layer]
        
        # Sort features and optionally limit
        sorted_features = sorted(layer_data.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) and str(x).isdigit() else 0)
        if max_features:
            sorted_features = sorted_features[:max_features]
        
        for feature_id in sorted_features:
            print(f"  Interpreting feature {feature_id}...")
            
            result = await interpret_feature(
                client,
                int(feature_id),
                int(layer),
                layer_data[feature_id]
            )
            
            all_interpretations[layer][feature_id] = result
            
            if result["status"] == "success":
                print(f"    ✓ Success")
            else:
                print(f"    ✗ Error: {result['interpretation']}")
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "interpretations.json")
    with open(output_file, 'w') as f:
        json.dump(all_interpretations, f, indent=2)
    
    print(f"\n✓ Interpretations saved to {output_file}")
    
    return all_interpretations

