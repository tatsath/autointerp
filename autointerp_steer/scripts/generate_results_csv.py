#!/usr/bin/env python3
"""
Generate CSV summary from interpretations JSON file.
Creates a CSV with columns: layer,feature,label,steering_effect
Uses steering effect score (based on text variance) to measure feature impact.
"""
import json
import csv
import sys
from pathlib import Path

def calculate_steering_effect_score(steering_data: dict) -> float:
    """
    Calculate steering effect score based on how much outputs vary with steering.
    
    This measures how much the feature affects generation when steered.
    Higher score = more steering effect = more interpretable/important feature.
    
    Args:
        steering_data: Dictionary with {prompt: {original: str, strength: str, ...}}
    
    Returns:
        Score between 0.0 and 1.0 (normalized)
    """
    if not steering_data:
        return 0.0
    
    # Collect all texts (original + steered at different strengths)
    all_texts = []
    for prompt_data in steering_data.values():
        if isinstance(prompt_data, dict):
            original = prompt_data.get('original', '')
            if original:
                all_texts.append(original)
            
            # Get steered texts at various strengths (supports both old 14-level and new 4-level)
            # Check for new 4-level first, fallback to old strengths
            steering_strengths = ['-2.0', '-1.0', '1.0', '2.0', 
                                  '-4.0', '-3.0', '-2.5', '-2.0', '-1.5', '-1.0', '-0.5', 
                                  '0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '4.0']
            for strength in steering_strengths:
                steered = prompt_data.get(strength, '')
                if steered:
                    all_texts.append(steered)
    
    if len(all_texts) < 2:
        return 0.0
    
    # Calculate average text difference/variance
    # Simple approach: count unique texts and measure diversity
    unique_texts = set(all_texts)
    
    # Normalize: if all texts are identical, score = 0; if all different, score = 1
    # But we also consider how different they are (not just count)
    
    # More sophisticated: calculate average token-level differences
    from collections import Counter
    
    # Tokenize and calculate diversity
    all_tokens = []
    for text in all_texts:
        tokens = text.split()[:50]  # Limit to first 50 words
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return 0.0
    
    # Calculate token diversity (unique tokens / total tokens)
    token_counter = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counter)
    
    # Diversity ratio (higher = more diverse outputs = stronger steering effect)
    diversity = unique_tokens / total_tokens if total_tokens > 0 else 0.0
    
    # Also consider text uniqueness ratio
    uniqueness_ratio = len(unique_texts) / len(all_texts) if all_texts else 0.0
    
    # Combined score: weighted average of diversity metrics
    score = (diversity * 0.6 + uniqueness_ratio * 0.4)
    
    return min(1.0, max(0.0, score))


def generate_csv_from_interpretations(json_file: str, steering_outputs_dir: str = None, output_csv: str = None):
    """
    Convert interpretations JSON to CSV format.
    
    Args:
        json_file: Path to interpretations JSON file
        steering_outputs_dir: Optional path to steering outputs directory (for calculating steering effect score)
        output_csv: Path to output CSV file (default: same name as JSON but .csv)
    """
    json_path = Path(json_file)
    
    if not json_path.exists():
        print(f"❌ Error: JSON file not found: {json_file}")
        return False
    
    # Default output CSV name
    if output_csv is None:
        output_csv = json_path.with_suffix('_summary.csv')
    else:
        output_csv = Path(output_csv)
    
    print(f"📊 Generating CSV from: {json_file}")
    
    try:
        # Load interpretations
        with open(json_path, 'r') as f:
            interpretations = json.load(f)
        
        # Load steering outputs if available (for calculating steering effect score)
        steering_data_cache = {}
        if steering_outputs_dir:
            steering_dir = Path(steering_outputs_dir)
            if steering_dir.exists():
                try:
                    # Add parent directory to path for imports
                    import sys
                    script_dir = Path(__file__).parent.parent
                    if str(script_dir) not in sys.path:
                        sys.path.insert(0, str(script_dir))
                    from sae_pipeline.feature_interpreter import load_steering_outputs
                    steering_data_cache = load_steering_outputs(str(steering_dir))
                    print(f"✓ Loaded steering outputs from {steering_outputs_dir}")
                except Exception as e:
                    print(f"⚠️  Could not load steering outputs: {e}")
                    print("   Will calculate scores based on interpretation status")
        
        # Prepare CSV rows
        rows = []
        for layer, layer_data in interpretations.items():
            for feature_id, feature_data in layer_data.items():
                if isinstance(feature_data, dict):
                    label = feature_data.get('interpretation', '')
                    status = feature_data.get('status', 'error')
                    
                    # Calculate steering effect score if steering data available
                    if steering_data_cache:
                        layer_str = str(layer)
                        feature_str = str(feature_id)
                        if (layer_str in steering_data_cache and 
                            feature_str in steering_data_cache[layer_str]):
                            steering_data = steering_data_cache[layer_str][feature_str]
                            steering_score = calculate_steering_effect_score(steering_data)
                        else:
                            steering_score = 0.0
                    else:
                        # Use status-based placeholder
                        steering_score = 0.5 if status == 'success' else 0.0
                    
                    # Extract short label (max 20 words) from interpretation
                    try:
                        # Add parent directory to path for imports
                        import sys
                        from pathlib import Path as PathLib
                        script_dir = PathLib(__file__).parent.parent
                        if str(script_dir) not in sys.path:
                            sys.path.insert(0, str(script_dir))
                        from sae_pipeline.feature_interpreter import extract_short_label
                        label = extract_short_label(label)
                    except Exception as e:
                        # Fallback: clean up and truncate to 20 words
                        print(f"Warning: Could not extract short label: {e}, using fallback")
                        label = ' '.join(label.split())
                        words = label.split()
                        if len(words) > 20:
                            label = ' '.join(words[:20])
                    
                    rows.append({
                        'layer': int(layer),
                        'feature': int(feature_id),
                        'label': label,
                        'steering_effect': round(steering_score, 3)  # Round to 3 decimals
                    })
        
        # Sort by layer, then feature
        rows.sort(key=lambda x: (x['layer'], x['feature']))
        
        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['layer', 'feature', 'label', 'steering_effect'])
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ CSV generated: {output_csv}")
        print(f"   Total features: {len(rows)}")
        print(f"   Successful interpretations: {sum(1 for r in rows if r['label'] and r['label'] != 'Error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating CSV: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_results_csv.py <interpretations.json> [steering_outputs_dir] [output.csv]")
        print("  interpretations.json: JSON file with feature interpretations")
        print("  steering_outputs_dir: (optional) Directory with steering outputs for calculating scores")
        print("  output.csv: (optional) Output CSV file path")
        sys.exit(1)
    
    json_file = sys.argv[1]
    steering_outputs_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith('.csv') else None
    output_csv = sys.argv[3] if len(sys.argv) > 3 else (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2].endswith('.csv') else None)
    
    success = generate_csv_from_interpretations(json_file, steering_outputs_dir, output_csv)
    sys.exit(0 if success else 1)

