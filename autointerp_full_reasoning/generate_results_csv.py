#!/usr/bin/env python3
"""
Generate CSV summary of AutoInterp Full results matching topk_sae_results_summary.csv format
Format: layer,feature,label,f1_score,detection_f1,fuzz_f1,f1_good,detection_f1_good,fuzz_good
"""
import json
import csv
import os
from pathlib import Path

def extract_f1_from_scorer(score_file):
    """Extract F1 score from a scorer file (detection or fuzz)"""
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list) and len(data) > 0:
            # Calculate F1 score from predictions
            true_positives = sum(1 for item in data if item.get('correct', False) and item.get('prediction', False))
            false_positives = sum(1 for item in data if item.get('prediction', False) and not item.get('correct', False))
            false_negatives = sum(1 for item in data if item.get('activating', False) and not item.get('prediction', False))
            
            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            # Calculate F1 score
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return round(f1_score, 4)
        return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None

def generate_csv(results_dir="results/llm_api_example"):
    """Generate CSV file with results summary matching topk_sae_results_summary.csv format"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return
    
    csv_data = []
    
    # Check explanations directory
    explanations_dir = results_path / "explanations"
    detection_scores_dir = results_path / "scores" / "detection"
    fuzz_scores_dir = results_path / "scores" / "fuzz"
    
    if explanations_dir.exists():
        for explanation_file in explanations_dir.glob("*.txt"):
            # Extract layer and feature number from filename
            # Format: layers.16_latent133.txt or encoder.layer.2_latent0.txt
            filename = explanation_file.stem
            if "latent" in filename:
                parts = filename.split("_")
                if len(parts) >= 2:
                    layer_part = parts[0]  # layers.16 or encoder.layer.2
                    feature_part = parts[1]  # latent133 or latent0
                    
                    # Handle both formats: layers.X or encoder.layer.X
                    if "encoder.layer" in layer_part:
                        # Format: encoder.layer.2 -> extract 2
                        layer_num = layer_part.split(".")[-1]
                    elif "layers." in layer_part:
                        # Format: layers.16 -> extract 16
                        layer_num = layer_part.split(".")[1]
                    else:
                        continue  # Skip if format doesn't match
                    
                    feature_num = feature_part.replace("latent", "")
                    
                    # Read explanation
                    try:
                        with open(explanation_file, 'r') as f:
                            label = f.read().strip().strip('"')
                    except:
                        label = "Error reading explanation"
                    
                    # Get F1 scores from detection and fuzz scorers
                    detection_score_file = detection_scores_dir / explanation_file.name
                    fuzz_score_file = fuzz_scores_dir / explanation_file.name
                    
                    detection_f1 = extract_f1_from_scorer(detection_score_file) if detection_score_file.exists() else None
                    fuzz_f1 = extract_f1_from_scorer(fuzz_score_file) if fuzz_score_file.exists() else None
                    
                    # Calculate overall f1_score (average of detection and fuzz, or just detection if fuzz not available)
                    if detection_f1 is not None and fuzz_f1 is not None:
                        f1_score = round((detection_f1 + fuzz_f1) / 2, 4)
                    elif detection_f1 is not None:
                        f1_score = detection_f1
                    elif fuzz_f1 is not None:
                        f1_score = fuzz_f1
                    else:
                        f1_score = None
                    
                    # Determine "good" flags (threshold: >= 0.7 for good)
                    f1_good = f1_score >= 0.7 if f1_score is not None else False
                    detection_f1_good = detection_f1 >= 0.7 if detection_f1 is not None else False
                    fuzz_good = fuzz_f1 >= 0.7 if fuzz_f1 is not None else False
                    
                    csv_data.append({
                        'layer': layer_num,
                        'feature': feature_num,
                        'label': label,
                        'f1_score': f1_score,
                        'detection_f1': detection_f1,
                        'fuzz_f1': fuzz_f1,
                        'f1_good': f1_good,
                        'detection_f1_good': detection_f1_good,
                        'fuzz_good': fuzz_good
                    })
    
    # Sort by feature number (as integer)
    csv_data.sort(key=lambda x: int(x['feature']))
    
    # Write CSV file
    if csv_data:
        csv_file = results_path / "results_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['layer', 'feature', 'label', 'f1_score', 'detection_f1', 'fuzz_f1', 'f1_good', 'detection_f1_good', 'fuzz_good'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✅ CSV summary generated: {csv_file}")
        print(f"📊 Found {len(csv_data)} features with results")
        
        # Print summary
        for row in csv_data[:5]:  # Print first 5
            print(f"  Feature {row['feature']}: {row['label'][:50]}... (F1: {row['f1_score']}, Det: {row['detection_f1']}, Fuzz: {row['fuzz_f1']})")
    else:
        print("❌ No results found to generate CSV")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/llm_api_example"
    generate_csv(results_dir)
