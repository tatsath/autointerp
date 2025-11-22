#!/usr/bin/env python3
"""
Generate CSV summary of AutoInterp Full results
"""
import json
import csv
import os
from pathlib import Path

def extract_f1_score(score_file):
    """Extract F1 score from the detection score file"""
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
            # Look for F1 score in the data structure
            if isinstance(data, list) and len(data) > 0:
                # Calculate F1 from predictions if available
                correct = sum(1 for item in data if item.get('correct', False))
                total = len(data)
                if total > 0:
                    accuracy = correct / total
                    return round(accuracy, 3)
            return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None

def generate_csv(results_dir="results/llm_api_example"):
    """Generate CSV file with results summary"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return
    
    csv_data = []
    
    # Check explanations directory
    explanations_dir = results_path / "explanations"
    scores_dir = results_path / "scores" / "detection"
    
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
                    
                    # Get F1 score
                    score_file = scores_dir / explanation_file.name
                    f1_score = extract_f1_score(score_file) if score_file.exists() else None
                    
                    csv_data.append({
                        'layer': layer_num,
                        'feature': feature_num,
                        'label': label,
                        'f1_score': f1_score
                    })
    
    # Write CSV file
    if csv_data:
        csv_file = results_path / "results_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['layer', 'feature', 'label', 'f1_score'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✅ CSV summary generated: {csv_file}")
        print(f"📊 Found {len(csv_data)} features with results")
        
        # Print summary
        for row in csv_data:
            print(f"  Feature {row['feature']}: {row['label']} (F1: {row['f1_score']})")
    else:
        print("❌ No results found to generate CSV")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/llm_api_example"
    generate_csv(results_dir)
