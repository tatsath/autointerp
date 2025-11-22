#!/usr/bin/env python3
"""
Generate enhanced CSV summary of AutoInterp Full results with F1 score and accuracy
"""
import json
import csv
import os
from pathlib import Path

def extract_metrics(score_file):
    """Extract F1 score and accuracy from the detection score file"""
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Calculate accuracy from correct field
                correct = sum(1 for item in data if item.get('correct', False))
                total = len(data)
                accuracy = correct / total if total > 0 else 0
                
                # Calculate F1 score from predictions
                true_positives = sum(1 for item in data if item.get('correct', False) and item.get('prediction', False))
                false_positives = sum(1 for item in data if not item.get('correct', False) and item.get('prediction', False))
                false_negatives = sum(1 for item in data if not item.get('correct', False) and not item.get('prediction', False))
                
                # Calculate precision and recall
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                # Calculate F1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                return {
                    'accuracy': round(accuracy, 3),
                    'f1_score': round(f1_score, 3),
                    'precision': round(precision, 3),
                    'recall': round(recall, 3)
                }
            return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None

def generate_csv(results_dir="results/llm_api_example"):
    """Generate enhanced CSV file with results summary including accuracy"""
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
            # Format: layers.16_latent133.txt
            filename = explanation_file.stem
            if "layers." in filename and "latent" in filename:
                parts = filename.split("_")
                layer_part = parts[0]  # layers.16
                feature_part = parts[1]  # latent133
                
                layer_num = layer_part.split(".")[1]  # 16
                feature_num = feature_part.replace("latent", "")  # 133
                
                # Read explanation
                try:
                    with open(explanation_file, 'r') as f:
                        label = f.read().strip().strip('"')
                except:
                    label = "Error reading explanation"
                
                # Get metrics
                score_file = scores_dir / explanation_file.name
                metrics = extract_metrics(score_file) if score_file.exists() else None
                
                if metrics:
                    csv_data.append({
                        'layer': layer_num,
                        'feature': feature_num,
                        'label': label,
                        'f1_score': metrics['f1_score'],
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall']
                    })
                else:
                    csv_data.append({
                        'layer': layer_num,
                        'feature': feature_num,
                        'label': label,
                        'f1_score': None,
                        'accuracy': None,
                        'precision': None,
                        'recall': None
                    })
    
    # Write CSV file
    if csv_data:
        csv_file = results_path / "results_summary.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['layer', 'feature', 'label', 'f1_score', 'accuracy', 'precision', 'recall'])
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"✅ Enhanced CSV summary generated: {csv_file}")
        print(f"📊 Found {len(csv_data)} features with results")
        
        # Print summary
        for row in csv_data:
            print(f"  Feature {row['feature']}: {row['label']} (F1: {row['f1_score']}, Acc: {row['accuracy']})")
    else:
        print("❌ No results found to generate CSV")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results/llm_api_example"
    generate_csv(results_dir)
