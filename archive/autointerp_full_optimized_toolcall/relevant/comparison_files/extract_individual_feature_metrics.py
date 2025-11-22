#!/usr/bin/env python3
"""
Extract individual feature metrics from AutoInterp results
"""

import json
import os
import glob
from pathlib import Path

def extract_feature_metrics(results_dir, features):
    """Extract metrics for specific features from AutoInterp results"""
    feature_metrics = {}
    
    # Find all detection score files
    detection_files = glob.glob(os.path.join(results_dir, "scores", "detection", "*.txt"))
    
    for file_path in detection_files:
        # Extract feature number from filename
        filename = os.path.basename(file_path)
        if "latent" in filename:
            feature_num = int(filename.split("latent")[1].split(".")[0])
            
            if feature_num in features:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Calculate metrics from the data
                    total_examples = len(data)
                    correct_predictions = sum(1 for item in data if item.get('correct', False))
                    activating_examples = sum(1 for item in data if item.get('activating', False))
                    
                    # Calculate precision, recall, F1
                    true_positives = sum(1 for item in data if item.get('activating', False) and item.get('correct', False))
                    false_positives = sum(1 for item in data if item.get('activating', False) and not item.get('correct', False))
                    false_negatives = sum(1 for item in data if not item.get('activating', False) and item.get('correct', False))
                    
                    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    feature_metrics[feature_num] = {
                        'total_examples': total_examples,
                        'correct_predictions': correct_predictions,
                        'activating_examples': activating_examples,
                        'true_positives': true_positives,
                        'false_positives': false_positives,
                        'false_negatives': false_negatives,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    }
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    return feature_metrics

def main():
    # Define the features we're analyzing (from finetuning impact analysis)
    features = [299, 335, 387, 347, 269, 32, 176, 209, 362, 312]
    
    # Base model results
    base_model_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/base_model_layer4"
    base_metrics = extract_feature_metrics(base_model_dir, features)
    
    # Finetuned model results  
    finetuned_model_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/finetuned_model_layer4"
    finetuned_metrics = extract_feature_metrics(finetuned_model_dir, features)
    
    # Print results
    print("Individual Feature Metrics Comparison")
    print("=" * 80)
    print(f"{'Feature':<8} {'Model':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'TP':<4} {'FP':<4} {'FN':<4}")
    print("-" * 80)
    
    for feature in features:
        if feature in base_metrics and feature in finetuned_metrics:
            base = base_metrics[feature]
            finetuned = finetuned_metrics[feature]
            
            print(f"{feature:<8} {'Base':<12} {base['f1']:<8.3f} {base['precision']:<10.3f} {base['recall']:<8.3f} {base['true_positives']:<4} {base['false_positives']:<4} {base['false_negatives']:<4}")
            print(f"{'':8} {'Finetuned':<12} {finetuned['f1']:<8.3f} {finetuned['precision']:<10.3f} {finetuned['recall']:<8.3f} {finetuned['true_positives']:<4} {finetuned['false_positives']:<4} {finetuned['false_negatives']:<4}")
            print()
    
    # Save to JSON for README update
    results = {
        'base_model': base_metrics,
        'finetuned_model': finetuned_metrics
    }
    
    with open('individual_feature_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to individual_feature_metrics.json")

if __name__ == "__main__":
    main()
