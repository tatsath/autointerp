#!/usr/bin/env python3

import json
import os
import re

def extract_f1_from_file(file_path):
    """Extract F1 score from a detection results file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Calculate F1 score from the data
        true_positives = sum(1 for item in data if item.get('correct') == True and item.get('prediction') == True)
        false_positives = sum(1 for item in data if item.get('correct') == False and item.get('prediction') == True)
        false_negatives = sum(1 for item in data if item.get('correct') == True and item.get('prediction') == False)
        
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1, precision, recall
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, 0, 0

def main():
    # Features to analyze
    features = [299, 335, 387, 347, 269, 32, 176, 209, 362, 312]
    
    # Base model results
    base_results = {}
    base_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/base_model_layer4_generic/scores/detection/"
    
    print("Base Model Results:")
    for feature in features:
        file_path = f"{base_dir}layers.4_latent{feature}.txt"
        f1, precision, recall = extract_f1_from_file(file_path)
        base_results[feature] = {'f1': f1, 'precision': precision, 'recall': recall}
        print(f"Feature {feature}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    print("\nFinetuned Model Results:")
    # Finetuned model results
    finetuned_results = {}
    finetuned_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results/finetuned_model_layer4_generic/scores/detection/"
    
    for feature in features:
        file_path = f"{finetuned_dir}layers.4_latent{feature}.txt"
        f1, precision, recall = extract_f1_from_file(file_path)
        finetuned_results[feature] = {'f1': f1, 'precision': precision, 'recall': recall}
        print(f"Feature {feature}: F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}")
    
    # Save results to JSON
    results = {
        'base_model': base_results,
        'finetuned_model': finetuned_results
    }
    
    with open('f1_scores_generic.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to f1_scores_generic.json")

if __name__ == "__main__":
    main()
