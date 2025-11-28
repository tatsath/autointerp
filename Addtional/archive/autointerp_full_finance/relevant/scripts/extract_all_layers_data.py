#!/usr/bin/env python3
"""
Script to extract comprehensive data from all layers for README update
"""

import json
import os
import glob
from pathlib import Path

def extract_f1_scores(detection_file):
    """Extract F1 scores from detection file"""
    try:
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return 0.0
        
        # Calculate F1 score from the data
        correct_predictions = sum(1 for item in data if item.get('correct', False))
        total_predictions = len(data)
        
        if total_predictions == 0:
            return 0.0
        
        # Calculate precision and recall
        true_positives = correct_predictions
        false_positives = total_predictions - correct_predictions
        false_negatives = 0  # We don't have this information in the current format
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return round(f1_score, 3)
    except Exception as e:
        print(f"Error processing {detection_file}: {e}")
        return 0.0

def extract_feature_label(explanation_file):
    """Extract feature label from explanation file"""
    try:
        with open(explanation_file, 'r') as f:
            content = f.read().strip()
        return content.strip('"')
    except Exception as e:
        print(f"Error reading {explanation_file}: {e}")
        return "Unknown feature"

def extract_activation_data(detection_file):
    """Extract activation statistics from detection file"""
    try:
        with open(detection_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            return {"mean_activation": 0.0, "max_activation": 0.0, "total_activations": 0}
        
        all_activations = []
        for item in data:
            if 'activations' in item:
                all_activations.extend(item['activations'])
        
        if not all_activations:
            return {"mean_activation": 0.0, "max_activation": 0.0, "total_activations": 0}
        
        mean_activation = sum(all_activations) / len(all_activations)
        max_activation = max(all_activations)
        
        return {
            "mean_activation": round(mean_activation, 4),
            "max_activation": round(max_activation, 4),
            "total_activations": len(all_activations)
        }
    except Exception as e:
        print(f"Error processing activations in {detection_file}: {e}")
        return {"mean_activation": 0.0, "max_activation": 0.0, "total_activations": 0}

def process_layer_data(base_dir, finetuned_dir, layer_name):
    """Process data for a specific layer"""
    print(f"Processing {layer_name}...")
    
    # Get all feature files for this layer
    base_detection_files = glob.glob(os.path.join(base_dir, "scores", "detection", f"layers.{layer_name}_latent*.txt"))
    finetuned_detection_files = glob.glob(os.path.join(finetuned_dir, "scores", "detection", f"layers.{layer_name}_latent*.txt"))
    
    # Get corresponding explanation files
    base_explanation_files = glob.glob(os.path.join(base_dir, "explanations", f"layers.{layer_name}_latent*.txt"))
    finetuned_explanation_files = glob.glob(os.path.join(finetuned_dir, "explanations", f"layers.{layer_name}_latent*.txt"))
    
    layer_data = {
        "layer": layer_name,
        "features": {}
    }
    
    # Process base model features
    for detection_file in base_detection_files:
        feature_id = os.path.basename(detection_file).replace(f"layers.{layer_name}_latent", "").replace(".txt", "")
        
        # Find corresponding explanation file
        explanation_file = os.path.join(base_dir, "explanations", f"layers.{layer_name}_latent{feature_id}.txt")
        
        f1_score = extract_f1_scores(detection_file)
        activation_data = extract_activation_data(detection_file)
        label = extract_feature_label(explanation_file) if os.path.exists(explanation_file) else "Unknown feature"
        
        layer_data["features"][feature_id] = {
            "base_model": {
                "f1_score": f1_score,
                "label": label,
                "mean_activation": activation_data["mean_activation"],
                "max_activation": activation_data["max_activation"]
            }
        }
    
    # Process finetuned model features
    for detection_file in finetuned_detection_files:
        feature_id = os.path.basename(detection_file).replace(f"layers.{layer_name}_latent", "").replace(".txt", "")
        
        # Find corresponding explanation file
        explanation_file = os.path.join(finetuned_dir, "explanations", f"layers.{layer_name}_latent{feature_id}.txt")
        
        f1_score = extract_f1_scores(detection_file)
        activation_data = extract_activation_data(detection_file)
        label = extract_feature_label(explanation_file) if os.path.exists(explanation_file) else "Unknown feature"
        
        if feature_id in layer_data["features"]:
            layer_data["features"][feature_id]["finetuned_model"] = {
                "f1_score": f1_score,
                "label": label,
                "mean_activation": activation_data["mean_activation"],
                "max_activation": activation_data["max_activation"]
            }
        else:
            layer_data["features"][feature_id] = {
                "finetuned_model": {
                    "f1_score": f1_score,
                    "label": label,
                    "mean_activation": activation_data["mean_activation"],
                    "max_activation": activation_data["max_activation"]
                }
            }
    
    # Calculate improvements for features present in both models
    for feature_id, feature_data in layer_data["features"].items():
        if "base_model" in feature_data and "finetuned_model" in feature_data:
            base_activation = feature_data["base_model"]["mean_activation"]
            finetuned_activation = feature_data["finetuned_model"]["mean_activation"]
            activation_improvement = finetuned_activation - base_activation
            
            base_f1 = feature_data["base_model"]["f1_score"]
            finetuned_f1 = feature_data["finetuned_model"]["f1_score"]
            f1_improvement = finetuned_f1 - base_f1
            
            feature_data["activation_improvement"] = round(activation_improvement, 4)
            feature_data["f1_improvement"] = round(f1_improvement, 3)
    
    return layer_data

def main():
    """Main function to process all layers"""
    results_dir = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/results"
    
    layers = ["4", "10", "16", "22", "28"]
    all_layers_data = {}
    
    for layer in layers:
        base_dir = os.path.join(results_dir, f"base_model_layer{layer}_all_layers")
        finetuned_dir = os.path.join(results_dir, f"finetuned_model_layer{layer}_all_layers")
        
        if os.path.exists(base_dir) and os.path.exists(finetuned_dir):
            layer_data = process_layer_data(base_dir, finetuned_dir, layer)
            all_layers_data[layer] = layer_data
        else:
            print(f"Warning: Missing directories for layer {layer}")
    
    # Save comprehensive data
    output_file = "/home/nvidia/Documents/Hariom/autointerp/autointerp_full/all_layers_comprehensive_data.json"
    with open(output_file, 'w') as f:
        json.dump(all_layers_data, f, indent=2)
    
    print(f"Comprehensive data saved to {output_file}")
    
    # Generate summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for layer, data in all_layers_data.items():
        print(f"\nLayer {layer}:")
        
        # Count features with both models
        both_models = [f for f in data["features"].values() if "base_model" in f and "finetuned_model" in f]
        print(f"  Features analyzed: {len(both_models)}")
        
        if both_models:
            # Calculate average improvements
            avg_activation_improvement = sum(f.get("activation_improvement", 0) for f in both_models) / len(both_models)
            avg_f1_improvement = sum(f.get("f1_improvement", 0) for f in both_models) / len(both_models)
            
            print(f"  Average activation improvement: {avg_activation_improvement:.4f}")
            print(f"  Average F1 improvement: {avg_f1_improvement:.3f}")
            
            # Find top improvements
            top_activation = max(both_models, key=lambda x: x.get("activation_improvement", 0))
            top_f1 = max(both_models, key=lambda x: x.get("f1_improvement", 0))
            
            print(f"  Best activation improvement: {top_activation.get('activation_improvement', 0):.4f}")
            print(f"  Best F1 improvement: {top_f1.get('f1_improvement', 0):.3f}")

if __name__ == "__main__":
    main()
