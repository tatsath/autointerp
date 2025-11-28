#!/usr/bin/env python3
"""Add labels and F1 scores to top features from CSV lookup"""

import json
import csv
import sys
import os

# Paths - updated to use results/ subfolder
script_dir = os.path.dirname(os.path.abspath(__file__))
scores_file = os.path.join(script_dir, "results", "top_features_scores.json")
csv_file = "/home/nvidia/Documents/Hariom/InterpUseCases_autointerp/Autointerp_clustering/Archive/small_sae_autointerp_results/small_sae_results_summary.csv"
output_file = os.path.join(script_dir, "results", "top_features_with_labels.json")

# Load top features
if not os.path.exists(scores_file):
    print(f"Error: {scores_file} not found. Run feature search first.")
    sys.exit(1)

with open(scores_file, 'r') as f:
    data = json.load(f)

# Load CSV mapping
feature_map = {}
if os.path.exists(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_num = int(row['feature'])
            feature_map[feature_num] = {
                'label': row['label'],
                'f1_score': float(row['f1_score'])
            }
else:
    print(f"Warning: CSV file {csv_file} not found. Labels will be 'N/A'.")

# Match features with labels
results = []
for idx, feat_idx in enumerate(data['feature_indices']):
    if feat_idx in feature_map:
        results.append({
            'feature_index': feat_idx,
            'score': data['scores'][idx],
            'label': feature_map[feat_idx]['label'],
            'f1_score': feature_map[feat_idx]['f1_score']
        })
    else:
        results.append({
            'feature_index': feat_idx,
            'score': data['scores'][idx],
            'label': 'N/A',
            'f1_score': None
        })

# Save output
output_data = {
    'quantile_threshold': data['quantile_threshold'],
    'quantile_value': data['quantile_value'],
    'features': results
}

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✅ Saved {len(results)} features with labels to {output_file}")


