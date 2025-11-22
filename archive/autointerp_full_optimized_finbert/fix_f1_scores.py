#!/usr/bin/env python3
"""
Fix F1 scores for features where all examples are activating.
For topk SAE, if a feature is in top-k for most positions, we have no non-activating examples.
This script adjusts F1 calculation to handle this case.
"""

import json
import sys
from pathlib import Path
import pandas as pd

def compute_f1_with_all_activating(predictions, activating_labels):
    """
    Compute F1 when all examples are activating (no negatives).
    In this case:
    - TP = correct positive predictions
    - FP = 0 (no negatives to misclassify)
    - FN = missed positives (predicted False but should be True)
    - TN = 0 (no negatives)
    """
    if not predictions:
        return 0.0, 0.0, 0.0  # precision, recall, f1
    
    tp = sum(1 for p, a in zip(predictions, activating_labels) if p and a)
    fp = sum(1 for p, a in zip(predictions, activating_labels) if p and not a)
    fn = sum(1 for p, a in zip(predictions, activating_labels) if not p and a)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def fix_score_file(score_file: Path):
    """Fix a single score file by computing F1 correctly."""
    with open(score_file) as f:
        data = json.load(f)
    
    # Filter to examples with predictions
    with_predictions = [x for x in data if x.get("prediction") is not None]
    
    if not with_predictions:
        print(f"  ⚠️  No predictions found, skipping")
        return None
    
    activating_labels = [x.get("activating", False) for x in with_predictions]
    predictions = [x.get("prediction", False) for x in with_predictions]
    
    # Check if all are activating
    all_activating = all(activating_labels)
    all_non_activating = not any(activating_labels)
    
    if all_activating:
        precision, recall, f1 = compute_f1_with_all_activating(predictions, activating_labels)
        print(f"  📊 All activating: TP={sum(1 for p, a in zip(predictions, activating_labels) if p and a)}, "
              f"FN={sum(1 for p, a in zip(predictions, activating_labels) if not p and a)}, "
              f"F1={f1:.4f}")
    elif all_non_activating:
        # All non-activating case
        tn = sum(1 for p, a in zip(predictions, activating_labels) if not p and not a)
        fp = sum(1 for p, a in zip(predictions, activating_labels) if p and not a)
        precision = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        recall = 1.0  # All negatives correctly identified
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  📊 All non-activating: TN={tn}, FP={fp}, F1={f1:.4f}")
    else:
        # Mixed case - standard calculation
        tp = sum(1 for p, a in zip(predictions, activating_labels) if p and a)
        fp = sum(1 for p, a in zip(predictions, activating_labels) if p and not a)
        fn = sum(1 for p, a in zip(predictions, activating_labels) if not p and a)
        tn = sum(1 for p, a in zip(predictions, activating_labels) if not p and not a)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"  📊 Mixed: TP={tp}, FP={fp}, FN={fn}, TN={tn}, F1={f1:.4f}")
    
    return f1

def main():
    results_dir = Path("results/finbert_layer10_all_features_action")
    scores_dir = results_dir / "scores" / "detection"
    
    if not scores_dir.exists():
        print(f"❌ Scores directory not found: {scores_dir}")
        sys.exit(1)
    
    print("🔧 Fixing F1 scores for detection scorer...")
    print("="*60)
    
    score_files = list(scores_dir.glob("*.txt"))
    print(f"Found {len(score_files)} score files\n")
    
    f1_scores = {}
    for score_file in score_files:
        latent_id = int(score_file.stem.split("latent")[-1])
        print(f"Feature {latent_id}:")
        f1 = fix_score_file(score_file)
        if f1 is not None:
            f1_scores[latent_id] = f1
        print()
    
    # Update CSV
    csv_file = results_dir / "results_summary.csv"
    if csv_file.exists():
        print(f"📝 Updating {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Update F1 scores
        for idx, row in df.iterrows():
            feature_id = int(row['feature'])
            if feature_id in f1_scores:
                old_f1 = df.at[idx, 'f1_score']
                new_f1 = f1_scores[feature_id]
                df.at[idx, 'f1_score'] = new_f1
                df.at[idx, 'detection_f1'] = new_f1
                df.at[idx, 'f1_good'] = new_f1 > 0.3  # Threshold for "good"
                df.at[idx, 'detection_f1_good'] = new_f1 > 0.3
                print(f"  Feature {feature_id}: {old_f1:.4f} → {new_f1:.4f}")
        
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Updated CSV with corrected F1 scores")
    else:
        print(f"⚠️  CSV file not found: {csv_file}")
    
    print(f"\n✅ Fixed F1 scores for {len(f1_scores)} features")

if __name__ == "__main__":
    main()





