#!/usr/bin/env python3
"""
Generate enhanced CSV summary for AutoInterp results.
Format: layer,feature,label,f1_score,detection_f1,fuzz_f1,f1_good,detection_f1_good,fuzz_good.
Handles backbone.layers.X format filenames.
"""
import argparse
import json
import csv
import os
from pathlib import Path


def calculate_classification_metrics(score_data):
    """Calculate F1 from classification score data
    Uses the same method as generate_results_csv.py for consistency
    """
    if not isinstance(score_data, list) or len(score_data) == 0:
        return None
    
    # Count different types - using simpler method that matches generate_results_csv.py
    # TP: prediction=True AND correct=True (regardless of activating status)
    true_positives = sum(1 for item in score_data if item.get('correct', False) and item.get('prediction', False))
    # FP: prediction=True BUT correct=False
    false_positives = sum(1 for item in score_data if item.get('prediction', False) and not item.get('correct', False))
    # FN: activating=True BUT prediction=False
    false_negatives = sum(1 for item in score_data if item.get('activating', False) and not item.get('prediction', False))
    
    # Calculate F1
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return round(f1, 4)


def extract_metrics_from_file(score_file, scorer_type):
    """Extract F1 metric from a score file"""
    try:
        with open(score_file, 'r') as f:
            data = json.load(f)
        
        if scorer_type in ['detection', 'fuzz']:
            return calculate_classification_metrics(data)
        else:
            return None
    except Exception as e:
        print(f"Error reading {score_file}: {e}")
        return None


def check_thresholds(f1_score, detection_f1, fuzz_f1):
    """Check if metrics meet threshold criteria"""
    # f1_score threshold: >= 0.70
    f1_good = f1_score is not None and float(f1_score) >= 0.70
    
    # detection_f1 threshold: >= 0.70
    detection_f1_good = detection_f1 is not None and float(detection_f1) >= 0.70
    
    # fuzz_f1 threshold: <= 0.30 (lower is better)
    fuzz_good = fuzz_f1 is not None and float(fuzz_f1) <= 0.30
    
    return {
        'f1_good': f1_good,
        'detection_f1_good': detection_f1_good,
        'fuzz_good': fuzz_good,
    }


def generate_enhanced_csv(results_dir, output_name="nemotron_finance_results_summary_enhanced.csv"):
    """Generate enhanced CSV file with results summary matching topk_sae_results_summary_enhanced.csv format.

    Args:
        results_dir: Directory containing AutoInterp outputs.
        output_name: Filename for the enhanced CSV summary (defaults to nemotron_finance_results_summary_enhanced.csv).
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_path}")
        return
    
    csv_data = []
    
    # Check explanations directory
    explanations_dir = results_path / "explanations"
    detection_scores_dir = results_path / "scores" / "detection"
    fuzz_scores_dir = results_path / "scores" / "fuzz"
    
    if not explanations_dir.exists():
        print(f"❌ Explanations directory not found: {explanations_dir}")
        return
    
    print(f"✅ Found explanations directory: {explanations_dir}")
    if detection_scores_dir.exists():
        print(f"✅ Found detection scorer results")
    if fuzz_scores_dir.exists():
        print(f"✅ Found fuzz scorer results")
    
    # Process each explanation file
    for explanation_file in explanations_dir.glob("*.txt"):
        # Extract layer and feature number from filename
        # Format: backbone.layers.28_latent27716.txt
        filename = explanation_file.stem
        
        if "latent" not in filename:
            continue
        
        # Parse backbone.layers.28_latent27716 format
        parts = filename.split("_")
        if len(parts) < 2:
            continue
        
        layer_part = parts[0]  # backbone.layers.28
        feature_part = parts[1]  # latent27716
        
        # Extract layer number from backbone.layers.28
        if "backbone.layers." in layer_part:
            layer_num = layer_part.split(".")[-1]
        elif "layers." in layer_part:
            layer_num = layer_part.split(".")[1]
        else:
            print(f"⚠️  Skipping {filename}: unexpected layer format")
            continue
        
        # Extract feature number from latent27716
        feature_num = feature_part.replace("latent", "")
        
        # Read explanation label
        try:
            with open(explanation_file, 'r') as f:
                label = f.read().strip().strip('"')
        except Exception as e:
            print(f"⚠️  Error reading explanation {explanation_file}: {e}")
            label = "Error reading explanation"
        
        # Get F1 scores from detection and fuzz scorers
        detection_f1 = None
        if detection_scores_dir.exists():
            detection_file = detection_scores_dir / explanation_file.name
            if detection_file.exists():
                detection_f1 = extract_metrics_from_file(detection_file, 'detection')
        
        fuzz_f1 = None
        if fuzz_scores_dir.exists():
            fuzz_file = fuzz_scores_dir / explanation_file.name
            if fuzz_file.exists():
                fuzz_f1 = extract_metrics_from_file(fuzz_file, 'fuzz')
        
        # Calculate overall f1_score (average of detection and fuzz, or just detection if fuzz not available)
        if detection_f1 is not None and fuzz_f1 is not None:
            f1_score = round((detection_f1 + fuzz_f1) / 2, 4)
        elif detection_f1 is not None:
            f1_score = detection_f1
        elif fuzz_f1 is not None:
            f1_score = fuzz_f1
        else:
            f1_score = None
        
        # Check thresholds
        thresholds = check_thresholds(f1_score, detection_f1, fuzz_f1)
        
        # Try to load top sentences if available
        top_positive_sentence = None
        top_negative_sentence = None
        all_top_positive_sentences = None
        analysis_dir = results_path / "Analysis" / "feature_analysis"
        sentences_json = analysis_dir / f"TOP_SENTENCES_{feature_num}.json"
        if sentences_json.exists():
            try:
                with open(sentences_json, 'r') as f:
                    sentences_data = json.load(f)
                    if sentences_data.get('positive_sentences'):
                        # Get top positive sentence (clean text, truncated)
                        top_pos = sentences_data['positive_sentences'][0]
                        top_positive_sentence = top_pos.get('text_clean', '')[:100]
                        # Get all top positive sentences/phrases separated by |
                        all_top_sentences = [
                            sent.get('text_clean', sent.get('text', ''))
                            for sent in sentences_data['positive_sentences']
                        ]
                        all_top_positive_sentences = '|'.join(all_top_sentences)
                    if sentences_data.get('negative_sentences'):
                        # Get top negative sentence (clean text, truncated)
                        top_neg = sentences_data['negative_sentences'][0]
                        top_negative_sentence = top_neg.get('text_clean', '')[:100]
            except Exception as e:
                # Log error but continue processing
                print(f"⚠️  Warning: Could not load sentences for feature {feature_num}: {e}")
        
        csv_data.append({
            'layer': layer_num,
            'feature': feature_num,
            'label': label,
            'f1_score': f1_score,
            'detection_f1': detection_f1,
            'fuzz_f1': fuzz_f1,
            'f1_good': thresholds['f1_good'],
            'detection_f1_good': thresholds['detection_f1_good'],
            'fuzz_good': thresholds['fuzz_good'],
            'top_positive_sentence': top_positive_sentence or '',
            'top_negative_sentence': top_negative_sentence or '',
            'all_top_positive_sentences': all_top_positive_sentences or '',
        })
    
    # Sort by feature number (as integer)
    csv_data.sort(key=lambda x: int(x['feature']))
    
    # Write CSV file
    if csv_data:
        csv_file = results_path / output_name
        fieldnames = [
            'layer', 'feature', 'label',
            'f1_score', 'detection_f1', 'fuzz_f1',
            'f1_good', 'detection_f1_good', 'fuzz_good',
            'top_positive_sentence', 'top_negative_sentence',
            'all_top_positive_sentences'
        ]
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\n✅ Enhanced CSV saved to: {csv_file}")
        print(f"📊 Generated {len(csv_data)} feature results")
        
        # Print summary with thresholds
        print("\n📈 Summary of metrics with thresholds:")
        print("   Thresholds: f1_score >= 0.70, detection_f1 >= 0.70, fuzz_f1 <= 0.30")
        print("\n   First 10 features:")
        for row in csv_data[:10]:
            f1 = row.get('f1_score', 'N/A')
            det_f1 = row.get('detection_f1', 'N/A')
            fuzz = row.get('fuzz_f1', 'N/A')
            f1_good = "✅" if row.get('f1_good') else "❌"
            det_good = "✅" if row.get('detection_f1_good') else "❌"
            fuzz_good = "✅" if row.get('fuzz_good') else "❌"
            label_short = row['label'][:50] + "..." if len(row['label']) > 50 else row['label']
            print(f"  Feature {row['feature']}: {label_short}")
            print(f"    f1={f1} {f1_good}, det_f1={det_f1} {det_good}, fuzz={fuzz} {fuzz_good}")
        
        # Count good features
        f1_good_count = sum(1 for row in csv_data if row.get('f1_good'))
        det_good_count = sum(1 for row in csv_data if row.get('detection_f1_good'))
        fuzz_good_count = sum(1 for row in csv_data if row.get('fuzz_good'))
        print(f"\n📊 Summary counts:")
        print(f"   Features with f1_score >= 0.70: {f1_good_count}/{len(csv_data)}")
        print(f"   Features with detection_f1 >= 0.70: {det_good_count}/{len(csv_data)}")
        print(f"   Features with fuzz_f1 <= 0.30: {fuzz_good_count}/{len(csv_data)}")
    else:
        print("❌ No results found to generate CSV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate enhanced CSV summary for AutoInterp results"
    )
    parser.add_argument(
        "results_dir",
        help="Path to the AutoInterp results directory (must contain explanations/ and scores/)",
    )
    parser.add_argument(
        "--output_name",
        default="nemotron_finance_results_summary_enhanced.csv",
        help="Filename for the enhanced CSV summary (default: nemotron_finance_results_summary_enhanced.csv)",
    )

    args = parser.parse_args()
    generate_enhanced_csv(args.results_dir, args.output_name)

