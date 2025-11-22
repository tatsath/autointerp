#!/usr/bin/env python3
"""
Generate comprehensive results table with quality assessments
"""

import json
import os
from pathlib import Path

def load_latest_results():
    """Load the most recent comprehensive analysis results"""
    results_dir = Path("results")
    if not results_dir.exists():
        return None
    
    # Find the most recent comprehensive analysis directory
    comprehensive_dirs = [d for d in results_dir.iterdir() if d.name.startswith("comprehensive_analysis_")]
    if not comprehensive_dirs:
        return None
    
    latest_dir = max(comprehensive_dirs, key=lambda x: x.name)
    results_file = latest_dir / "comprehensive_results.json"
    
    if not results_file.exists():
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)

def assess_feature_quality(f1, clusters, polysemanticity, specialization):
    """Assess feature quality based on multiple metrics"""
    
    # Quality thresholds
    F1_EXCELLENT = 0.9
    F1_GOOD = 0.7
    F1_FAIR = 0.6
    
    CLUSTERS_EXCELLENT = 2
    CLUSTERS_GOOD = 3
    CLUSTERS_FAIR = 4
    
    POLY_EXCELLENT = 0.3
    POLY_GOOD = 0.5
    POLY_FAIR = 0.7
    
    SPEC_EXCELLENT = 0.5
    SPEC_GOOD = 0.3
    SPEC_FAIR = 0.2
    
    # Assess each metric
    f1_quality = "Excellent" if f1 >= F1_EXCELLENT else "Good" if f1 >= F1_GOOD else "Fair" if f1 >= F1_FAIR else "Poor"
    cluster_quality = "Excellent" if clusters <= CLUSTERS_EXCELLENT else "Good" if clusters <= CLUSTERS_GOOD else "Fair" if clusters <= CLUSTERS_FAIR else "Poor"
    poly_quality = "Excellent" if polysemanticity <= POLY_EXCELLENT else "Good" if polysemanticity <= POLY_GOOD else "Fair" if polysemanticity <= POLY_FAIR else "Poor"
    spec_quality = "Excellent" if specialization >= SPEC_EXCELLENT else "Good" if specialization >= SPEC_GOOD else "Fair" if specialization >= SPEC_FAIR else "Poor"
    
    # Overall quality assessment
    excellent_count = sum([f1_quality == "Excellent", cluster_quality == "Excellent", poly_quality == "Excellent", spec_quality == "Excellent"])
    good_count = sum([f1_quality == "Good", cluster_quality == "Good", poly_quality == "Good", spec_quality == "Good"])
    fair_count = sum([f1_quality == "Fair", cluster_quality == "Fair", poly_quality == "Fair", spec_quality == "Fair"])
    poor_count = sum([f1_quality == "Poor", cluster_quality == "Poor", poly_quality == "Poor", spec_quality == "Poor"])
    
    if excellent_count >= 3:
        overall_quality = "Excellent"
    elif excellent_count >= 2 or good_count >= 3:
        overall_quality = "Good"
    elif good_count >= 2 or fair_count >= 3:
        overall_quality = "Fair"
    else:
        overall_quality = "Poor"
    
    return {
        'f1_quality': f1_quality,
        'cluster_quality': cluster_quality,
        'poly_quality': poly_quality,
        'spec_quality': spec_quality,
        'overall_quality': overall_quality
    }

def generate_results_table():
    """Generate comprehensive results table"""
    results = load_latest_results()
    if not results:
        print("❌ No comprehensive analysis results found!")
        print("Run the analysis first with: python run_analysis.py --comprehensive")
        return
    
    print("=" * 120)
    print("🎯 COMPREHENSIVE FEATURE ANALYSIS RESULTS WITH QUALITY ASSESSMENTS")
    print("=" * 120)
    print()
    
    # Table header
    print(f"{'Feature':<8} {'Label':<45} {'F1':<6} {'F1_Q':<8} {'Clus':<6} {'Clus_Q':<8} {'Poly':<6} {'Poly_Q':<8} {'Spec':<6} {'Spec_Q':<8} {'Overall':<8}")
    print("-" * 120)
    
    # Process each feature
    for feature in results:
        feature_id = feature['feature_id']
        label = feature['label']
        f1 = feature['f1']
        clusters = feature['n_clusters']
        polysemanticity = feature['polysemanticity']
        specialization = feature['specialization_score']
        
        # Assess quality
        quality = assess_feature_quality(f1, clusters, polysemanticity, specialization)
        
        # Format values
        f1_str = f"{f1:.3f}"
        clusters_str = f"{clusters}"
        poly_str = f"{polysemanticity:.3f}"
        spec_str = f"{specialization:.3f}"
        
        # Truncate label if too long
        label_short = label[:42] + "..." if len(label) > 45 else label
        
        print(f"{feature_id:<8} {label_short:<45} {f1_str:<6} {quality['f1_quality']:<8} {clusters_str:<6} {quality['cluster_quality']:<8} {poly_str:<6} {quality['poly_quality']:<8} {spec_str:<6} {quality['spec_quality']:<8} {quality['overall_quality']:<8}")
    
    print("-" * 120)
    print()
    
    # Quality summary
    print("📊 QUALITY ASSESSMENT SUMMARY")
    print("=" * 50)
    
    # Count features by overall quality
    quality_counts = {}
    for feature in results:
        quality = assess_feature_quality(
            feature['f1'], 
            feature['n_clusters'], 
            feature['polysemanticity'], 
            feature['specialization_score']
        )
        overall = quality['overall_quality']
        quality_counts[overall] = quality_counts.get(overall, 0) + 1
    
    for quality, count in sorted(quality_counts.items()):
        print(f"{quality}: {count} features")
    
    print()
    
    # Threshold explanations
    print("🎯 QUALITY THRESHOLDS")
    print("=" * 30)
    print("F1 Score: Excellent ≥0.9, Good ≥0.7, Fair ≥0.6, Poor <0.6")
    print("Clusters: Excellent ≤2, Good ≤3, Fair ≤4, Poor >4")
    print("Polysemanticity: Excellent ≤0.3, Good ≤0.5, Fair ≤0.7, Poor >0.7")
    print("Specialization: Excellent ≥0.5, Good ≥0.3, Fair ≥0.2, Poor <0.2")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("=" * 20)
    excellent_features = [f for f in results if assess_feature_quality(f['f1'], f['n_clusters'], f['polysemanticity'], f['specialization_score'])['overall_quality'] == 'Excellent']
    good_features = [f for f in results if assess_feature_quality(f['f1'], f['n_clusters'], f['polysemanticity'], f['specialization_score'])['overall_quality'] == 'Good']
    
    if excellent_features:
        print(f"✅ Use these {len(excellent_features)} excellent features for production:")
        for f in excellent_features:
            print(f"   - Feature {f['feature_id']}: {f['label']}")
    
    if good_features:
        print(f"✅ Consider these {len(good_features)} good features:")
        for f in good_features:
            print(f"   - Feature {f['feature_id']}: {f['label']}")
    
    print()
    print("=" * 120)

if __name__ == "__main__":
    generate_results_table()
