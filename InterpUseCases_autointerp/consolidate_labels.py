#!/usr/bin/env python3
"""
Consolidate feature labels from all layers into a single CSV
"""

import pandas as pd
import os
from pathlib import Path

def consolidate_all_labels(results_dir="complete_financial_analysis"):
    """Consolidate labels from all layers into a single CSV"""
    
    results_path = Path(results_dir)
    all_labels = []
    
    # Get all layer directories
    layer_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith('layer_')]
    layer_dirs.sort(key=lambda x: int(x.name.split('_')[1]))
    
    print(f"Found {len(layer_dirs)} layer directories: {[d.name for d in layer_dirs]}")
    
    for layer_dir in layer_dirs:
        layer_num = int(layer_dir.name.split('_')[1])
        label_file = layer_dir / "feature_labels_clean_financial.csv"
        
        if label_file.exists():
            print(f"Processing layer {layer_num}...")
            df = pd.read_csv(label_file)
            df['layer'] = layer_num
            all_labels.append(df)
            print(f"  Loaded {len(df)} features from layer {layer_num}")
        else:
            print(f"  No label file found for layer {layer_num}")
    
    if all_labels:
        # Combine all labels
        consolidated_df = pd.concat(all_labels, ignore_index=True)
        
        # Reorder columns
        consolidated_df = consolidated_df[['layer', 'feature_number', 'label']]
        
        # Sort by layer, then by feature number
        consolidated_df = consolidated_df.sort_values(['layer', 'feature_number'])
        
        # Save consolidated file
        output_file = results_path / "all_layers_features_and_labels.csv"
        consolidated_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Consolidated labels saved to: {output_file}")
        print(f"Total features with labels: {len(consolidated_df)}")
        print(f"Layers included: {sorted(consolidated_df['layer'].unique())}")
        
        # Show sample
        print(f"\nSample of consolidated labels:")
        print(consolidated_df.head(10).to_string(index=False))
        
        return consolidated_df
    else:
        print("❌ No label files found!")
        return None

def create_top_features_with_labels(results_dir="complete_financial_analysis"):
    """Create a CSV with top features and their labels"""
    
    results_path = Path(results_dir)
    
    # Load top features
    top_features_file = results_path / "top_50_financial_features.csv"
    if not top_features_file.exists():
        print(f"❌ Top features file not found: {top_features_file}")
        return None
    
    top_features_df = pd.read_csv(top_features_file)
    print(f"Loaded {len(top_features_df)} top features")
    
    # Load consolidated labels
    labels_file = results_path / "all_layers_features_and_labels.csv"
    if not labels_file.exists():
        print("Creating consolidated labels first...")
        consolidate_all_labels(results_dir)
    
    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} feature labels")
    
    # Merge top features with labels
    merged_df = pd.merge(
        top_features_df, 
        labels_df, 
        on=['layer', 'feature_number'], 
        how='left'
    )
    
    # Reorder columns
    merged_df = merged_df[['rank', 'layer', 'feature_number', 'label', 'specialization', 'financial_activation', 'general_activation']]
    
    # Save merged file
    output_file = results_path / "top_features_with_labels.csv"
    merged_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Top features with labels saved to: {output_file}")
    print(f"Features with labels: {merged_df['label'].notna().sum()}/{len(merged_df)}")
    
    # Show sample
    print(f"\nTop 10 features with labels:")
    sample_df = merged_df.head(10)[['rank', 'layer', 'feature_number', 'label', 'specialization']]
    print(sample_df.to_string(index=False, max_colwidth=50))
    
    return merged_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "complete_financial_analysis"
    
    print("="*80)
    print("CONSOLIDATING FEATURE LABELS")
    print("="*80)
    
    # Create consolidated labels
    consolidated_df = consolidate_all_labels(results_dir)
    
    if consolidated_df is not None:
        print("\n" + "="*80)
        print("CREATING TOP FEATURES WITH LABELS")
        print("="*80)
        
        # Create top features with labels
        top_with_labels_df = create_top_features_with_labels(results_dir)
