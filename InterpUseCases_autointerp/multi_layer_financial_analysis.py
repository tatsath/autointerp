#!/usr/bin/env python3
"""
Multi-Layer Financial Feature Analysis - Complete System
Analyzes top 30 features across different layers of Llama model SAE and identifies financial features
"""

import argparse
import subprocess
import sys
import json
import pandas as pd
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns

class MultiLayerFinancialAnalyzer:
    def __init__(self, base_model_name, sae_model_path, output_dir="multi_layer_results"):
        """
        Initialize the multi-layer analyzer
        
        Args:
            base_model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            sae_model_path: Path to the SAE model directory
            output_dir: Directory to save results
        """
        self.base_model_name = base_model_name
        self.sae_model_path = sae_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Extract model name for run naming
        self.model_name = base_model_name.split('/')[-1].replace('-', '_')
        self.sae_name = Path(sae_model_path).name
        
        # Default layers for Llama-2-7B
        self.default_layers = [4, 10, 16, 22, 28]
    
    def run_feature_analysis_for_layer(self, layer_idx, top_n=30):
        """Run feature analysis for a specific layer"""
        print(f"\n{'='*80}")
        print(f"ANALYZING LAYER {layer_idx} - TOP {top_n} FEATURES")
        print(f"{'='*80}")
        
        # Create layer-specific output directory
        layer_output_dir = self.output_dir / f"layer_{layer_idx}"
        layer_output_dir.mkdir(exist_ok=True)
        
        # Run feature analysis
        cmd = [
            sys.executable, "generic_feature_analysis.py",
            "--base_model", self.base_model_name,
            "--sae_model", self.sae_model_path,
            "--top_n", str(top_n),
            "--layer_idx", str(layer_idx),
            "--output_dir", str(layer_output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Layer {layer_idx} analysis completed successfully!")
            return True, layer_output_dir
        else:
            print(f"❌ Layer {layer_idx} analysis failed!")
            print("STDERR:", result.stderr)
            return False, None
    
    def run_feature_labeling_for_layer(self, layer_output_dir, top_n=30):
        """Run feature labeling for a specific layer"""
        print(f"\n{'='*60}")
        print(f"LABELING FEATURES FOR LAYER")
        print(f"{'='*60}")
        
        # Find the analysis file
        analysis_files = list(layer_output_dir.glob("top_*_features_analysis.json"))
        if not analysis_files:
            print("❌ No analysis file found!")
            return False
        
        analysis_file = analysis_files[0]  # Take the first one
        
        # Run feature labeling
        cmd = [
            sys.executable, "generic_feature_labeling.py",
            "--analysis_file", str(analysis_file),
            "--labeling_model", "meta-llama/Llama-2-7b-chat-hf",
            "--domain", "financial",
            "--top_n", str(top_n),
            "--output_dir", str(layer_output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Layer labeling completed successfully!")
            return True
        else:
            print(f"❌ Layer labeling failed!")
            print("STDERR:", result.stderr)
            return False
    
    def consolidate_results(self, layers):
        """Consolidate results from all layers"""
        print(f"\n{'='*80}")
        print("CONSOLIDATING RESULTS FROM ALL LAYERS")
        print(f"{'='*80}")
        
        all_results = []
        
        for layer_idx in layers:
            layer_output_dir = self.output_dir / f"layer_{layer_idx}"
            
            # Load feature analysis results
            analysis_files = list(layer_output_dir.glob("top_*_features_analysis.csv"))
            if analysis_files:
                df = pd.read_csv(analysis_files[0])
                df['layer'] = layer_idx
                all_results.append(df)
                print(f"✅ Loaded {len(df)} features from layer {layer_idx}")
            else:
                print(f"❌ No analysis results found for layer {layer_idx}")
        
        if all_results:
            # Combine all results
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # Sort by specialization (descending)
            combined_df = combined_df.sort_values('specialization', ascending=False)
            
            # Save consolidated results
            consolidated_file = self.output_dir / "consolidated_top_features_all_layers.csv"
            combined_df.to_csv(consolidated_file, index=False)
            
            print(f"\n✅ Consolidated results saved to: {consolidated_file}")
            print(f"Total features analyzed: {len(combined_df)}")
            print(f"Layers analyzed: {sorted(combined_df['layer'].unique())}")
            
            # Show top 10 features across all layers
            print(f"\nTop 10 Financial Features Across All Layers:")
            top_10 = combined_df.head(10)
            for _, row in top_10.iterrows():
                print(f"  Feature {row['feature_number']} (Layer {row['layer']}): Specialization = {row['specialization']:.3f}")
            
            return combined_df
        else:
            print("❌ No results to consolidate!")
            return None
    
    def create_layer_comparison_report(self, layers):
        """Create a comparison report across layers"""
        print(f"\n{'='*80}")
        print("CREATING LAYER COMPARISON REPORT")
        print(f"{'='*80}")
        
        layer_stats = []
        
        for layer_idx in layers:
            layer_output_dir = self.output_dir / f"layer_{layer_idx}"
            
            # Load results
            analysis_files = list(layer_output_dir.glob("top_*_features_analysis.csv"))
            if analysis_files:
                df = pd.read_csv(analysis_files[0])
                
                stats = {
                    'layer': layer_idx,
                    'total_features': len(df),
                    'avg_specialization': df['specialization'].mean(),
                    'max_specialization': df['specialization'].max(),
                    'min_specialization': df['specialization'].min(),
                    'std_specialization': df['specialization'].std(),
                    'top_feature': df.loc[df['specialization'].idxmax(), 'feature_number'],
                    'top_specialization': df['specialization'].max()
                }
                layer_stats.append(stats)
                print(f"✅ Layer {layer_idx}: {len(df)} features, avg specialization = {stats['avg_specialization']:.3f}")
        
        if layer_stats:
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(layer_stats)
            comparison_df = comparison_df.sort_values('avg_specialization', ascending=False)
            
            # Save comparison report
            comparison_file = self.output_dir / "layer_comparison_report.csv"
            comparison_df.to_csv(comparison_file, index=False)
            
            print(f"\n✅ Layer comparison report saved to: {comparison_file}")
            
            # Show layer ranking
            print(f"\nLayer Ranking by Average Specialization:")
            for _, row in comparison_df.iterrows():
                print(f"  Layer {row['layer']}: {row['avg_specialization']:.3f} (Top feature: {row['top_feature']})")
            
            return comparison_df
        else:
            print("❌ No layer statistics to compare!")
            return None
    
    def identify_top_financial_features(self, df, top_n=50, min_specialization=0.1):
        """Identify top financial features based on specialization"""
        print(f"\n{'='*80}")
        print("IDENTIFYING TOP FINANCIAL FEATURES")
        print(f"{'='*80}")
        
        # Filter by minimum specialization
        filtered_df = df[df['specialization'] >= min_specialization].copy()
        print(f"Features with specialization >= {min_specialization}: {len(filtered_df)}")
        
        # Sort by specialization (descending)
        filtered_df = filtered_df.sort_values('specialization', ascending=False)
        
        # Get top N features
        top_features = filtered_df.head(top_n)
        
        print(f"\nTop {len(top_features)} Financial Features:")
        print("-" * 80)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. Feature {int(row['feature_number']):3d} (Layer {int(row['layer']):2d}): "
                  f"Specialization = {row['specialization']:.3f}, "
                  f"Financial Act = {row['financial_activation']:.3f}, "
                  f"General Act = {row['general_activation']:.3f}")
        
        return top_features
    
    def create_visualizations(self, consolidated_df, layer_df, top_features):
        """Create visualizations for the analysis"""
        print(f"\n{'='*80}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*80}")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Layer Financial Feature Analysis', fontsize=16)
        
        # 1. Specialization distribution by layer
        layer_specialization = consolidated_df.groupby('layer')['specialization'].mean().sort_values(ascending=False)
        axes[0, 0].bar(layer_specialization.index, layer_specialization.values)
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Average Specialization')
        axes[0, 0].set_title('Average Specialization by Layer')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top features by layer
        top_features_by_layer = top_features.groupby('layer').size()
        axes[0, 1].bar(top_features_by_layer.index, top_features_by_layer.values)
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Number of Top Features')
        axes[0, 1].set_title('Top Financial Features by Layer')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Specialization vs Financial Activation
        axes[1, 0].scatter(consolidated_df['financial_activation'], consolidated_df['specialization'], 
                          alpha=0.6, c=consolidated_df['layer'], cmap='viridis')
        axes[1, 0].set_xlabel('Financial Activation')
        axes[1, 0].set_ylabel('Specialization')
        axes[1, 0].set_title('Specialization vs Financial Activation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Feature count by layer
        feature_counts = consolidated_df['layer'].value_counts().sort_index()
        axes[1, 1].bar(feature_counts.index, feature_counts.values)
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Number of Features')
        axes[1, 1].set_title('Total Features Analyzed by Layer')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'financial_feature_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {plot_file}")
    
    def generate_summary_report(self, consolidated_df, layer_df, top_features):
        """Generate a comprehensive summary report"""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Calculate statistics
        total_features = len(consolidated_df)
        total_layers = len(layer_df)
        avg_specialization = consolidated_df['specialization'].mean()
        max_specialization = consolidated_df['specialization'].max()
        
        # Find best layer
        best_layer = layer_df.loc[layer_df['avg_specialization'].idxmax()]
        
        # Find best feature
        best_feature = consolidated_df.loc[consolidated_df['specialization'].idxmax()]
        
        # Create summary
        summary = {
            'analysis_summary': {
                'total_features_analyzed': total_features,
                'total_layers_analyzed': total_layers,
                'layers_analyzed': sorted(consolidated_df['layer'].unique().tolist()),
                'average_specialization': avg_specialization,
                'max_specialization': max_specialization,
                'best_layer': {
                    'layer_number': int(best_layer['layer']),
                    'avg_specialization': float(best_layer['avg_specialization']),
                    'max_specialization': float(best_layer['max_specialization']),
                    'top_feature': int(best_layer['top_feature'])
                },
                'best_feature': {
                    'feature_number': int(best_feature['feature_number']),
                    'layer': int(best_feature['layer']),
                    'specialization': float(best_feature['specialization']),
                    'financial_activation': float(best_feature['financial_activation']),
                    'general_activation': float(best_feature['general_activation'])
                }
            },
            'top_financial_features': top_features.to_dict('records'),
            'layer_performance': layer_df.to_dict('records')
        }
        
        # Save summary report
        summary_file = self.output_dir / 'financial_feature_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Summary report saved to: {summary_file}")
        
        # Print summary
        print(f"\n📊 FINANCIAL FEATURE ANALYSIS SUMMARY:")
        print(f"  Total Features Analyzed: {total_features}")
        print(f"  Total Layers Analyzed: {total_layers}")
        print(f"  Layers: {sorted(consolidated_df['layer'].unique().tolist())}")
        print(f"  Average Specialization: {avg_specialization:.3f}")
        print(f"  Maximum Specialization: {max_specialization:.3f}")
        print(f"  Best Layer: {int(best_layer['layer'])} (avg specialization: {best_layer['avg_specialization']:.3f})")
        print(f"  Best Feature: {int(best_feature['feature_number'])} in layer {int(best_feature['layer'])} (specialization: {best_feature['specialization']:.3f})")
        
        return summary
    
    def run_complete_analysis(self, layers=None, top_n=30, min_specialization=0.1, top_features_n=50):
        """Run complete analysis for multiple layers"""
        if layers is None:
            layers = self.default_layers
        
        print("="*80)
        print("MULTI-LAYER FINANCIAL FEATURE ANALYSIS")
        print("="*80)
        print(f"Base Model: {self.base_model_name}")
        print(f"SAE Model: {self.sae_model_path}")
        print(f"Layers: {layers}")
        print(f"Top Features per Layer: {top_n}")
        print(f"Output Directory: {self.output_dir}")
        print("="*80)
        
        successful_layers = []
        
        # Process each layer
        for layer_idx in layers:
            print(f"\n🔄 Processing Layer {layer_idx}...")
            
            # Step 1: Feature Analysis
            success, layer_output_dir = self.run_feature_analysis_for_layer(layer_idx, top_n)
            if not success:
                print(f"❌ Skipping layer {layer_idx} due to analysis failure")
                continue
            
            # Step 2: Feature Labeling
            success = self.run_feature_labeling_for_layer(layer_output_dir, top_n)
            if not success:
                print(f"❌ Skipping layer {layer_idx} due to labeling failure")
                continue
            
            successful_layers.append(layer_idx)
            print(f"✅ Layer {layer_idx} completed successfully!")
        
        # Consolidate results
        if successful_layers:
            print(f"\n🎉 Analysis completed for layers: {successful_layers}")
            
            # Consolidate all results
            consolidated_df = self.consolidate_results(successful_layers)
            
            # Create layer comparison report
            comparison_df = self.create_layer_comparison_report(successful_layers)
            
            # Identify top financial features
            top_features = self.identify_top_financial_features(consolidated_df, top_features_n, min_specialization)
            
            # Create visualizations
            self.create_visualizations(consolidated_df, comparison_df, top_features)
            
            # Generate summary report
            summary = self.generate_summary_report(consolidated_df, comparison_df, top_features)
            
            # Save top features to CSV
            top_features_file = self.output_dir / f'top_{len(top_features)}_financial_features.csv'
            top_features.to_csv(top_features_file, index=False)
            print(f"✅ Top financial features saved to: {top_features_file}")
            
            print(f"\n📊 FINAL SUMMARY:")
            print(f"  Successful layers: {successful_layers}")
            print(f"  Total features analyzed: {len(consolidated_df) if consolidated_df is not None else 0}")
            print(f"  Results saved in: {self.output_dir}")
            
            return consolidated_df, comparison_df, top_features, summary
        else:
            print("❌ No layers completed successfully!")
            return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Multi-Layer Financial Feature Analysis")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--sae_model", required=True, help="Path to SAE model directory")
    parser.add_argument("--layers", nargs='+', type=int, help="Layer indices to analyze (e.g., 4 10 16 22 28)")
    parser.add_argument("--top_n", type=int, default=30, help="Number of top features per layer")
    parser.add_argument("--output_dir", default="multi_layer_results", help="Output directory")
    parser.add_argument("--min_specialization", type=float, default=0.1, help="Minimum specialization threshold")
    parser.add_argument("--top_features_n", type=int, default=50, help="Number of top features to extract")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = MultiLayerFinancialAnalyzer(
        base_model_name=args.base_model,
        sae_model_path=args.sae_model,
        output_dir=args.output_dir
    )
    
    # Run complete analysis
    consolidated_df, comparison_df, top_features, summary = analyzer.run_complete_analysis(
        layers=args.layers,
        top_n=args.top_n,
        min_specialization=args.min_specialization,
        top_features_n=args.top_features_n
    )

if __name__ == "__main__":
    main()
