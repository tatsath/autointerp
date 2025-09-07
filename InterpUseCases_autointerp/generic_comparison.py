#!/usr/bin/env python3
"""
Generic Comparison System
Compares labels from feature labeling vs Delphi explanations
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

class GenericComparison:
    def __init__(self, output_dir="results"):
        """
        Initialize the generic comparison system
        
        Args:
            output_dir: Directory containing the results
        """
        self.output_dir = Path(output_dir)
        
        # Load sentence transformer for semantic similarity
        print("Loading sentence transformer for semantic similarity...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_results(self, labeling_file, delphi_file):
        """Load results from both methods"""
        print("Loading results...")
        
        # Load labeling results
        labeling_df = pd.read_csv(self.output_dir / labeling_file)
        print(f"Loaded {len(labeling_df)} features from labeling: {labeling_file}")
        
        # Load Delphi results
        delphi_df = pd.read_csv(self.output_dir / delphi_file)
        print(f"Loaded {len(delphi_df)} features from Delphi: {delphi_file}")
        
        return labeling_df, delphi_df
    
    def merge_results(self, labeling_df, delphi_df):
        """Merge results from both methods"""
        print("Merging results...")
        
        # Merge on feature_number
        merged_df = pd.merge(
            labeling_df, 
            delphi_df, 
            on='feature_number', 
            how='inner',
            suffixes=('_labeling', '_delphi')
        )
        
        print(f"Merged {len(merged_df)} features")
        return merged_df
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def analyze_comparison(self, merged_df):
        """Analyze the comparison between methods"""
        print("Analyzing comparison...")
        
        results = []
        
        for _, row in merged_df.iterrows():
            feature_num = row['feature_number']
            labeling_label = str(row['label_labeling'])
            delphi_explanation = str(row['delphi_explanation_delphi'])
            
            # Calculate semantic similarity
            similarity = self.calculate_semantic_similarity(labeling_label, delphi_explanation)
            
            # Get other metrics
            f1_score = row.get('f1_score_delphi', 0.0)
            specialization = row.get('specialization_labeling', 0.0)
            
            results.append({
                'feature_number': feature_num,
                'labeling_label': labeling_label,
                'delphi_explanation': delphi_explanation,
                'semantic_similarity': similarity,
                'f1_score': f1_score,
                'specialization': specialization
            })
        
        return pd.DataFrame(results)
    
    def generate_comparison_report(self, comparison_df):
        """Generate a comprehensive comparison report"""
        print("Generating comparison report...")
        
        # Calculate statistics
        avg_similarity = comparison_df['semantic_similarity'].mean()
        median_similarity = comparison_df['semantic_similarity'].median()
        high_similarity_count = len(comparison_df[comparison_df['semantic_similarity'] > 0.7])
        low_similarity_count = len(comparison_df[comparison_df['semantic_similarity'] < 0.3])
        
        # Correlation analysis
        correlation_sim_f1 = comparison_df['semantic_similarity'].corr(comparison_df['f1_score'])
        correlation_sim_spec = comparison_df['semantic_similarity'].corr(comparison_df['specialization'])
        correlation_f1_spec = comparison_df['f1_score'].corr(comparison_df['specialization'])
        
        report = {
            'summary_statistics': {
                'total_features': len(comparison_df),
                'average_semantic_similarity': avg_similarity,
                'median_semantic_similarity': median_similarity,
                'high_similarity_features': high_similarity_count,
                'low_similarity_features': low_similarity_count,
                'high_similarity_percentage': (high_similarity_count / len(comparison_df)) * 100,
                'low_similarity_percentage': (low_similarity_count / len(comparison_df)) * 100
            },
            'correlations': {
                'similarity_vs_f1': correlation_sim_f1,
                'similarity_vs_specialization': correlation_sim_spec,
                'f1_vs_specialization': correlation_f1_spec
            },
            'top_features': {
                'highest_similarity': comparison_df.nlargest(3, 'semantic_similarity')[['feature_number', 'semantic_similarity', 'labeling_label', 'delphi_explanation']].to_dict('records'),
                'lowest_similarity': comparison_df.nsmallest(3, 'semantic_similarity')[['feature_number', 'semantic_similarity', 'labeling_label', 'delphi_explanation']].to_dict('records'),
                'highest_f1': comparison_df.nlargest(3, 'f1_score')[['feature_number', 'f1_score', 'labeling_label', 'delphi_explanation']].to_dict('records'),
                'highest_specialization': comparison_df.nlargest(3, 'specialization')[['feature_number', 'specialization', 'labeling_label', 'delphi_explanation']].to_dict('records')
            }
        }
        
        return report
    
    def create_visualizations(self, comparison_df, report):
        """Create visualization plots"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Labeling vs Delphi Comparison Analysis', fontsize=16)
        
        # 1. Semantic Similarity Distribution
        axes[0, 0].hist(comparison_df['semantic_similarity'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(report['summary_statistics']['average_semantic_similarity'], 
                          color='red', linestyle='--', label=f'Mean: {report["summary_statistics"]["average_semantic_similarity"]:.3f}')
        axes[0, 0].set_xlabel('Semantic Similarity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Semantic Similarity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Similarity vs F1 Score
        axes[0, 1].scatter(comparison_df['semantic_similarity'], comparison_df['f1_score'], alpha=0.6)
        axes[0, 1].set_xlabel('Semantic Similarity')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title(f'Similarity vs F1 Score (r={report["correlations"]["similarity_vs_f1"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Similarity vs Specialization
        axes[1, 0].scatter(comparison_df['semantic_similarity'], comparison_df['specialization'], alpha=0.6)
        axes[1, 0].set_xlabel('Semantic Similarity')
        axes[1, 0].set_ylabel('Specialization')
        axes[1, 0].set_title(f'Similarity vs Specialization (r={report["correlations"]["similarity_vs_specialization"]:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. F1 Score vs Specialization
        axes[1, 1].scatter(comparison_df['f1_score'], comparison_df['specialization'], alpha=0.6)
        axes[1, 1].set_xlabel('F1 Score')
        axes[1, 1].set_ylabel('Specialization')
        axes[1, 1].set_title(f'F1 Score vs Specialization (r={report["correlations"]["f1_vs_specialization"]:.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'comparison_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {plot_file}")
    
    def run_comparison(self, labeling_file, delphi_file):
        """Run the complete comparison analysis"""
        print("="*80)
        print("GENERIC COMPARISON ANALYSIS")
        print("="*80)
        print(f"Labeling File: {labeling_file}")
        print(f"Delphi File: {delphi_file}")
        print("="*80)
        
        # Load results
        labeling_df, delphi_df = self.load_results(labeling_file, delphi_file)
        
        # Merge results
        merged_df = self.merge_results(labeling_df, delphi_df)
        
        # Analyze comparison
        comparison_df = self.analyze_comparison(merged_df)
        
        # Generate report
        report = self.generate_comparison_report(comparison_df)
        
        # Create visualizations
        self.create_visualizations(comparison_df, report)
        
        # Save results
        comparison_file = self.output_dir / 'comparison_results.csv'
        comparison_df.to_csv(comparison_file, index=False)
        
        report_file = self.output_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Comparison complete!")
        print(f"Results saved to: {comparison_file}")
        print(f"Report saved to: {report_file}")
        
        # Print summary
        print(f"\n📊 SUMMARY:")
        print(f"  Total Features: {report['summary_statistics']['total_features']}")
        print(f"  Average Semantic Similarity: {report['summary_statistics']['average_semantic_similarity']:.3f}")
        print(f"  High Similarity Features (>0.7): {report['summary_statistics']['high_similarity_features']} ({report['summary_statistics']['high_similarity_percentage']:.1f}%)")
        print(f"  Low Similarity Features (<0.3): {report['summary_statistics']['low_similarity_features']} ({report['summary_statistics']['low_similarity_percentage']:.1f}%)")
        print(f"  Correlation (Similarity vs F1): {report['correlations']['similarity_vs_f1']:.3f}")
        print(f"  Correlation (Similarity vs Specialization): {report['correlations']['similarity_vs_specialization']:.3f}")
        
        return comparison_df, report

def main():
    parser = argparse.ArgumentParser(description="Generic Comparison Analysis")
    parser.add_argument("--labeling_file", required=True, help="Labeling results CSV file")
    parser.add_argument("--delphi_file", required=True, help="Delphi results CSV file")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create comparison system
    comparison = GenericComparison(output_dir=args.output_dir)
    
    # Run comparison
    comparison_df, report = comparison.run_comparison(
        labeling_file=args.labeling_file,
        delphi_file=args.delphi_file
    )

if __name__ == "__main__":
    main()
