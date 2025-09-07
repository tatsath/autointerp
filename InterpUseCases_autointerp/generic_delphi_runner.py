#!/usr/bin/env python3
"""
Generic Delphi Runner
Works with any model, SAE, and any number of features
"""

import subprocess
import os
import json
import pandas as pd
import sys
from pathlib import Path
import argparse

# Add autointerp_full to path
sys.path.append('/home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp/autointerp_full')

from delphi.log.result_analysis import compute_classification_metrics, compute_confusion

class GenericDelphiRunner:
    def __init__(self, base_model_name, sae_model_path, output_dir="results"):
        """
        Initialize the generic Delphi runner
        
        Args:
            base_model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            sae_model_path: Path to the SAE model directory
            output_dir: Directory to save results
        """
        self.base_model_name = base_model_name
        self.sae_model_path = sae_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set environment variables
        self.env = os.environ.copy()
        self.env['PYTHONPATH'] = '/home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp'
        self.env['OPENROUTER_API_KEY'] = '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'
    
    def run_delphi_analysis(self, feature_numbers, run_name="generic_delphi_run", 
                           explainer_model="openai/gpt-3.5-turbo", n_tokens=100000):
        """
        Run Delphi analysis for specific features
        
        Args:
            feature_numbers: List of feature numbers to analyze
            run_name: Name for this run
            explainer_model: Model to use for explanations
            n_tokens: Number of tokens to process
        """
        print("="*80)
        print("GENERIC DELPHI RUNNER")
        print("="*80)
        print(f"Base Model: {self.base_model_name}")
        print(f"SAE Model: {self.sae_model_path}")
        print(f"Features: {feature_numbers}")
        print(f"Explainer Model: {explainer_model}")
        print(f"Run Name: {run_name}")
        print("="*80)
        
        # Change to autointerp directory
        os.chdir('/home/nvidia/Documents/Hariom/saetrain/InterpUseCases/UseCase_FinancialFeatureFinding/autointerp')
        
        # Build command
        cmd = [
            "python", "-m", "autointerp_full.delphi",
            self.base_model_name,
            self.sae_model_path,
            "--n_tokens", str(n_tokens),
            "--feature_num"] + [str(f) for f in feature_numbers] + [
            "--hookpoints", "layers.16",
            "--scorers", "detection",
            "--explainer", "default",
            "--explainer_model", explainer_model,
            "--explainer_provider", "openrouter",
            "--explainer_model_max_len", "512",
            "--dataset_repo", "wikitext",
            "--dataset_name", "wikitext-103-raw-v1",
            "--dataset_split", "train[:1%]",
            "--filter_bos",
            "--num_gpus", "4",
            "--num_examples_per_scorer_prompt", "1",
            "--n_non_activating", "50",
            "--non_activating_source", "FAISS",
            "--faiss_embedding_model", "sentence-transformers/all-MiniLM-L6-v2",
            "--faiss_embedding_cache_dir", ".embedding_cache",
            "--faiss_embedding_cache_enabled",
            "--name", run_name
        ]
        
        print(f"Running command: {' '.join(cmd[:8])}... (truncated)")
        print(f"Features to process: {feature_numbers}")
        
        try:
            result = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=1800)
            print(f"\nDelphi exit code: {result.returncode}")
            
            if result.returncode == 0:
                print("✅ Delphi completed successfully!")
                return True
            else:
                print("❌ Delphi failed!")
                if result.stderr:
                    print("STDERR:", result.stderr[-500:])
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Delphi timed out after 30 minutes")
            return False
        except Exception as e:
            print(f"❌ Error running Delphi: {e}")
            return False
    
    def calculate_f1_scores(self, run_name, feature_numbers):
        """Calculate F1 scores using Delphi's built-in calculation"""
        print(f"\nCalculating F1 scores for run: {run_name}")
        
        results_dir = f"runs/{run_name}"
        explanations_dir = os.path.join(results_dir, "explanations")
        scores_dir = os.path.join(results_dir, "scores", "detection")
        
        if not os.path.exists(explanations_dir):
            print(f"❌ Results directory not found: {explanations_dir}")
            return []
        
        results = []
        
        for feature_num in feature_numbers:
            print(f"  Processing Feature {feature_num}...")
            
            score_file = os.path.join(scores_dir, f"layers.16_latent{feature_num}.txt")
            explanation_file = os.path.join(explanations_dir, f"layers.16_latent{feature_num}.txt")
            
            if not os.path.exists(score_file):
                print(f"    ❌ Score file not found: {score_file}")
                continue
            
            # Read and parse the score data
            with open(score_file, 'r') as f:
                score_data = json.load(f)
            
            # Read Delphi explanation
            delphi_explanation = "No explanation"
            if os.path.exists(explanation_file):
                with open(explanation_file, 'rb') as f:
                    delphi_explanation = f.read().decode('utf-8').strip('"')
            
            # Convert to DataFrame format expected by Delphi
            df_data = []
            for item in score_data:
                df_data.append({
                    'activating': item.get('activating', False),
                    'prediction': item.get('prediction', False),
                    'correct': item.get('correct', False)
                })
            
            df = pd.DataFrame(df_data)
            
            # Use Delphi's built-in F1 calculation
            confusion = compute_confusion(df)
            metrics = compute_classification_metrics(confusion)
            
            f1_score = metrics['f1_score']
            precision = metrics['precision']
            recall = metrics['recall']
            
            print(f"    📊 Total examples: {len(df)}")
            print(f"    📊 Activating examples: {df['activating'].sum()}")
            print(f"    📊 LLM predictions: {df['prediction'].sum()}")
            print(f"    📊 Correct predictions: {df['correct'].sum()}")
            print(f"    📊 Precision: {precision:.3f}")
            print(f"    📊 Recall: {recall:.3f}")
            print(f"    ✅ Delphi F1 Score: {f1_score:.3f}")
            
            results.append({
                'feature_number': feature_num,
                'delphi_explanation': delphi_explanation,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall
            })
        
        return results
    
    def run_complete_analysis(self, feature_numbers, run_name="generic_delphi_run", 
                             explainer_model="openai/gpt-3.5-turbo", n_tokens=100000):
        """Run complete Delphi analysis and return results"""
        
        # Run Delphi
        success = self.run_delphi_analysis(feature_numbers, run_name, explainer_model, n_tokens)
        
        if not success:
            print("❌ Delphi analysis failed!")
            return []
        
        # Calculate F1 scores
        results = self.calculate_f1_scores(run_name, feature_numbers)
        
        # Save results
        if results:
            df = pd.DataFrame(results)
            output_file = self.output_dir / f"{run_name}_delphi_results.csv"
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ Complete analysis finished!")
            print(f"Results saved to: {output_file}")
            
            print(f"\nDelphi Results:")
            for result in results:
                print(f"  Feature {result['feature_number']}: F1={result['f1_score']:.3f}")
                print(f"    Explanation: {result['delphi_explanation']}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Generic Delphi Runner")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--sae_model", required=True, help="Path to SAE model directory")
    parser.add_argument("--features", required=True, help="Comma-separated list of feature numbers (e.g., 163,59,333)")
    parser.add_argument("--run_name", default="generic_delphi_run", help="Name for this run")
    parser.add_argument("--explainer_model", default="openai/gpt-3.5-turbo", help="Explainer model")
    parser.add_argument("--n_tokens", type=int, default=100000, help="Number of tokens to process")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Parse feature numbers
    feature_numbers = [int(f.strip()) for f in args.features.split(',')]
    
    # Create runner
    runner = GenericDelphiRunner(
        base_model_name=args.base_model,
        sae_model_path=args.sae_model,
        output_dir=args.output_dir
    )
    
    # Run complete analysis
    results = runner.run_complete_analysis(
        feature_numbers=feature_numbers,
        run_name=args.run_name,
        explainer_model=args.explainer_model,
        n_tokens=args.n_tokens
    )

if __name__ == "__main__":
    main()
