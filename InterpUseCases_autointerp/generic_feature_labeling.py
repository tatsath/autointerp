#!/usr/bin/env python3
"""
Generic Feature Labeling System
Works with any model and any number of features
"""

import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import argparse

class GenericFeatureLabeler:
    def __init__(self, labeling_model_name="meta-llama/Llama-2-7b-chat-hf", output_dir="results"):
        """
        Initialize the generic feature labeler
        
        Args:
            labeling_model_name: Model to use for labeling (e.g., "meta-llama/Llama-2-7b-chat-hf")
            output_dir: Directory to save results
        """
        self.labeling_model_name = labeling_model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load the labeling model
        self.model, self.tokenizer = self.load_labeling_model()
    
    def load_labeling_model(self):
        """Load the model for feature labeling"""
        print(f"Loading labeling model: {self.labeling_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.labeling_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.labeling_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()
        
        return model, tokenizer
    
    def create_prompt(self, sentences_data, domain="financial"):
        """Create the prompt for feature labeling"""
        prompt_parts = []
        
        for i, sent_data in enumerate(sentences_data):
            sentence = sent_data['sentence']
            activation = sent_data['activation']
            sentence_type = sent_data['type']
            
            # Create highlighted sentence
            highlighted_sentence = f"<<{sentence}>>"
            prompt_parts.append(f"{highlighted_sentence} ({activation:.3f})")
        
        examples_text = "\n".join(prompt_parts)
        
        # Domain-specific prompts
        if domain == "financial":
            system_prompt = f"""Analyze the text examples and provide a SPECIFIC, GRANULAR phrase (≤8 words) describing what exact {domain} concept the neural network feature represents.

Focus on SPECIFIC {domain} domains, not broad categories. Examples of GOOD granular labels:
- "Corporate earnings reports and guidance"
- "Stock market trading and volatility"
- "Banking loan loss provisions"
- "Federal Reserve interest rate policy"
- "Cryptocurrency market capitalization"
- "Real estate investment trusts"
- "Insurance underwriting ratios"
- "Merger and acquisition deals"
- "Economic inflation indicators"
- "Investment portfolio management"

Examples of BAD broad labels (avoid these):
- "{domain.title()} data and metrics" (too broad)
- "{domain.title()} performance" (too generic)
- "{domain.title()} information" (not specific)

Text examples:
{{examples_text}}

Specific {domain} concept:"""
        else:
            system_prompt = f"""Analyze the text examples and provide a SPECIFIC, GRANULAR phrase (≤8 words) describing what exact {domain} concept the neural network feature represents.

Focus on SPECIFIC {domain} domains, not broad categories. Examples of GOOD granular labels:
- "Scientific research methodology"
- "Medical diagnosis procedures"
- "Legal contract analysis"
- "Educational curriculum design"
- "Technical documentation standards"

Examples of BAD broad labels (avoid these):
- "{domain.title()} data and metrics" (too broad)
- "{domain.title()} performance" (too generic)
- "{domain.title()} information" (not specific)

Text examples:
{{examples_text}}

Specific {domain} concept:"""
        
        return system_prompt.format(examples_text=examples_text)
    
    def generate_feature_label(self, prompt, max_length=100):
        """Generate a label for a feature using the model"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.3,  # Lower temperature for more focused responses
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode the response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Clean up the response - extract the key phrase
        explanation = response.strip()
        
        # Remove common prefixes and suffixes
        explanation = explanation.replace("Explanation:", "").replace("Phrase:", "").strip()
        
        # Try to extract the phrase in quotes first
        if '"' in explanation:
            # Extract text between quotes
            start = explanation.find('"')
            end = explanation.find('"', start + 1)
            if end > start:
                explanation = explanation[start+1:end]
        elif "'" in explanation:
            # Extract text between single quotes
            start = explanation.find("'")
            end = explanation.find("'", start + 1)
            if end > start:
                explanation = explanation[start+1:end]
        else:
            # If no quotes, take the first meaningful phrase
            # Split by common delimiters and take the first part
            for delimiter in ['.', 'Explanation:', 'Phrase:', '\n']:
                if delimiter in explanation:
                    explanation = explanation.split(delimiter)[0].strip()
                    break
            
            # Remove numbers and bullet points
            words = explanation.split()
            clean_words = []
            for word in words:
                # Skip numbers and bullet points
                if not word.isdigit() and not word in ['*', '-', '5.', '1.', '2.', '3.', '4.']:
                    clean_words.append(word)
            explanation = " ".join(clean_words)
        
        # Clean up
        explanation = explanation.strip('"').strip("'").strip()
        
        # Limit to reasonable length (≤10 words)
        words = explanation.split()
        if len(words) > 10:
            explanation = " ".join(words[:10])
        
        return explanation
    
    def label_features(self, feature_sentences, domain="financial", top_n=None):
        """Label the features"""
        print(f"Labeling features using {self.labeling_model_name}...")
        
        if top_n is None:
            top_n = len(feature_sentences)
        
        results = []
        
        for i, (feature_idx_str, sentences_data) in enumerate(feature_sentences.items()):
            if i >= top_n:
                break
                
            feature_idx = int(feature_idx_str)
            print(f"  Processing feature {feature_idx} ({i+1}/{min(top_n, len(feature_sentences))})...")
            
            # Create prompt
            prompt = self.create_prompt(sentences_data, domain)
            
            # Generate label
            try:
                label = self.generate_feature_label(prompt)
                print(f"    Label: {label}")
                if len(label) < 5:
                    print(f"    Warning: Very short label generated")
            except Exception as e:
                print(f"    Error generating label: {e}")
                label = "Error generating label"
            
            # Get feature stats
            financial_sentences = [s for s in sentences_data if s['type'] == 'financial']
            general_sentences = [s for s in sentences_data if s['type'] == 'general']
            
            avg_financial_activation = sum(s['activation'] for s in financial_sentences) / len(financial_sentences) if financial_sentences else 0
            avg_general_activation = sum(s['activation'] for s in general_sentences) / len(general_sentences) if general_sentences else 0
            specialization = avg_financial_activation - avg_general_activation
            
            results.append({
                'feature_number': feature_idx,
                'label': label,
                'financial_activation': avg_financial_activation,
                'general_activation': avg_general_activation,
                'specialization': specialization,
                'num_financial_examples': len(financial_sentences),
                'num_general_examples': len(general_sentences)
            })
        
        return results
    
    def run_labeling(self, analysis_file, domain="financial", top_n=None):
        """Run the complete feature labeling process"""
        print("="*80)
        print("GENERIC FEATURE LABELING")
        print("="*80)
        print(f"Labeling Model: {self.labeling_model_name}")
        print(f"Domain: {domain}")
        print(f"Analysis File: {analysis_file}")
        print("="*80)
        
        # Load analysis results
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        feature_sentences = analysis_data['feature_sentences']
        top_features = analysis_data['top_features']
        
        print(f"Loaded analysis for {len(feature_sentences)} features")
        
        # Label features
        results = self.label_features(feature_sentences, domain, top_n)
        
        # Create results DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('specialization', ascending=False)
        
        # Save clean results (just feature and label)
        clean_df = df[['feature_number', 'label']].copy()
        clean_output = self.output_dir / f'feature_labels_clean_{domain}.csv'
        clean_df.to_csv(clean_output, index=False)
        
        # Save detailed results
        detailed_output = self.output_dir / f'feature_labels_detailed_{domain}.csv'
        df.to_csv(detailed_output, index=False)
        
        print(f"\n✅ Labeling complete!")
        print(f"Results saved to: {clean_output}")
        print(f"Detailed results saved to: {detailed_output}")
        
        print(f"\nTop labeled features:")
        print(clean_df.head(10).to_string(index=False, max_colwidth=60))
        
        return clean_df, df

def main():
    parser = argparse.ArgumentParser(description="Generic Feature Labeling")
    parser.add_argument("--analysis_file", required=True, help="Path to analysis results JSON file")
    parser.add_argument("--labeling_model", default="meta-llama/Llama-2-7b-chat-hf", help="Model for labeling")
    parser.add_argument("--domain", default="financial", help="Domain for labeling (financial, medical, legal, etc.)")
    parser.add_argument("--top_n", type=int, help="Number of top features to label (default: all)")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create labeler
    labeler = GenericFeatureLabeler(
        labeling_model_name=args.labeling_model,
        output_dir=args.output_dir
    )
    
    # Run labeling
    clean_df, detailed_df = labeler.run_labeling(
        analysis_file=args.analysis_file,
        domain=args.domain,
        top_n=args.top_n
    )

if __name__ == "__main__":
    main()
