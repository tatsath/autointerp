#!/usr/bin/env python3
"""
Generate feature labels from extracted examples.
Uses the JSONL output from extract_examples.py to generate human-readable labels.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any


def generate_label_search_based(
    feature_id: int,
    examples: List[Dict[str, Any]],
    max_examples: int = 20
):
    """
    Generate a label using search-based approach - extract common keywords/patterns.
    
    Args:
        feature_id: Feature index
        examples: List of example dicts with 'text' and 'activation'
        max_examples: Maximum number of examples to analyze
    """
    if not examples:
        return "N/A"
    
    import re
    from collections import Counter
    
    # Sort by activation and take top examples
    sorted_examples = sorted(examples, key=lambda x: x.get('activation', 0.0), reverse=True)
    top_examples = sorted_examples[:max_examples]
    
    # Extract text from examples
    all_text = " ".join([ex['text'].lower() for ex in top_examples])
    
    # Financial keywords to look for
    financial_keywords = [
        'stock', 'price', 'market', 'trading', 'revenue', 'earnings', 'profit', 'loss',
        'dividend', 'share', 'equity', 'bond', 'investment', 'portfolio', 'volatility',
        'sentiment', 'analyst', 'buy', 'sell', 'hold', 'rating', 'target', 'forecast',
        'growth', 'decline', 'increase', 'decrease', 'quarter', 'annual', 'year',
        'company', 'firm', 'corporation', 'sector', 'industry', 'financial', 'economic',
        'currency', 'dollar', 'percent', 'percentage', 'million', 'billion', 'trillion'
    ]
    
    # Find common financial terms
    found_keywords = []
    for keyword in financial_keywords:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        count = len(re.findall(pattern, all_text))
        if count >= 2:  # Appears in at least 2 examples
            found_keywords.append((keyword, count))
    
    # Sort by frequency
    found_keywords.sort(key=lambda x: x[1], reverse=True)
    
    # Build label from top keywords (max 3-4 keywords, less than 10 words)
    if found_keywords:
        # Remove duplicates while preserving order
        seen = set()
        top_keywords = []
        for kw, count in found_keywords:
            if kw not in seen:
                seen.add(kw)
                top_keywords.append(kw)
            if len(top_keywords) >= 4:
                break
        
        # Create a readable label with better formatting
        if len(top_keywords) == 1:
            label = top_keywords[0].capitalize()
        elif len(top_keywords) == 2:
            label = f"{top_keywords[0].capitalize()} {top_keywords[1]}"
        elif len(top_keywords) == 3:
            label = f"{top_keywords[0].capitalize()} {top_keywords[1]} {top_keywords[2]}"
        else:
            label = f"{top_keywords[0].capitalize()} {top_keywords[1]} {top_keywords[2]} {top_keywords[3]}"
        
        # Add descriptive context if needed
        if 'price' in top_keywords and 'stock' not in top_keywords:
            label = f"Stock {label.lower()}"
        elif 'market' in top_keywords and 'sentiment' not in label.lower():
            if 'analysis' not in label.lower():
                label = f"Market {label.lower()}"
        
        # Limit to less than 10 words
        words = label.split()
        if len(words) > 9:
            label = " ".join(words[:9])
        
        return label
    else:
        # Fallback: extract most common words (excluding stopwords)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
        
        words = re.findall(r'\b[a-z]{3,}\b', all_text)
        word_counts = Counter([w for w in words if w not in stopwords])
        top_words = [word for word, count in word_counts.most_common(3)]
        
        if top_words:
            label = " ".join([w.capitalize() for w in top_words])
            return label[:50]  # Limit length
    
    return "Financial Pattern"


def generate_label_from_examples(
    feature_id: int,
    examples: List[Dict[str, Any]],
    explainer_model,
    explainer_tokenizer,
    device: str,
    max_examples: int = 10
):
    """
    Generate a label for a feature based on its activating examples using LLM.
    
    Args:
        feature_id: Feature index
        examples: List of example dicts with 'text' and 'activation'
        explainer_model: Model to use for label generation
        explainer_tokenizer: Tokenizer for the model
        device: Device to run on
        max_examples: Maximum number of examples to use in prompt
    """
    if not examples:
        return "N/A"
    
    # Sort by activation and take top examples
    sorted_examples = sorted(examples, key=lambda x: x.get('activation', 0.0), reverse=True)
    top_examples = sorted_examples[:max_examples]
    
    # Format examples for prompt
    activating_text = "\n".join([
        f"{i+1}. \"{ex['text'][:150]}...\" (activation: {ex.get('activation', 0.0):.4f})"
        for i, ex in enumerate(top_examples)
    ])
    
    prompt = f"""Analyze this neural network feature pattern from a financial language model.

Feature {feature_id} activates strongly on these examples:
{activating_text}

Provide ONLY a concise label (less than 10 words) describing what this feature detects in financial text. Just the label, nothing else:
Label:"""
    
    # Generate label
    inputs = explainer_tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = explainer_model.generate(
            **inputs,
            max_new_tokens=30,  # Increased to allow up to 10 words
            temperature=0.2,
            do_sample=True,
            pad_token_id=explainer_tokenizer.eos_token_id,
            eos_token_id=explainer_tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    response = explainer_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    label = response.strip()
    
    # Clean up label
    for prefix in ["Label:", "label:", "The label is", "The feature detects", "This feature detects"]:
        if prefix in label:
            label = label.split(prefix)[-1].strip()
    
    # Remove quotes and limit length
    label = label.strip('"').strip("'").strip()
    label = label.split('\n')[0].strip()
    
    # Limit to less than 10 words
    words = label.split()
    if len(words) > 10:
        label = " ".join(words[:10])
    
    if label.endswith('.') and len(label) > 10:
        label = label[:-1].strip()
    
    return label if label else "N/A"


def generate_labels(
    examples_jsonl_path: str,
    output_path: str,
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
    max_examples_per_feature: int = 10,
    use_same_model: bool = True
):
    """
    Generate labels for all features from a JSONL file of examples.
    
    Args:
        examples_jsonl_path: Path to JSONL file from extract_examples.py
        output_path: Path to save labeled features JSON file
        model_path: Model to use for label generation
        max_examples_per_feature: Max examples to use per feature in prompt
        use_same_model: If True, reuse model (saves memory)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")
    
    # Load examples
    print(f">>> Loading examples from {examples_jsonl_path}...")
    features_data = []
    with open(examples_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                features_data.append(json.loads(line))
    
    print(f">>> Found {len(features_data)} features")
    
    # Load model for label generation
    print(f">>> Loading model for label generation: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        if device == "cuda":
            model = model.to(device)
    model.eval()
    
    print(">>> Generating labels...")
    
    # Generate labels for each feature
    results = []
    for idx, feat_data in enumerate(features_data):
        feature_id = feat_data['feature_id']
        finance_score = feat_data['finance_score']
        examples = feat_data.get('examples', [])
        
        print(f"[{idx+1}/{len(features_data)}] Processing feature {feature_id}...")
        
        # Generate label using LLM approach
        try:
            label_llm = generate_label_from_examples(
                feature_id,
                examples,
                model,
                tokenizer,
                device,
                max_examples=max_examples_per_feature
            )
            print(f"    ✅ LLM Label: {label_llm}")
        except Exception as e:
            print(f"    ❌ LLM Error: {e}")
            label_llm = "N/A"
        
        # Generate label using search-based approach
        try:
            label_search = generate_label_search_based(
                feature_id,
                examples,
                max_examples=max_examples_per_feature
            )
            print(f"    ✅ Search Label: {label_search}")
        except Exception as e:
            print(f"    ❌ Search Error: {e}")
            label_search = "N/A"
        
        results.append({
            'feature_index': feature_id,
            'score': finance_score,
            'label_llm': label_llm,
            'label_search': label_search,
            'label': label_llm,  # Keep 'label' for backward compatibility
            'num_examples': len(examples)
        })
    
    # Save results
    output_data = {
        'num_features': len(results),
        'features': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f">>> Saved labels to {output_path}")

