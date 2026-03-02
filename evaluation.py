#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for PolyPersona model.
Loads fine-tuned model and calculates BLEU, ROUGE, and BERTScore on val/test sets.
"""

import os
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd
import warnings

# Suppress warnings
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# BERTScore
BERTSCORE_AVAILABLE = False
BERTSCORE_ERROR_SHOWN = False
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
    print("BERTScore loaded successfully")
except ImportError as e:
    print(f"WARNING: bert_score not available ({e}), will skip BERTScore calculations")


def load_json_file(path):
    """Load JSONL or JSON file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Try as full JSON
                f.seek(0)
                records = json.load(f)
                break
    return records


def persona_to_text(persona):
    """Convert persona dict to readable text."""
    if isinstance(persona, str):
        return persona
    if isinstance(persona, dict):
        parts = []
        for k, v in persona.items():
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            parts.append(f"{k}: {v}")
        return "; ".join(parts)
    return str(persona)


def build_prompt(persona_text, question):
    """Build the prompt for generation."""
    SYSTEM_PROMPT = (
        "You are a survey respondent. Answer as a consistent persona given below. "
        "Be concise and realistic. If the question is multiple-choice, pick the most fitting option and give one short reason."
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Persona: {persona_text}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def load_model_and_tokenizer(model_dir, base_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load base model + LoRA adapter using safetensors format."""
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_safetensors=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load base model with safetensors - on GPU if available for compatibility with PEFT
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        use_safetensors=True  # Force safetensors format
    )
    
    # Load LoRA adapter if available
    adapter_path = Path(model_dir)
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        print(f"Loading LoRA adapter from: {adapter_path}")
        try:
            model = PeftModel.from_pretrained(base_model, str(adapter_path))
            print("Merging LoRA weights...")
            model = model.merge_and_unload()  # Merge for faster inference
            print("LoRA adapter loaded and merged successfully")
        except Exception as e:
            print(f"ERROR loading adapter: {e}")
            print("Falling back to base model only")
            model = base_model
    else:
        print(f"WARNING: No adapter found at {adapter_path}, using base model only")
        model = base_model
    
    model.eval()
    return model, tokenizer


def generate_response(prompt, model, tokenizer, max_new_tokens=256):
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer part
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    
    return text


def calculate_bleu(reference, prediction):
    """Calculate BLEU score."""
    reference_tokens = reference.lower().split()
    prediction_tokens = prediction.lower().split()
    
    # Use smoothing to avoid zero scores
    smoothing = SmoothingFunction().method1
    
    try:
        score = sentence_bleu([reference_tokens], prediction_tokens, smoothing_function=smoothing)
    except:
        score = 0.0
    
    return score


def calculate_rouge(reference, prediction, rouge_types=['rouge1', 'rouge2', 'rougeL']):
    """Calculate ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    scores = scorer.score(reference, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure,
    }

def calculate_bertscore_batch(references, predictions, model_type='bert-base-uncased'):
    """Calculate BERTScore for a batch using safetensors.
    
    Uses bert-base-uncased instead of deberta-xlarge-mnli as it's lighter
    and more reliably supports safetensors format.
    """
    global BERTSCORE_ERROR_SHOWN
    
    if not BERTSCORE_AVAILABLE:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    try:
        # Use bert_score with safetensors-compatible settings
        P, R, F1 = bert_score_fn(
            predictions, 
            references, 
            model_type=model_type, 
            verbose=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            batch_size=8,  # Smaller batch for stability
            rescale_with_baseline=False  # Faster, avoid extra downloads
        )
        return {
            'precision': float(P.mean().item()),
            'recall': float(R.mean().item()),
            'f1': float(F1.mean().item()),
        }
    except Exception as e:
        # Only print error once using global flag
        if not BERTSCORE_ERROR_SHOWN:
            print(f"\nWARNING: BERTScore calculation failed: {e}")
            print("Continuing without BERTScore. Use --skip-bertscore to suppress this warning.\n")
            BERTSCORE_ERROR_SHOWN = True
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def evaluate_split(data, model, tokenizer, split_name, batch_size=32, skip_bertscore=False):
    """Evaluate model on a data split."""
    print(f"\n{'='*60}")
    print(f"Evaluating {split_name} set ({len(data)} examples)")
    print(f"{'='*60}")
    
    results = []
    all_references = []
    all_predictions = []
    domain_results = defaultdict(lambda: {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': []})
    
    for ex in tqdm(data, desc=f"Generating {split_name}"):
        persona_text = persona_to_text(ex.get('persona', {}))
        question = ex.get('question', '')
        reference = ex.get('reference', '')
        domain = ex.get('domain', 'unknown')
        
        # Skip if no reference
        if not reference or reference == "N/A":
            continue
        
        # Generate prediction
        prompt = build_prompt(persona_text, question)
        try:
            prediction = generate_response(prompt, model, tokenizer)
        except Exception as e:
            print(f"Generation failed for {ex.get('id', 'unknown')}: {e}")
            prediction = ""
        
        # Calculate metrics
        bleu = calculate_bleu(reference, prediction)
        rouge = calculate_rouge(reference, prediction)
        
        # Store for batch BERTScore
        all_references.append(reference)
        all_predictions.append(prediction)
        
        # Store by domain
        domain_results[domain]['bleu'].append(bleu)
        domain_results[domain]['rouge1'].append(rouge['rouge1'])
        domain_results[domain]['rouge2'].append(rouge['rouge2'])
        domain_results[domain]['rougeL'].append(rouge['rougeL'])
        
        results.append({
            'id': ex.get('id', ''),
            'domain': domain,
            'reference': reference,
            'prediction': prediction,
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL'],
        })
    
    # Calculate BERTScore in batches (optional)
    bertscore_results = []
    if not skip_bertscore:
        print("\nCalculating BERTScore...")
        for i in range(0, len(all_references), batch_size):
            batch_refs = all_references[i:i+batch_size]
            batch_preds = all_predictions[i:i+batch_size]
            batch_scores = calculate_bertscore_batch(batch_refs, batch_preds)
            bertscore_results.extend([batch_scores['f1']] * len(batch_refs))
        
        # Add BERTScore to results
        for i, result in enumerate(results):
            if i < len(bertscore_results):
                result['bertscore_f1'] = bertscore_results[i]
                domain = result['domain']
                if 'bertscore_f1' not in domain_results[domain]:
                    domain_results[domain]['bertscore_f1'] = []
                domain_results[domain]['bertscore_f1'].append(bertscore_results[i])
    else:
        print("\nSkipping BERTScore calculation (--skip-bertscore enabled)")
        for result in results:
            result['bertscore_f1'] = 0.0
    
    # Calculate overall metrics
    overall = {
        'split': split_name,
        'n_examples': len(results),
        'bleu': np.mean([r['bleu'] for r in results]),
        'rouge1': np.mean([r['rouge1'] for r in results]),
        'rouge2': np.mean([r['rouge2'] for r in results]),
        'rougeL': np.mean([r['rougeL'] for r in results]),
        'bertscore_f1': np.mean(bertscore_results) if bertscore_results else 0.0,
    }
    
    # Calculate domain-wise metrics
    domain_metrics = {}
    for domain, scores in domain_results.items():
        domain_metrics[domain] = {
            'n_examples': len(scores['bleu']),
            'bleu': np.mean(scores['bleu']),
            'rouge1': np.mean(scores['rouge1']),
            'rouge2': np.mean(scores['rouge2']),
            'rougeL': np.mean(scores['rougeL']),
            'bertscore_f1': np.mean(scores.get('bertscore_f1', [0.0])),
        }
    
    return overall, domain_metrics, results


def print_results(overall, domain_metrics):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS - {overall['split'].upper()}")
    print(f"{'='*60}")
    print(f"Examples:      {overall['n_examples']}")
    print(f"BLEU:          {overall['bleu']:.4f}")
    print(f"ROUGE-1:       {overall['rouge1']:.4f}")
    print(f"ROUGE-2:       {overall['rouge2']:.4f}")
    print(f"ROUGE-L:       {overall['rougeL']:.4f}")
    print(f"BERTScore F1:  {overall['bertscore_f1']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"DOMAIN-WISE RESULTS")
    print(f"{'='*60}")
    
    # Sort by domain name
    for domain in sorted(domain_metrics.keys()):
        metrics = domain_metrics[domain]
        print(f"\n{domain} (n={metrics['n_examples']})")
        print(f"  BLEU:         {metrics['bleu']:.4f}")
        print(f"  ROUGE-1:      {metrics['rouge1']:.4f}")
        print(f"  ROUGE-2:      {metrics['rouge2']:.4f}")
        print(f"  ROUGE-L:      {metrics['rougeL']:.4f}")
        print(f"  BERTScore F1: {metrics['bertscore_f1']:.4f}")


def save_results(overall_results, domain_results, detailed_results, output_dir):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save overall results
    overall_df = pd.DataFrame(overall_results)
    overall_path = output_dir / "overall_metrics.csv"
    overall_df.to_csv(overall_path, index=False)
    print(f"\nSaved overall metrics to: {overall_path}")
    
    # Save domain results
    domain_rows = []
    for split_name, domains in domain_results.items():
        for domain, metrics in domains.items():
            row = {'split': split_name, 'domain': domain}
            row.update(metrics)
            domain_rows.append(row)
    
    domain_df = pd.DataFrame(domain_rows)
    domain_path = output_dir / "domain_metrics.csv"
    domain_df.to_csv(domain_path, index=False)
    print(f"Saved domain metrics to: {domain_path}")
    
    # Save detailed results (predictions)
    for split_name, results in detailed_results.items():
        detail_path = output_dir / f"{split_name}_predictions.jsonl"
        with open(detail_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"Saved {split_name} predictions to: {detail_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PolyPersona model")
    parser.add_argument("--model-dir", type=str, default="./outputs/personaverse",
                        help="Directory containing model checkpoint")
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model name")
    parser.add_argument("--data-dir", type=str, default="./generated_data_converted",
                        help="Directory containing train/val/test JSON files")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--splits", nargs="+", default=["val", "test"],
                        help="Which splits to evaluate (default: val, test)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Maximum examples per split (for testing)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for BERTScore calculation")
    parser.add_argument("--skip-bertscore", action="store_true",
                        help="Skip BERTScore calculation (useful if torch<2.6)")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model)
    
    # Evaluate each split
    overall_results = []
    domain_results = {}
    detailed_results = {}
    
    for split in args.splits:
        data_path = Path(args.data_dir) / f"{split}.json"
        if not data_path.exists():
            print(f"WARNING: {data_path} not found, skipping")
            continue
        
        # Load data
        data = load_json_file(data_path)
        if args.max_examples:
            data = data[:args.max_examples]
        
        # Evaluate
        overall, domain_metrics, results = evaluate_split(
            data, model, tokenizer, split, batch_size=args.batch_size,
            skip_bertscore=args.skip_bertscore
        )
        
        overall_results.append(overall)
        domain_results[split] = domain_metrics
        detailed_results[split] = results
        
        # Print results
        print_results(overall, domain_metrics)
    
    # Save all results
    save_results(overall_results, domain_results, detailed_results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
