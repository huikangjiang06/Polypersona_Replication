"""
Response generation script for PolyPersona using local models.
Generates synthetic persona-based survey responses using smaller models via transformers.
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Set Hugging Face cache directory
HF_CACHE_DIR = '/proj/arise/arise/hf_cache'
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR

# Create cache directory if it doesn't exist
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# System prompt template with placeholders
# Modify this string to customize instructions for generating synthetic responses. 
SYSTEM_PROMPT_TEMPLATE = """You are PolyPersona, a realistic survey respondent. You will answer questions based on the persona provided below.

Your task is to respond naturally and authentically as this persona would. Consider their background, values, traits, and interests when formulating your response.

**Persona:**
{persona}

**Instructions:**
- Answer concisely but realistically
- Stay consistent with the persona's characteristics
- For yes/no questions: Give a clear answer with one brief reason
- For likert scale questions: State your position and justify briefly
- For open questions: Provide 2-3 sentences maximum
- Be natural and conversational, not robotic"""

# Modify this string to customize the user prompt for each question.
USER_PROMPT_TEMPLATE = """**Domain:** {domain}
**Question Type:** {question_type}
**Question:** {question}

Provide your answer as this persona:"""


# This function formats the persona dictionary into a string.
# It follows the persona format used in poly.py
def format_persona(persona: Dict[str, Any], indent: str = '') -> str:
    """Format persona dict into readable text.
    
    Args:
        persona: Persona dictionary or string
        indent: Optional indentation prefix for each line (e.g., '  ')
    
    Returns:
        Formatted persona string
    """
    if isinstance(persona, str):
        return persona
    
    lines = []
    if 'age' in persona:
        lines.append(f"Age: {persona['age']}")
    if 'gender' in persona:
        lines.append(f"Gender: {persona['gender']}")
    if 'occupation' in persona:
        lines.append(f"Occupation: {persona['occupation']}")
    if 'education' in persona:
        lines.append(f"Education: {persona['education']}")
    if 'region' in persona:
        lines.append(f"Region: {persona['region']}")
    if 'values' in persona:
        values = persona['values'] if isinstance(persona['values'], list) else [persona['values']]
        lines.append(f"Values: {', '.join(map(str, values))}")
    if 'traits' in persona:
        traits = persona['traits'] if isinstance(persona['traits'], list) else [persona['traits']]
        lines.append(f"Traits: {', '.join(map(str, traits))}")
    if 'interests' in persona:
        interests = persona['interests'] if isinstance(persona['interests'], list) else [persona['interests']]
        lines.append(f"Interests: {', '.join(map(str, interests))}")
    if 'income_bracket' in persona:
        lines.append(f"Income: {persona['income_bracket']}")
    
    separator = f'\n{indent}' if indent else '\n'
    return separator.join(lines)


def load_json_file(path: str) -> List[Dict[str, Any]]:
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


def build_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages from example for model input."""
    persona = example.get('persona', {})
    question = example.get('question', '')
    domain = example.get('domain', 'general')
    question_type = example.get('question_type', 'open')
    
    persona_text = format_persona(persona)
    
    system_content = SYSTEM_PROMPT_TEMPLATE.format(persona=persona_text)
    # Fill in the user prompt with domain, question type, and question as they are provided in the dataset.
    user_content = USER_PROMPT_TEMPLATE.format(
        domain=domain,
        question_type=question_type,
        question=question
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def build_messages_array(example: Dict[str, Any], generated_response: str) -> List[Dict[str, str]]:
    """Build messages array in original dataset format."""
    persona = example.get('persona', {})
    question = example.get('question', '')
    domain = example.get('domain', 'general')
    
    persona_text = format_persona(persona, indent='  ')
    
    system_message = {
        "role": "system",
        "content": "You are PolyPersona, a survey respondent. Answer faithfully as the given persona."
    }
    
    user_message = {
        "role": "user",
        "content": f"Persona:\n  {persona_text}\n\nDomain: {domain}\nQuestion: {question}\nAnswer succinctly but realistically."
    }
    
    assistant_message = {
        "role": "assistant",
        "content": generated_response
    }
    
    return [system_message, user_message, assistant_message]


def load_model(model_name: str, device: str = "auto"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model in FP16 for efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model.eval()
    print(f"Model loaded successfully")
    print(f"Model device: {model.device}")
    return model, tokenizer


def extract_answer(text):
    """Extract answer part from generated text if it contains markers."""
    # Truncate to only keep content after "### Answer"
    if "### Answer" in text:
        return text.split("### Answer", 1)[1].strip()
    return text


def generate_responses_batch(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    temperature: float = 0.9,
    max_new_tokens: int = 256,
    top_p: float = 0.95
) -> List[str]:
    """Generate responses for a batch of examples in parallel."""
    # Build messages for all examples
    all_messages = [build_messages(ex) for ex in examples]
    
    # Apply chat template to all
    prompts = []
    for messages in all_messages:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
    
    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode - only the generated part (skip input tokens)
    generated_texts = []
    for i, output in enumerate(outputs):
        input_length = inputs['input_ids'][i].shape[0]
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        # Apply answer extraction
        generated = extract_answer(generated)
        generated_texts.append(generated)
    
    return generated_texts


def generate_batch(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """Generate responses for examples using batch processing."""
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(examples), batch_size), desc="Generating responses"):
        batch = examples[i:i+batch_size]
        
        try:
            # Generate for batch
            generated_texts = generate_responses_batch(
                batch, model, tokenizer, temperature, max_new_tokens, top_p
            )
            
            # Build results for each example
            for example, generated_text in zip(batch, generated_texts):
                messages_array = build_messages_array(example, generated_text)
                
                results.append({
                    'id': example.get('id', ''),
                    'domain': example.get('domain', ''),
                    'persona': example.get('persona', {}),
                    'question': example.get('question', ''),
                    'question_type': example.get('question_type', 'open'),
                    'reference': generated_text,
                    'messages': messages_array,
                    'meta': example.get('meta', {})
                })
        
        except Exception as e:
            print(f"\nBatch generation failed: {e}")
            # Fallback: use empty or original references
            for example in batch:
                fallback_reference = example.get('reference', '')
                messages_array = build_messages_array(example, fallback_reference)
                
                results.append({
                    'id': example.get('id', ''),
                    'domain': example.get('domain', ''),
                    'persona': example.get('persona', {}),
                    'question': example.get('question', ''),
                    'question_type': example.get('question_type', 'open'),
                    'reference': fallback_reference,
                    'messages': messages_array,
                    'meta': example.get('meta', {}),
                    '_error': str(e)
                })
        
        # Clear cache periodically
        if len(results) % 50 == 0:
            torch.cuda.empty_cache()
    
    return results


def load_all_data(data_dir: str, splits: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load data from specified splits."""
    data = {}
    for split in splits:
        path = Path(data_dir) / f"{split}.json"
        if path.exists():
            data[split] = load_json_file(str(path))
            print(f"Loaded {split}: {len(data[split])} examples")
        else:
            print(f"WARNING: {path} not found, skipping")
    return data


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSONL file in original dataset format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} results to {output_path}")


def print_statistics(all_results: Dict[str, List[Dict[str, Any]]]):
    """Print generation statistics."""
    print("\n" + "="*60)
    print("GENERATION STATISTICS")
    print("="*60)
    
    for split, results in all_results.items():
        total = len(results)
        successful = sum(1 for r in results if not r.get('_error'))
        failed = total - successful
        
        # Domain breakdown
        domain_counts = defaultdict(int)
        for r in results:
            if not r.get('_error'):
                domain_counts[r.get('domain', 'unknown')] += 1
        
        print(f"\n{split.upper()}:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Success rate: {successful/total*100:.1f}%")
        
        if domain_counts:
            print(f"  By domain:")
            for domain in sorted(domain_counts.keys()):
                print(f"    {domain}: {domain_counts[domain]}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic persona-based responses using local models"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data",
        help="Directory containing train/val/test JSON files"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./synthetic_data",
        help="Directory to save synthetic responses (in original dataset format)"
    )
    parser.add_argument(
        "--splits", 
        nargs="+", 
        default=["train", "val", "test"],
        help="Which splits to generate for (default: all)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL,
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.9,
        help="Sampling temperature (higher = more diverse)"
    )
    parser.add_argument(
        "--max-new-tokens", 
        type=int, 
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.95,
        help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--max-examples", 
        type=int, 
        default=None,
        help="Limit number of examples per split (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation (4-16 recommended depending on model size)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto)"
    )
    
    args = parser.parse_args()
    
    print(f"Response Generation for PolyPersona")
    print(f"="*60)
    print(f"Model: {args.model}")
    print(f"Cache directory: {HF_CACHE_DIR}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_new_tokens}")
    print(f"="*60)
    
    # Load model once
    model, tokenizer = load_model(args.model, args.device)
    
    # Load data
    all_data = load_all_data(args.data_dir, args.splits)
    
    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        return
    
    # Generate responses for each split
    all_results = {}
    
    for split, examples in all_data.items():
        print(f"\n{'='*60}")
        print(f"Generating for {split} set ({len(examples)} examples)")
        print(f"{'='*60}")
        
        # Optional: limit examples for testing
        if args.max_examples:
            examples = examples[:args.max_examples]
            print(f"Limited to {len(examples)} examples for testing")
        
        # Generate
        results = generate_batch(
            examples,
            model,
            tokenizer,
            args.model,
            args.temperature,
            args.max_new_tokens,
            args.top_p,
            args.batch_size
        )
        
        all_results[split] = results
        
        # Save immediately after each split (in original dataset format)
        output_file = Path(args.output_dir) / f"{split}.json"
        save_results(results, str(output_file))
    
    # Print statistics
    print_statistics(all_results)
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
