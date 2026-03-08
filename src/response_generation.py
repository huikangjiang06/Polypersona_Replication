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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Create cache directory if it doesn't exist
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# This function formats the persona dictionary into a string.
# Matches persona_to_text from poly_consistent_prompt.py
def persona_to_text(persona) -> str:
    """Format persona dict into readable text (semicolon-separated).
    Matches the format used in poly_consistent_prompt.py.
    
    Args:
        persona: Persona dictionary or string
    
    Returns:
        Formatted persona string (semicolon-separated)
    """
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


def build_prompt(persona_text: str, question: str, qtype: str = None) -> str:
    """Build prompt matching poly_consistent_prompt.py format.
    
    Args:
        persona_text: Formatted persona string
        question: Question text
        qtype: Question type (yesno, likert, agreement, open)
    
    Returns:
        Formatted prompt string
    """
    SYSTEM_PROMPT = (
        "You are PolyPersona, a helpful and realistic survey respondent. "
        "Answer faithfully based on the given persona."
        "ACT OUT THE PERSONA AS REALISTICALLY AS POSSIBLE."
        "YOU MUST STRICTLY FOLLOW THE INSTRUCTIONS AND FORMAT."
    )

    # short behavioral hints per question type
    if qtype == "yesno":
        hint = "Respond with 'Yes.' or 'No.' and add one short reason."
    elif qtype == "likert":
        hint = "Respond on a 5-point Likert scale (Strongly Disagree → Strongly Agree) and justify briefly."
    elif qtype == "agreement":
        hint = "Indicate your level of agreement and explain in one line."
    else:
        hint = "Answer naturally and concisely from the persona's perspective."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Persona: {persona_text}\n"
        f"Question ({qtype or 'open'}): {question}\n"
        f"{hint}\nAnswer:"
    )


def build_messages(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages from example for model input.
    Uses proper system/user message format for Qwen chat template.
    """
    persona = example.get('persona', {})
    question = example.get('question', '')
    question_type = example.get('question_type', 'open')
    
    persona_text = persona_to_text(persona)
    
    # Build system message with instructions
    system_content = (
        "You are PolyPersona, a helpful and realistic survey respondent. "
        "Answer faithfully based on the given persona. "
        "ACT OUT THE PERSONA AS REALISTICALLY AS POSSIBLE. "
        "YOU MUST STRICTLY FOLLOW THE INSTRUCTIONS AND FORMAT. "
        "YOU MUST DIRECTLY ANSWER THE QUESTION IN THE DESIRED FORMAT AND EXPLAIN IN AROUND FIVE TO SEVEN SENTENCES."
        "BE DETAILED AND NARRATIVE IN YOUR RESPONSES."
    )
    
    # Build user message with persona and question
    # Short behavioral hints per question type
    if question_type == "yesno":
        hint = "Respond with 'Yes.' or 'No.' and add one short reason."
    elif question_type == "likert":
        hint = "Respond on a 5-point Likert scale (Strongly Disagree → Strongly Agree) and justify briefly."
    elif question_type == "agreement":
        hint = "Indicate your level of agreement and explain in one line."
    else:
        hint = "Answer naturally and concisely from the persona's perspective."
    
    user_content = (
        f"Persona: {persona_text}\n"
        f"Question ({question_type or 'open'}): {question}\n"
        f"{hint}\nAnswer:"
    )
    
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def build_messages_array(example: Dict[str, Any], generated_response: str) -> List[Dict[str, str]]:
    """Build messages array in dataset format matching training data.
    Uses the same format as poly_consistent_prompt.py for consistency.
    """
    persona = example.get('persona', {})
    question = example.get('question', '')
    domain = example.get('domain', 'general')
    question_type = example.get('question_type', 'open')
    
    # Use semicolon-separated persona format
    persona_text = persona_to_text(persona)
    
    system_message = {
        "role": "system",
        "content": "You are PolyPersona, a survey respondent. Answer faithfully as the given persona."
    }
    
    # Format user message to match training format
    # Note: Using newline-separated format for the messages array (for readability in dataset)
    # but generation uses the semicolon format
    persona_lines = []
    p = example.get('persona', {})
    if 'age' in p: persona_lines.append(f"Age: {p['age']}")
    if 'gender' in p: persona_lines.append(f"Gender: {p['gender']}")
    if 'occupation' in p: persona_lines.append(f"Occupation: {p['occupation']}")
    if 'education' in p: persona_lines.append(f"Education: {p['education']}")
    if 'region' in p: persona_lines.append(f"Region: {p['region']}")
    if 'values' in p:
        vals = p['values'] if isinstance(p['values'], list) else [p['values']]
        persona_lines.append(f"Values: {', '.join(map(str, vals))}")
    if 'traits' in p:
        traits = p['traits'] if isinstance(p['traits'], list) else [p['traits']]
        persona_lines.append(f"Traits: {', '.join(map(str, traits))}")
    if 'interests' in p:
        ints = p['interests'] if isinstance(p['interests'], list) else [p['interests']]
        persona_lines.append(f"Interests: {', '.join(map(str, ints))}")
    if 'income_bracket' in p: persona_lines.append(f"Income: {p['income_bracket']}")
    
    persona_display = "\n  ".join(persona_lines)
    
    user_message = {
        "role": "user",
        "content": f"Persona:\n  {persona_display}\n\nDomain: {domain}\nQuestion: {question}\nAnswer succinctly but realistically."
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
    
    # Set padding side to left for decoder-only models (required for correct batch generation)
    tokenizer.padding_side = 'left'
    
    # Ensure pad token is set (some models don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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


def extract_answer(text: str) -> str:
    """Extract answer part from generated text.
    Since prompt ends with 'Answer:', the model should generate the answer directly.
    Clean up any remaining prompt fragments.
    """
    # The model should generate directly after the prompt
    # But sometimes it repeats parts of the prompt, so clean that up
    
    # Remove any prompt repetition
    if "Answer:" in text:
        # Take everything after the last "Answer:"
        text = text.split("Answer:")[-1].strip()
    
    # Remove any markdown headers that might appear
    if "###" in text:
        text = text.split("###")[0].strip()
    
    # Truncate at newlines (keep only first line/paragraph for conciseness)
    lines = text.split('\n')
    # Keep first non-empty line
    for line in lines:
        line = line.strip()
        if line:
            return line
    
    return text.strip()


def generate_responses_batch(
    examples: List[Dict[str, Any]],
    model,
    tokenizer,
    temperature: float = 0.9,
    max_new_tokens: int = 256,
    top_p: float = 0.95
) -> List[str]:
    """Generate responses for a batch of examples in parallel."""
    # Build chat messages using proper format for instruct models (especially Qwen)
    all_messages = []
    for ex in examples:
        messages = build_messages(ex)
        all_messages.append(messages)
    
    # Apply chat template to each message list
    # This adds proper special tokens (<|im_start|>, <|im_end|>, etc.) that Qwen expects
    prompts = [
        tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True  # Adds <|im_start|>assistant to prompt response
        )
        for messages in all_messages
    ]
    
    # Tokenize with padding (use larger max_length to avoid truncation)
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096  # Increased to avoid truncating prompts
    )
    
    # Check if any prompts were truncated
    if any(len(ids) == 4096 for ids in inputs['input_ids']):
        print("WARNING: Some prompts were truncated. Consider increasing max_length.")
    
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
    # With left-padding, all inputs in batch have same length (includes padding)
    input_length = inputs['input_ids'].shape[1]
    
    generated_texts = []
    for i, output in enumerate(outputs):
        # Skip the entire input (including padding) to get only generated tokens
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
        
        # Clean up: take only first line/sentence for conciseness
        # The model should generate directly after "Answer:"
        if '\n' in generated:
            lines = [l.strip() for l in generated.split('\n') if l.strip()]
            generated = lines[0] if lines else generated
        
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
        default="./outputs/experiment_11_synthetic_data",
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
        default=0.01,
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
        default=0.3,
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
        default=64,
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
