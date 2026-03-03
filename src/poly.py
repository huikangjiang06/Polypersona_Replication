
import os
from pathlib import Path

# Set Hugging Face cache directory
HF_CACHE_DIR = '/proj/arise/arise/hj2742'
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR

# Create cache directory if it doesn't exist
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)

from dataclasses import dataclass, asdict
from typing import List, Optional
import math, random, json, pathlib, time
import pandas as pd
import numpy as np

@dataclass
class Config:
    # Data
    dataset_dir: str = "./data/personaverse"   
    train_file: str = "./generated_data_2/train.json"
    val_file: str = "./generated_data_2/val.json"
    test_file: str = "./generated_data_2/test.json"
    text_fields: dict = None  

    # Model
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
    torch_dtype: str = "auto"
    use_4bit: bool = False

    # LoRA / PEFT
    do_lora_finetune: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = None  

    # Training
    output_dir: str = "./outputs/personaverse"
    seed: int = 42
    num_epochs: int = 3
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    max_grad_norm: float = 0.3
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    max_samples_per_domain: Optional[int] = None  
    # Generation
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    
    do_generate_after_training: bool = False

    # Evaluation
    do_bertscore: bool = True
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  
    persona_consistency_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

cfg = Config()
if cfg.text_fields is None:
    cfg.text_fields = {
        "id": "id",
        "domain": "domain",
        "persona": "persona",  # dict or str
        "question": "question",
        "reference": "reference",  # optional gold
        "messages": "messages"     # optional ChatML format
    }

os.makedirs(cfg.output_dir, exist_ok=True)
print(cfg)

def build_prompt(persona_text, question, qtype=None):
    """Constructs the system–user prompt with qtype hints."""
    SYSTEM_PROMPT = (
        "You are PolyPersona, a helpful and realistic survey respondent. "
        "Answer faithfully based on the given persona."
    )

    # short behavioral hints per question type
    if qtype == "yesno":
        hint = "Respond with 'Yes.' or 'No.' and add one short reason."
    elif qtype == "likert":
        hint = "Respond on a 5-point Likert scale (Strongly Disagree → Strongly Agree) and justify briefly."
    elif qtype == "agreement":
        hint = "Indicate your level of agreement and explain in one line."
    else:
        hint = "Answer naturally and concisely from the persona’s perspective."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Persona: {persona_text}\n"
        f"Question ({qtype or 'open'}): {question}\n"
        f"{hint}\nAnswer:"
    )

def tokenize_sft(batch, tokenizer):
    """Tokenizes prompt–answer pairs; masks prompt tokens with -100."""
    input_ids, attn_masks, labels = [], [], []

    for p, a in zip(batch["prompt"], batch["answer"]):
        tok_p = tokenizer(p, truncation=True, add_special_tokens=True)
        tok_a = tokenizer(a, truncation=True, add_special_tokens=False)

        ids = tok_p["input_ids"] + tok_a["input_ids"] + [tokenizer.eos_token_id]
        lbls = [-100] * len(tok_p["input_ids"]) + tok_a["input_ids"] + [tokenizer.eos_token_id]

        input_ids.append(ids)
        attn_masks.append([1] * len(ids))
        labels.append(lbls)

    return {"input_ids": input_ids, "attention_mask": attn_masks, "labels": labels}


def to_text_pair(ex, cfg):
    """Extracts prompt and target answer from one example."""
    persona = ex.get(cfg.text_fields["persona"], {})
    persona_text = json.dumps(persona, ensure_ascii=False)
    question = ex.get(cfg.text_fields["question"], "")
    qtype = ex.get("question_type", "open")
    ref = ex.get(cfg.text_fields["reference"], "")
    ref = "" if ref in ["N/A", None] else ref
    prompt = build_prompt(persona_text, question, qtype)
    return {"prompt": prompt, "answer": ref, "qtype": qtype}

def decode_params(qtype):
    """Returns temperature / top_p for deterministic vs. open responses."""
    if qtype in {"yesno", "likert", "agreement"}:
        return dict(temperature=0.0, top_p=1.0, do_sample=False, max_new_tokens=64)
    return dict(temperature=0.7, top_p=0.9, do_sample=True, max_new_tokens=256)


import json, itertools, pathlib
from typing import Dict, Any, Iterable

def load_json_file(path: str) -> list:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    try:
        # Try json list
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "data" in data:
            return data["data"]
        else:
            return [data]
    except json.JSONDecodeError:
        # Try jsonlines
        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

def load_split(cfg: Config, split: str) -> list:
    file_map = {"train": cfg.train_file, "val": cfg.val_file, "test": cfg.test_file}
    path = file_map[split]  # Use the file path directly, don't join with dataset_dir
    data = load_json_file(path)
    # optional downsample per domain for quick experiments
    if cfg.max_samples_per_domain:
        by_dom = {}
        for ex in data:
            dom = ex.get(cfg.text_fields["domain"], "unknown")
            by_dom.setdefault(dom, []).append(ex)
        sampled = []
        for dom, items in by_dom.items():
            sampled.extend(items[:cfg.max_samples_per_domain])
        data = sampled
    print(f"Loaded {split}: {len(data)} examples from {path}")
    return data

train_data = load_split(cfg, "train")
val_data   = load_split(cfg, "val")
test_data  = load_split(cfg, "test")

# Quick diagnostics: counts and how many examples contain a non-empty reference (gold answer)
def _count_refs(split, cfg):
    n_total = len(split)
    r_key = cfg.text_fields.get("reference", "reference")
    n_with_ref = sum(1 for ex in split if ex.get(r_key) and ex.get(r_key) != "N/A")
    return n_total, n_with_ref

for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
    total, with_ref = _count_refs(split, cfg)
    print(f"Dataset {name}: {total} examples, {with_ref} with non-empty reference (gold)\n")

if len(train_data) > 0:
    # print one example to sanity-check field mapping
    sample = train_data[0]
    print("Sample train example keys:", list(sample.keys()))
    r_key = cfg.text_fields.get("reference", "reference")
    print("Sample reference value:", sample.get(r_key))

# For supervised fine-tuning we should only use examples that have a gold reference.
# Create labeled subsets and report sizes; the training function will use these.
r_key = cfg.text_fields.get("reference", "reference")
labeled_train = [ex for ex in train_data if ex.get(r_key) and ex.get(r_key) != "N/A"]
labeled_val   = [ex for ex in val_data   if ex.get(r_key) and ex.get(r_key) != "N/A"]
print(f"Using {len(labeled_train)} labeled examples for training and {len(labeled_val)} for validation.")

# If missing, create a tiny mock dataset for quick testing
if len(train_data) == 0 and len(val_data) == 0 and len(test_data) == 0:
    print("No dataset found. Creating a small mock set for demonstration.")
    mock = []
    personas = [
        {"age": 28, "gender": "female", "profession": "nurse", "values": ["empathy","community"]},
        {"age": 45, "gender": "male", "profession": "software engineer", "values": ["efficiency","privacy"]},
    ]
    questions = [
        ("healthcare", "How satisfied are you with your recent hospital visit? Please explain."),
        ("technology", "Do you prefer using cloud-based tools for work? Why or why not?")
    ]
    for i, (dom, q) in enumerate(questions):
        for j, p in enumerate(personas):
            mock.append({
                "id": f"mock-{i}-{j}",
                "domain": dom,
                "persona": p,
                "question": q,
                "reference": "N/A",
                "messages": []  # we will build prompts directly
            })
    train_data, val_data, test_data = mock[:2], mock[2:3], mock[3:]
    print(f"Mock splits -> train:{len(train_data)} val:{len(val_data)} test:{len(test_data)}")

    
def persona_to_text(persona) -> str:
    if isinstance(persona, str):
        return persona
    if isinstance(persona, dict):
        parts = []
        for k,v in persona.items():
            if isinstance(v, list):
                v = ", ".join(map(str,v))
            parts.append(f"{k}: {v}")
        return "; ".join(parts)
    return str(persona)

SYSTEM_PROMPT = (
    "You are a survey respondent. Answer as a consistent persona given below. "
    "Be concise and realistic. If the question is multiple-choice, pick the most fitting option and give one short reason."
)

def build_prompt(persona_text: str, question: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        f"Persona: {persona_text}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

# quick check
ex = train_data[0]
print(build_prompt(persona_to_text(ex.get(cfg.text_fields['persona'])), ex.get(cfg.text_fields['question'])))



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(cfg: Config):
    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.torch_dtype=="auto" else getattr(torch, cfg.torch_dtype)
        )
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype="auto" if cfg.torch_dtype=="auto" else getattr(torch, cfg.torch_dtype)
    )
    return model, tok

model, tokenizer = None, None
try:
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("Model loaded.")
except Exception as e:
    print("Model loading skipped (offline environment). Error:", e)


def _tokenize_pairs(batch):
    """Tokenize prompt–answer pairs with padding and correct label masking."""
    input_ids, attention_masks, labels = [], [], []
    for p, a in zip(batch["prompt"], batch["answer"]):
        tok_p = tokenizer(p, truncation=True, padding=False, add_special_tokens=True)
        tok_a = tokenizer(a, truncation=True, padding=False, add_special_tokens=False)

        # Combine prompt + answer
        ids = tok_p["input_ids"] + tok_a["input_ids"] + [tokenizer.eos_token_id]
        lbls = [-100] * len(tok_p["input_ids"]) + tok_a["input_ids"] + [tokenizer.eos_token_id]

        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
        labels.append(lbls)

    # Pad to the same length for batching
    max_len = max(len(x) for x in input_ids)
    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])
        input_ids[i] += [tokenizer.pad_token_id] * pad_len
        attention_masks[i] += [0] * pad_len
        labels[i] += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

    
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(cfg: Config):
    quant_config = None
    if cfg.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.torch_dtype=="auto" else getattr(torch, cfg.torch_dtype)
        )
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype="auto" if cfg.torch_dtype=="auto" else getattr(torch, cfg.torch_dtype)
    )
    return model, tok

model, tokenizer = None, None
try:
    model, tokenizer = load_model_and_tokenizer(cfg)
    print("Model loaded.")
except Exception as e:
    print("Model loading skipped (offline environment). Error:", e)


def _tokenize_pairs(batch):
    """Tokenize prompt–answer pairs with padding and correct label masking."""
    input_ids, attention_masks, labels = [], [], []
    for p, a in zip(batch["prompt"], batch["answer"]):
        tok_p = tokenizer(p, truncation=True, padding=False, add_special_tokens=True)
        tok_a = tokenizer(a, truncation=True, padding=False, add_special_tokens=False)

        # Combine prompt + answer
        ids = tok_p["input_ids"] + tok_a["input_ids"] + [tokenizer.eos_token_id]
        lbls = [-100] * len(tok_p["input_ids"]) + tok_a["input_ids"] + [tokenizer.eos_token_id]

        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
        labels.append(lbls)

    # Pad to the same length for batching
    max_len = max(len(x) for x in input_ids)
    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])
        input_ids[i] += [tokenizer.pad_token_id] * pad_len
        attention_masks[i] += [0] * pad_len
        labels[i] += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

import inspect
from transformers import TrainingArguments

def build_training_args(cfg, has_val: bool):
    # prefer evaluating at end of each epoch for small datasets so we can observe
    # validation loss/metrics more predictably (can be changed to 'steps')
    eval_strategy = "epoch" if has_val else "no"
    kw = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy=eval_strategy,                 # old TF may not have this
        eval_steps=cfg.eval_steps if has_val else None,
        save_total_limit=getattr(cfg, "save_total_limit", 2),
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        gradient_checkpointing=getattr(cfg, "grad_ckpt", False),
        bf16=(getattr(cfg, "torch_dtype", "auto") in ["bfloat16","auto"] and torch.cuda.is_available()),
        fp16=(getattr(cfg, "torch_dtype", "auto") == "float16" and torch.cuda.is_available()),
        report_to="none",
        max_grad_norm=getattr(cfg, "max_grad_norm", 1.0),
    )
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kw.items() if (k in allowed and v is not None)}
    if "evaluation_strategy" not in allowed and has_val and "evaluate_during_training" in allowed:
        filtered["evaluate_during_training"] = True
    return TrainingArguments(**filtered)


def _ensure_builder_funcs(cfg):
    """Return persona_to_text/build_prompt helpers, preferring the global ones."""
    persona_fn = globals().get("persona_to_text")
    prompt_fn = globals().get("build_prompt")

    if persona_fn is None:
        def persona_fn(persona):
            if persona is None:
                return ""
            if isinstance(persona, dict):
                return "; ".join(f"{k}: {v}" for k, v in persona.items())
            if isinstance(persona, list):
                return "; ".join(map(str, persona))
            return str(persona)

    if prompt_fn is None:
        def prompt_fn(persona_text, question):
            pt = (persona_text or "").strip()
            q = (question or "").strip()
            if pt and q:
                return f"""### Persona
{pt}

### Question
{q}

### Answer"""
            if q:
                return f"""### Question
{q}

### Answer"""
            return "### Answer"

    return persona_fn, prompt_fn

def _build_prompts_answers(data, cfg, persona_to_text=None, build_prompt=None):
    """
    Build parallel lists: prompts, answers.
    Expects cfg.text_fields with keys: 'persona','question','reference'.
    If 'reference' is missing or 'N/A', the answer is empty string (still valid for SFT).
    """
    if persona_to_text is None or build_prompt is None:
        persona_to_text, build_prompt = _ensure_builder_funcs(cfg)

    tf = cfg.text_fields
    p_key = tf.get("persona", "persona")
    q_key = tf.get("question", "question")
    r_key = tf.get("reference", "reference")

    prompts, answers = [], []
    for ex in (data or []):
        persona_obj = ex.get(p_key, None)
        question    = ex.get(q_key, "")
        reference   = ex.get(r_key, "")

        persona_txt = persona_to_text(persona_obj)
        prompt = build_prompt(persona_txt, question)
        ans = reference if (reference and reference != "N/A") else ""

        prompts.append(prompt)
        answers.append(ans)

    return prompts, answers

def _tokenize_pairs(tokenizer, prompts, answers, max_length: int = 1024):
    """
    Tokenize prompt–answer pairs for supervised fine-tuning (SFT).
    Masks out the prompt part in labels with -100.
    Returns dict of lists: input_ids, attention_mask, labels.
    """

    input_ids, attention_masks, labels = [], [], []

    for p, a in zip(prompts, answers):
        # tokenize separately so prompt tokens are masked
        tok_p = tokenizer(p, truncation=True, padding=False, add_special_tokens=True, max_length=max_length)
        tok_a = tokenizer(a, truncation=True, padding=False, add_special_tokens=False, max_length=max_length)

        ids = tok_p["input_ids"] + tok_a["input_ids"] + [tokenizer.eos_token_id]
        lbls = [-100] * len(tok_p["input_ids"]) + tok_a["input_ids"] + [tokenizer.eos_token_id]

        input_ids.append(ids)
        attention_masks.append([1] * len(ids))
        labels.append(lbls)

    # pad all sequences to same length
    max_len = max(len(x) for x in input_ids) if input_ids else 0
    for i in range(len(input_ids)):
        pad_len = max_len - len(input_ids[i])
        input_ids[i] += [tokenizer.pad_token_id] * pad_len
        attention_masks[i] += [0] * pad_len
        labels[i] += [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

def maybe_finetune_with_lora(model, tokenizer, train_data, val_data, cfg):
    import math, torch, inspect
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # ---- helpers (self-contained) ----
    def get_lora_target_modules(m):
        common = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        found = set()
        for n, _ in m.named_modules():
            for c in common:
                if n.endswith(c):
                    found.add(c)
        return sorted(found) if found else common

    def _ensure_builder_funcs(cfg_):
        def persona_to_text(persona):
            if persona is None: return ""
            if isinstance(persona, dict):
                return "\n".join(f"{k}: {v}" for k,v in persona.items())
            if isinstance(persona, list):
                return "\n".join(map(str, persona))
            return str(persona)
        def build_prompt(persona_text, question):
            pt = (persona_text or "").strip()
            q  = (question or "").strip()
            if pt and q:
                return f"### Persona\n{pt}\n\n### Question\n{q}\n\n### Answer"
            elif q:
                return f"### Question\n{q}\n\n### Answer"
            else:
                return "### Answer"
        return persona_to_text, build_prompt

    def _build_prompts_answers(data, cfg_, persona_to_text, build_prompt):
        tf = cfg_.text_fields
        p_key = tf.get("persona", "persona")
        q_key = tf.get("question", "question")
        r_key = tf.get("reference", "reference")
        prompts, answers = [], []
        for ex in (data or []):
            persona_obj = ex.get(p_key, None)
            question    = ex.get(q_key, "")
            reference   = ex.get(r_key, "")
            persona_txt = persona_to_text(persona_obj)
            prompt = build_prompt(persona_txt, question)
            ans = reference if (reference and reference != "N/A") else ""
            prompts.append(prompt)
            answers.append(ans)
        return prompts, answers

    def _tokenize_pairs(tokenizer, prompts, answers, max_length: int = 1024):
        input_ids, attention_masks, labels = [], [], []
        for p, a in zip(prompts, answers):
            tp = tokenizer(p, truncation=True, padding=False, add_special_tokens=True,  max_length=max_length)
            ta = tokenizer(a, truncation=True, padding=False, add_special_tokens=False, max_length=max_length)
            ids  = tp["input_ids"] + ta["input_ids"] + [tokenizer.eos_token_id]
            lbls = [-100] * len(tp["input_ids"]) + ta["input_ids"] + [tokenizer.eos_token_id]
            input_ids.append(ids)
            attention_masks.append([1]*len(ids))
            labels.append(lbls)
        if not input_ids:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        max_len = max(len(x) for x in input_ids)
        for i in range(len(input_ids)):
            pad = max_len - len(input_ids[i])
            input_ids[i]      += [tokenizer.pad_token_id]*pad
            attention_masks[i] += [0]*pad
            labels[i]         += [-100]*pad
        return {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels}

    class DictTensorDS(torch.utils.data.Dataset):
        def __init__(self, enc): self.enc = enc; self.n = len(enc.get("input_ids", []))
        def __len__(self): return self.n
        def __getitem__(self, i): return {k: torch.tensor(v[i]) for k,v in self.enc.items()}

    def build_training_args(cfg_, has_val: bool):
        # prefer per-epoch evaluation when validation set exists
        eval_strategy = "epoch" if has_val else "no"
        kw = dict(
            output_dir=cfg_.output_dir,
            per_device_train_batch_size=cfg_.per_device_train_batch_size,
            per_device_eval_batch_size=cfg_.per_device_eval_batch_size,
            gradient_accumulation_steps=cfg_.gradient_accumulation_steps,
            num_train_epochs=cfg_.num_epochs,
            learning_rate=cfg_.learning_rate,
            weight_decay=cfg_.weight_decay,
            logging_steps=cfg_.logging_steps,
            save_steps=cfg_.save_steps,
            evaluation_strategy=eval_strategy,             # shimmed below if unsupported
            eval_steps=cfg_.eval_steps if has_val else None,
            save_total_limit=getattr(cfg_, "save_total_limit", 2),
            warmup_ratio=cfg_.warmup_ratio,
            lr_scheduler_type="cosine",
            gradient_checkpointing=getattr(cfg_, "grad_ckpt", False),
            bf16=(getattr(cfg_, "torch_dtype", "auto") in ["bfloat16","auto"] and torch.cuda.is_available()),
            fp16=(getattr(cfg_, "torch_dtype", "auto") == "float16" and torch.cuda.is_available()),
            report_to="none",
            max_grad_norm=getattr(cfg_, "max_grad_norm", 1.0),
        )
        sig = inspect.signature(TrainingArguments.__init__)
        allowed = set(sig.parameters.keys())
        filtered = {k: v for k, v in kw.items() if (k in allowed and v is not None)}
        if "evaluation_strategy" not in allowed and has_val and "evaluate_during_training" in allowed:
            filtered["evaluate_during_training"] = True
        return TrainingArguments(**filtered)

    # ---- gates ----
    if not getattr(cfg, "do_lora_finetune", False):
        print("[LoRA] cfg.do_lora_finetune=False → skipping.")
        return model
    if not train_data or len(train_data) == 0:
        print("[LoRA] train_data is empty → skipping.")
        return model

    # ---- wrap with LoRA ----
    model = prepare_model_for_kbit_training(model)
    target_modules = (cfg.lora_target_modules or []) or get_lora_target_modules(model)
    lconf = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=target_modules, task_type=TaskType.CAUSAL_LM, bias="none"
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

    # ---- data → prompts/answers → encodings ----
    persona_to_text, build_prompt = _ensure_builder_funcs(cfg)

    tr_prompts, tr_answers = _build_prompts_answers(train_data, cfg, persona_to_text, build_prompt)
    tr_enc = _tokenize_pairs(tokenizer, tr_prompts, tr_answers, max_length=getattr(cfg, "max_length", 1024))
    if len(tr_enc["input_ids"]) == 0:
        print("[LoRA] After tokenization, train set is empty → skipping.")
        return model
    train_ds = DictTensorDS(tr_enc)

    if val_data and len(val_data) > 0:
        va_prompts, va_answers = _build_prompts_answers(val_data, cfg, persona_to_text, build_prompt)
        va_enc = _tokenize_pairs(tokenizer, va_prompts, va_answers, max_length=getattr(cfg, "max_length", 1024))
        val_ds = DictTensorDS(va_enc) if len(va_enc.get("input_ids", [])) > 0 else None
    else:
        val_ds = None

    # ---- trainer ----
    args = build_training_args(cfg, has_val=(val_ds is not None))

    def pad_collate(batch):
        pad_id = tokenizer.pad_token_id
        def pad_stack(key, pad_value):
            return torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch], batch_first=True, padding_value=pad_value
            )
        return {
            "input_ids": pad_stack("input_ids", pad_id),
            "attention_mask": pad_stack("attention_mask", 0),
            "labels": pad_stack("labels", -100),
        }

    steps_per_epoch = math.ceil(
        len(train_ds) / max(1, cfg.per_device_train_batch_size) / max(1, cfg.gradient_accumulation_steps)
    )
    print(f"[LoRA] train_samples={len(train_ds)} "
          f"val_samples={0 if val_ds is None else len(val_ds)} "
          f"steps_per_epoch≈{steps_per_epoch}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=pad_collate,
    )

    trainer.train()
    return model

# Use only labeled examples (non-empty 'reference') for supervised fine-tuning
model = maybe_finetune_with_lora(model, tokenizer, labeled_train, labeled_val, cfg)

# --- Save LoRA adapter + tokenizer ---
from pathlib import Path


# Path where LoRA adapter will be saved
out_dir = Path(cfg.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# Save the adapter weights (LoRA) and tokenizer
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
print(f"[LoRA] Adapter weights and tokenizer saved to {out_dir.resolve()}")



import os
import time
import torch
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Answer generation function
# ----------------------------
def generate_answer(prompt: str, model, tokenizer, cfg):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Extract cleaner part of output
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1].strip()
    return text


# ----------------------------
# Main generation loop (with tqdm)
# ----------------------------
def run_generation(split_data, model, tokenizer, cfg):
    rows = []
    start_time = time.time()

    for ex in tqdm(split_data, desc="Generating answers", unit="sample"):
        persona_text = persona_to_text(ex.get(cfg.text_fields["persona"]))
        q = ex.get(cfg.text_fields["question"])
        prompt = build_prompt(persona_text, q)

        try:
            ans = generate_answer(prompt, model, tokenizer, cfg)
        except Exception as e:
            ans = f"[GENERATION_ERROR] {e}"

        rows.append({
            "id": ex.get(cfg.text_fields["id"]),
            "domain": ex.get(cfg.text_fields["domain"]),
            "persona": persona_text,
            "question": q,
            "reference": ex.get(cfg.text_fields.get("reference", ""), ""),
            "answer": ans
        })

    total_time = time.time() - start_time
    avg_time = total_time / max(1, len(split_data))
    print(f"\nCompleted {len(split_data)} samples in {total_time/60:.2f} minutes "
          f"({avg_time:.2f} sec/sample)")
    return pd.DataFrame(rows)








