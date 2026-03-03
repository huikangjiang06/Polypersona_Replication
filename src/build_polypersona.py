#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build PersonaVerse from PersonaHub → train/val/test newline-JSON files.

Usage:
  python build_personaverse.py \
    --questionbank questionbank.json \
    --out-dir personaverse_out
"""

import json
import random
import hashlib
import argparse
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset  # pip install datasets

# ---------------------------- CLI ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questionbank", type=str, required=True,
                   help="Path to questionbank.json (10 domains × 4 qtypes)")
    p.add_argument("--out-dir", type=str, default="personaverse_out",
                   help="Output directory for train/val/test")
    p.add_argument("--total-examples", type=int, default=3568)
    p.add_argument("--total-personas", type=int, default=433)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hf-dataset", type=str, default="proj-persona/PersonaHub")
    p.add_argument("--hf-config", type=str, default="instruction")
    return p.parse_args()

# ------------------------ CONFIG (from paper) ------------------------

DOMAIN_TARGETS = {
    "demographics": 520,         # 14.6%
    "healthcare": 416,           # 11.7%
    "education": 416,            # 11.7%
    "work_experience": 400,      # 11.2%
    "technology": 384,           # 10.8%
    "consumer_preferences": 368, # 10.3%
    "finance": 368,              # 10.3%
    "social_issues": 264,        # 7.4%
    "environment": 216,          # 6.1%
    "lifestyle": 216             # 6.1%
}

# Question type ratios (approx)
QTYPE_RATIOS = {
    "open": 0.427,
    "likert": 0.317,
    "yesno": 0.183,
    "agreement": 0.073
}

# 71.4% appear in one domain, 28.6% across multiple domains
SINGLE_DOMAIN_SHARE = 0.714

# ----------------------- UTILITIES -----------------------

def deterministic_id(obj, prefix="pp", n=6):
    h = hashlib.md5(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return f"{prefix}{h[:n]}"

def normalize_persona(ph_row, rng):
    """
    Map a PersonaHub row into a compact 'persona card' for survey synthesis.
    PersonaHub schemas vary; we allow fallbacks.
    """
    persona_blob = ph_row.get("persona", ph_row.get("attributes", {}))

    def pull(*keys, default=None):
        for k in keys:
            if k in persona_blob:
                return persona_blob[k]
        return default

    card = {
        "age": pull("age", default=rng.choice([22, 27, 34, 41, 55])),
        "gender": pull("gender", default=rng.choice(["female", "male", "nonbinary"])),
        "occupation": pull("occupation", "profession") or rng.choice(
            ["software engineer","nurse","teacher","student","retail manager","analyst","technician"]
        ),
        "education": pull("education", default=rng.choice(["high-school","bachelor's","master's"])),
        "region": pull("region", "location", default=rng.choice(["US-West","US-East","EU","APAC"])),
        "values": persona_blob.get("values", []) or rng.sample(
            ["efficiency","privacy","community","sustainability","innovation","tradition"], k=2),
        "traits": persona_blob.get("traits", []) or rng.sample(
            ["analytical","empathetic","detail-oriented","risk-averse","early-adopter","pragmatic"], k=2),
        "interests": persona_blob.get("interests", []) or rng.sample(
            ["fitness","open-source","gardening","finance","gaming","volunteering","travel"], k=2),
        "income_bracket": persona_blob.get("income_bracket", rng.choice(["<40k","40k-60k","60k-100k",">100k"]))
    }
    pid = deterministic_id(ph_row, prefix="pp", n=6)
    return pid, card

def build_messages(persona_card, domain, question, reference):
    sys = "You are PolyPersona, a survey respondent. Answer faithfully as the given persona."
    user = (
        f"Persona:\n"
        f"  Age: {persona_card['age']}\n"
        f"  Gender: {persona_card['gender']}\n"
        f"  Occupation: {persona_card['occupation']}\n"
        f"  Education: {persona_card['education']}\n"
        f"  Region: {persona_card['region']}\n"
        f"  Values: {', '.join(persona_card['values'])}\n"
        f"  Traits: {', '.join(persona_card['traits'])}\n"
        f"  Interests: {', '.join(persona_card['interests'])}\n\n"
        f"Domain: {domain}\n"
        f"Question: {question}\n"
        f"Answer succinctly but realistically."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
        {"role": "assistant", "content": reference}
    ]

def default_answer(rng, persona_card, qtype):
    # Placeholder — replace with real PolyPersona inference
    if qtype == "yesno":
        base = rng.choice(["Yes.", "No."])
    elif qtype == "likert":
        base = rng.choice(["Strongly agree.", "Agree.", "Neutral.", "Disagree.", "Strongly disagree."])
    else:
        base = rng.choice(["I think", "In my experience", "From my perspective"])
    tail = f" As a {persona_card['occupation']}, I value {rng.choice(persona_card['values'])}."
    return f"{base} {tail}".strip()

def sample_qtype(rng, ratios):
    r = rng.random()
    cum = 0.0
    for qt, p in ratios.items():
        cum += p
        if r <= cum:
            return qt
    # numeric drift guard
    return list(ratios.keys())[-1]

def choose_question(rng, bank, domain, qtype):
    dom = bank.get(domain, {})
    pool = dom.get(qtype, [])
    if not pool:
        # fall back to any available in domain
        pool = []
        for arr in dom.values():
            pool.extend(arr)
    if not pool:
        # last resort: generic placeholder
        return f"[{domain}] Please answer this {qtype} question about your experiences."
    return rng.choice(pool)

def verify_questionbank(bank):
    need_domains = set(DOMAIN_TARGETS.keys())
    have_domains = set(bank.keys())
    missing = need_domains - have_domains
    if missing:
        raise ValueError(f"questionbank.json missing domains: {sorted(missing)}")

    for d in need_domains:
        for qt in ["open","likert","yesno","agreement"]:
            if qt not in bank[d] or not isinstance(bank[d][qt], list) or len(bank[d][qt]) == 0:
                # warn but do not fail; code will fallback gracefully
                print(f"[WARN] Domain '{d}' missing/nonlist/empty type '{qt}'. Will fallback at runtime.")

def verify_targets(total_examples):
    if sum(DOMAIN_TARGETS.values()) != total_examples:
        raise ValueError(f"DOMAIN_TARGETS sum {sum(DOMAIN_TARGETS.values())} != TOTAL_EXAMPLES {total_examples}")
    s = sum(QTYPE_RATIOS.values())
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"QTYPE_RATIOS must sum to 1.0 (got {s:.4f})")

def mint_record_id(domain, idx, pid):
    return f"{domain}-{idx:03d}-{pid}"

# --------------------- MAIN PIPELINE ---------------------

def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Load question bank
    with open(args.questionbank, "r", encoding="utf-8") as f:
        QUESTION_BANK = json.load(f)
    verify_questionbank(QUESTION_BANK)
    verify_targets(args.total_examples)

    # HF dataset
    print("[INFO] Loading PersonaHub from Hugging Face…")
    ds = load_dataset(args.hf_dataset, args.hf_config)

    # Build persona pool
    persona_pool = {}
    for split in ds:
        for row in ds[split]:
            pid, card = normalize_persona(row, rng)
            if pid not in persona_pool:
                persona_pool[pid] = card

    print(f"[INFO] Unique normalized personas available: {len(persona_pool)}")
    if len(persona_pool) < args.total_personas:
        raise ValueError(f"Not enough unique personas: have {len(persona_pool)}, need {args.total_personas}")

    # Sample desired number of personas
    persona_ids = list(persona_pool.keys())
    rng.shuffle(persona_ids)
    persona_ids = persona_ids[:args.total_personas]

    # Split personas into single-domain and multi-domain cohorts
    n_single = int(round(args.total_personas * SINGLE_DOMAIN_SHARE))
    single_domain_pids = set(persona_ids[:n_single])
    multi_domain_pids  = set(persona_ids[n_single:])

    print(f"[INFO] Personas: {len(single_domain_pids)} single-domain, {len(multi_domain_pids)} multi-domain")

    # Domain queues (targets to fill)
    domain_queues = dict(DOMAIN_TARGETS)
    idx_counters = defaultdict(int)
    examples = []

    # First pass: allocate single-domain personas
    for pid in single_domain_pids:
        remaining = [d for d, n in domain_queues.items() if n > 0]
        if not remaining:
            break
        dsel = rng.choice(remaining)
        # each persona contributes 3–10 examples in its domain
        k = min(rng.randint(3, 10), domain_queues[dsel])
        for _ in range(k):
            qtype = sample_qtype(rng, QTYPE_RATIOS)
            question = choose_question(rng, QUESTION_BANK, dsel, qtype)
            persona_card = persona_pool[pid]
            answer = default_answer(rng, persona_card, qtype)  # hook PolyPersona here
            idx_counters[dsel] += 1
            rec_id = mint_record_id(dsel, idx_counters[dsel], pid)
            ex = {
                "id": rec_id,
                "domain": dsel,
                "persona": persona_card,
                "question": question,
                "question_type": qtype,
                "reference": answer,
                "messages": build_messages(persona_card, dsel, question, answer),
                "meta": {
                    "persona_id": pid,
                    "question_id": f"{dsel}_q_{idx_counters[dsel]:03d}",
                    "domain_id": dsel
                }
            }
            examples.append(ex)
            domain_queues[dsel] -= 1

    # Second pass: multi-domain personas to fill remaining quotas
    for pid in multi_domain_pids:
        remaining = [d for d, n in domain_queues.items() if n > 0]
        if not remaining:
            break
        n_dom = rng.randint(1, min(3, len(remaining)))
        chosen_domains = rng.sample(remaining, n_dom)
        for dsel in chosen_domains:
            if domain_queues[dsel] <= 0:
                continue
            k = min(rng.randint(2, 6), domain_queues[dsel])
            for _ in range(k):
                qtype = sample_qtype(rng, QTYPE_RATIOS)
                question = choose_question(rng, QUESTION_BANK, dsel, qtype)
                persona_card = persona_pool[pid]
                answer = default_answer(rng, persona_card, qtype)  # hook PolyPersona here
                idx_counters[dsel] += 1
                rec_id = mint_record_id(dsel, idx_counters[dsel], pid)
                ex = {
                    "id": rec_id,
                    "domain": dsel,
                    "persona": persona_card,
                    "question": question,
                    "question_type": qtype,
                    "reference": answer,
                    "messages": build_messages(persona_card, dsel, question, answer),
                    "meta": {
                        "persona_id": pid,
                        "question_id": f"{dsel}_q_{idx_counters[dsel]:03d}",
                        "domain_id": dsel
                    }
                }
                examples.append(ex)
                domain_queues[dsel] -= 1

    # Balance to exact TOTAL_EXAMPLES
    if len(examples) > args.total_examples:
        rng.shuffle(examples)
        examples = examples[:args.total_examples]
    elif len(examples) < args.total_examples:
        # Duplicate a few (rare if question bank is well populated)
        rng.shuffle(examples)
        while len(examples) < args.total_examples and examples:
            e = dict(rng.choice(examples))
            e["id"] = e["id"] + "-dup" + str(len(examples))
            examples.append(e)

    # Final sanity
    assert len(examples) == args.total_examples, f"Have {len(examples)} examples, expected {args.total_examples}"

    # Splits
    rng.shuffle(examples)
    n_train = int(0.8 * len(examples))
    n_val   = int(0.1 * len(examples))
    train = examples[:n_train]
    val   = examples[n_train:n_train+n_val]
    test  = examples[n_train+n_val:]

    # Write
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out_dir / "train.json", train)
    dump(out_dir / "val.json",   val)
    dump(out_dir / "test.json",  test)

    print(f"[DONE] Wrote {len(train)} train, {len(val)} val, {len(test)} test → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
