# PolyPersona: Persona-Conditioned LLM Agents for Synthetic Survey Responses


---

<img width="1952" height="637" alt="image" src="https://github.com/user-attachments/assets/828c6206-8155-4a49-8130-b7e0dda8fab4" />


This repository contains a complete, code to implements the core pipeline described in the *PersonaVerse* paper: a generative-agent framework for **synthetic survey response generation** using persona-grounded small language models (SLMs).


## 📘 Overview

The file (`poly.py`) reproduces the PersonaVerse workflow:

1. **Load Persona-based Survey Data**  
   Structured dataset of personas, questions, and (optionally) human references.

2. **Prompt-based or Fine-tuned Generation**  
   Uses compact LLMs (TinyLlama, Phi-2, Qwen2, Mistral) to simulate realistic survey responses.

3. **Persona Consistency Enforcement**  
   Each virtual respondent maintains stable demographic and psychographic traits across questions.

4. **Evaluation Stack**  
   Multi-metric scoring for BLEU, ROUGE, BERTScore, Distinct-n diversity, sentiment alignment, and persona-answer embedding similarity.

5. **Reporting**  
   Domain-wise summaries, overall metrics CSV, and visual diagnostics (answer-length distribution, consistency).

-----

## 🧩 Installation

Run these once inside your Jupyter environment:

```bash
pip install transformers datasets accelerate peft bitsandbytes sentencepiece
pip install sacrebleu rouge-score bert-score sentence-transformers textstat nltk matplotlib pandas scikit-learn
python -m nltk.downloader punkt
```

Ensure you have access to a GPU with **≥8 GB VRAM** (TinyLlama works on 6 GB in 4-bit mode).

---

## 🗂️ Directory Layout

```
personaverse/
│
├── poly.py   # Main notebook
├── README.md                            # This file
├── data/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── outputs/
    ├── generated_test.csv
    ├── domain_metrics.csv
    └── length_by_domain.png
```

---

## 📄 Dataset Format

Each record must follow this structure (JSON list or JSON-Lines):

```json
{
  "id": "p001_q45",
  "domain": "healthcare",
  "persona": {
    "age": 34,
    "gender": "female",
    "profession": "teacher",
    "values": ["empathy", "community"]
  },
  "question": "How satisfied are you with your healthcare provider?",
  "reference": "I’m mostly satisfied but wish appointments were faster.",
  "messages": []
}
```

## 🧷 Citation

If you use this notebook or its methodology, please cite:
xxxxx
---

## ⚖️ License

MIT License © 2025 — you are free to use, modify, and redistribute with attribution.

---

