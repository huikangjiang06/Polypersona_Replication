### Usage
- **Build Data**: build_polypersona.py (from the original repo)
- **Generate Synthetic Response**: response_generation.py
- **Training Model**: poly.py (from the original repo)
- **Evalute Model**: evaluation.py
- **Compare Referencea and Prediction**: compare.py

### Prompt Formats Used:
1. **Persona_to_Text**:
- Option 1
 ```python
def persona_to_text(persona) -> str:
    if isinstance(persona, str):
        return persona
    if isinstance(persona, dict):
        parts = []
        for k,v in persona.items():
            if isinstance(v, list):
                v = ", ".join(map(str,v))
            parts.append(f"{k}: {v}")
        return "; ".join(parts)  # SEMICOLON-SEPARATED
    return str(persona)
```
- Option 2
```python
def persona_fn(persona):
    if persona is None:
        return ""
    if isinstance(persona, dict):
        return "; ".join(f"{k}: {v}" for k, v in persona.items())  # SEMICOLON-SEPARATED
    if isinstance(persona, list):
        return "; ".join(map(str, persona))
    return str(persona)
```
- Option 3
```python
def persona_to_text(persona):
    if persona is None: return ""
    if isinstance(persona, dict):
        return "\n".join(f"{k}: {v}" for k,v in persona.items())  # NEWLINE-SEPARATED!
    if isinstance(persona, list):
        return "\n".join(map(str, persona))
    return str(persona)
```
2. **Input Prompts**:
- Option 1
```python
def build_prompt(persona_text, question, qtype=None):
    SYSTEM_PROMPT = (
        "You are PolyPersona, a helpful and realistic survey respondent. "
        "Answer faithfully based on the given persona."
    )
    
    # Question-type specific hints
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
```
- Option 2
```python
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
```
- Option 3
```python
def build_prompt(persona_text, question):
    pt = (persona_text or "").strip()
    q  = (question or "").strip()
    if pt and q:
        return f"### Persona\n{pt}\n\n### Question\n{q}\n\n### Answer"
    elif q:
        return f"### Question\n{q}\n\n### Answer"
    else:
        return "### Answer"
```

### Experiments
1. **Experiment 1**
    - Reference Model: Qwen/Qwen2.5-7B-Instruct
    - Generation Hyperparameters:
        - Top-p: 0.9
        - Temperature: 0.7
        - Persona Prompt: Option 1
        - Input Prompt: Option 1
    - Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - Generation Hyperparameters:
        - Top-p: 0.9
        - Temperature: 0.7
        - Epochs: 3
        - Persona Prompt: Option 1
        - Input Prompt: Option 1
    - Outputs:
        - ./outputs/experiment_1_synthetic_data
        - ./outputs/experiment_1_personaverse
        - ./outputs/experiment_1_results