### Usage
- **Build Data**: build_polypersona.py (from the original repo)
- **Generate Synthetic Response**: response_generation.py
- **Train Model**: poly.py (from the original repo)
- **Train Model with consistent prompt**: poly_consistent_prompt.py (poly.py uses different prompt for training and inference)
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
    - General Try Out
    - Reference Model: Qwen/Qwen2.5-7B-Instruct
    - Inference Hyperparameters:
        - Top-p: 0.9
        - Temperature: 0.7
        - Persona Prompt: Option 1
        - Input Prompt: Option 1
    - Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - Inference Hyperparameters:
        - Top-p: 0.9
        - Temperature: 0.7
        - Persona Prompt: Option 1
        - Input Prompt: Option 1
    - Outputs:
        - ./outputs/experiment_1_synthetic_data
        - ./outputs/experiment_1_personaverse
        - ./outputs/experiment_1_results

2. **Experiment 2**
    - No Training Baseline (did not train the model)
    - Reference Model: Qwen/Qwen2.5-7B-Instruct
    - Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (untrained)
    - Outputs:
        - ./outputs/experiment_2_results

3. **Experiment 3**:
    - Variations in Inference Prompting
    - Reference Model: Qwen/Qwen2.5-7B-Instruct
    - Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - Inference Hyperparameters:
        - Persona Prompt: Option 3
        - Input Prompt: Option 3
    - Outputs:
        - ./outputs/experiment_3_results

4. **Experiment 4**:
    - Test for generalization across question domain
    - Reference Model: Qwen/Qwen2.5-7B-Instruct
    - Student Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
        - Trained on only "demographics" "healthcare" "education" and "work_experience"
    - Outputs: 
        - ./outputs/experiment_4_synthetic_data (redoed train-val-test split)
        - ./outputs/experiment_4_personaverse
        - ./outputs/experiment_4_results
    
5. **Experiment 5**:
    - Tried deocding with temperture = 0.01
    - Outputs:
        - ./outputs/experiment_5_results