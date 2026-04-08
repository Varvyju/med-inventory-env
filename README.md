---
title: MedInventoryEnv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# ðŸ¥ MedInventoryEnv

> **An OpenEnv environment for medical store inventory management â€” real-world AI agent training and evaluation.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/ðŸ¤—-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://python.org)

---

## ðŸŽ¯ What Is This?

MedInventoryEnv simulates the daily procurement decisions made by Indian pharmacy managers. An AI agent is placed in a realistic inventory scenario and must:

1. **Identify** which medications are running low (below reorder threshold)
2. **Calculate** the optimal quantity to reorder for each item
3. **Plan** multi-supplier procurement within a constrained budget

This is a genuine operational challenge â€” Indian medical stores (â‰ˆ1.2 million nationwide) manage 200â€“500 SKUs with varying demand, expiry constraints, and supplier relationships. Mistakes cause stockouts (patient harm) or overstock (expiry waste).

---

## ðŸ“ Environment Design

### Architecture

```
inference.py  â†â”€â”€â†’  FastAPI Server (port 7860)
                         â”‚
                    â”Œâ”€â”€â”€â”€â”´â”€â”€â”€â”€â”
                    â”‚  env.py â”‚   episode state, step logic
                    â”‚tasks.py â”‚   3 graders (easyâ†’hard)
                    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode. Body: `{"task_id": "task_1", "seed": 42}` |
| `POST` | `/step`  | Send action. Body: `{"message": "<json_string>"}` |
| `GET`  | `/state` | Current full state (debugging) |
| `GET`  | `/tasks` | List all task configs |
| `GET`  | `/health`| Health check |

### Action Space

A plain JSON string encoding the agent's decision. Format differs per task (see below).

### Observation Space

```json
{
  "echoed_message":   "<last action sent>",
  "task_id":          "task_1 | task_2 | task_3",
  "task_description": "<full task prompt with expected JSON format>",
  "inventory": [
    {
      "name":          "Paracetamol 500mg",
      "category":      "Analgesic",
      "stock_level":   45,
      "reorder_point": 150,
      "daily_demand":  13.2,
      "expiry_days":   180,    // task_2 and task_3 only
      "unit_cost":     2.47    // task_2 and task_3 only
    }
  ],
  "suppliers":   [...],   // task_3 only
  "budget":      4200.0,  // task_3 only (INR)
  "step_number": 1,
  "max_steps":   5,
  "feedback":    "<grader feedback from previous action>"
}
```

### Reward Function

- **Dense** â€” reward on every step, not just at episode end
- **Raw score** â€” computed by the task-specific grader (0.0â€“1.0)
- **Step efficiency penalty** â€” `(step - 1) * 0.02` subtracted to reward concise solutions
- **Final reward** = `clamp(raw_score - step_penalty, 0.0, 1.0)`

---

## ðŸ“‹ Tasks

### Task 1 â€” Reorder Threshold Identification *(Easy)*

**Objective:** Identify all medications where `stock_level < reorder_point`.

**Action format:**
```json
{"items_to_reorder": ["Paracetamol 500mg", "Metformin 500mg"]}
```

**Grader:** F1 score between predicted and actual below-threshold items.  
**Max steps:** 5 | **Success threshold:** 0.80

---

### Task 2 â€” Reorder Quantity Optimisation *(Medium)*

**Objective:** For each medication below threshold, compute optimal order quantity.

**Formula:** `optimal = max(0, min(30, expiry_days) Ã— daily_demand âˆ’ stock_level)`

**Action format:**
```json
{"order_quantities": {"Paracetamol 500mg": 351, "Metformin 500mg": 240}}
```

**Grader:** Mean per-item proximity score â€” peaks at exact optimal, degrades smoothly.  
**Max steps:** 8 | **Success threshold:** 0.75

---

### Task 3 â€” Multi-Supplier Budget Planning *(Hard)*

**Objective:** Select supplier + quantity for each needed item, staying within budget. Budget intentionally covers only ~70% of full needs â€” the agent must prioritise.

**Action format:**
```json
{
  "procurement_plan": [
    {"item": "Paracetamol 500mg", "supplier": "PharmaDirect",    "quantity": 300},
    {"item": "Metformin 500mg",   "supplier": "MedSupply India", "quantity": 200}
  ]
}
```

**Grader:** `0.75 Ã— coverage_score + 0.25 Ã— budget_efficiency`. Heavy penalty if over budget.  
**Max steps:** 10 | **Success threshold:** 0.70

---

## ðŸ“Š Baseline Scores

Run with `gpt-4o-mini` (seed=42, temperature=0.1):

| Task | Score | Steps |
|------|-------|-------|
| Task 1 â€” Reorder Identification | 0.78 | 2 |
| Task 2 â€” Quantity Optimisation  | 0.61 | 3 |
| Task 3 â€” Budget Planning        | 0.54 | 4 |
| **Overall Average**             | **0.64** | â€” |

---

## ðŸš€ Setup & Usage

### Option A â€” Docker (recommended)

```bash
git clone https://github.com/<your-username>/med-inventory-env
cd med-inventory-env

# Build and run the environment server
docker build -t med-inventory-env .
docker run -p 7860:7860 med-inventory-env

# In another terminal â€” run the inference script
pip install openai httpx
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Option B â€” Local (development)

```bash
# Start server
cd server
pip install -r requirements.txt
python main.py

# Run inference
cd ..
pip install openai httpx
export API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Option C â€” Hugging Face Space

The environment is live at: `https://huggingface.co/spaces/<username>/med-inventory-env`

```bash
export ENV_BASE_URL=https://<username>-med-inventory-env.hf.space
python inference.py
```

---

## ðŸ”§ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | âœ… | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME`   | âœ… | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN`     | âœ… | Hugging Face / OpenAI API key |
| `ENV_BASE_URL` | âŒ | Environment server URL (default: `http://localhost:7860`) |

---

## ðŸ“ Project Structure

```
med-inventory-env/
â”œâ”€â”€ Dockerfile            # Container build for HF Spaces
â”œâ”€â”€ openenv.yaml          # OpenEnv spec metadata
â”œâ”€â”€ inference.py          # Baseline inference script (root)
â”œâ”€â”€ requirements.txt      # inference.py dependencies
â”œâ”€â”€ README.md
â””â”€â”€ server/
    â”œâ”€â”€ main.py           # FastAPI server
    â”œâ”€â”€ env.py            # Episode management, step logic
    â”œâ”€â”€ tasks.py          # Task definitions + graders
    â”œâ”€â”€ models.py         # Pydantic typed models
    â””â”€â”€ requirements.txt  # Server dependencies
```

---

## ðŸ’¡ Why This Environment Matters

- **1.2 million** pharmacies in India lack AI-assisted inventory tools
- Stockouts of critical medications (insulin, antibiotics) cause direct patient harm
- Overstocking of short-expiry medications causes lakhs in waste annually
- This environment provides a rigorous, reproducible benchmark for agents tackling real supply-chain decisions in a healthcare context

---

## ðŸ“œ License

MIT


