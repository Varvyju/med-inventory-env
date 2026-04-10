---
title: MedInventoryEnv
emoji: 💊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# MedInventoryEnv

> An OpenEnv-compliant environment for training and evaluating AI agents on real-world pharmaceutical inventory management — a daily operational challenge faced by 1.2 million Indian pharmacies.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/openenv)
[![Phase 2](https://img.shields.io/badge/Phase%202-Passed-brightgreen)](https://huggingface.co/spaces/Varun1622/med-inventory-env)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem This Solves

Every day, pharmacy managers across India manually decide:

- Which medications are running dangerously low and need immediate reordering?
- How much should be ordered — enough to cover demand without expiry waste?
- Which supplier offers the best price, and can we stay within budget while covering the most critical items?

These decisions are made with spreadsheets, paper ledgers, or gut instinct. A wrong call means either a stockout (a patient cannot get insulin or antibiotics) or overstock (Rs 50,000 worth of short-expiry medication goes to waste).

**MedInventoryEnv turns this into a rigorous, reproducible benchmark for AI agents.**

---

## Environment Overview

An AI agent is placed inside a simulated Indian medical store. It receives real inventory data — stock levels, daily demand rates, expiry windows, supplier catalogues, and procurement budgets — and must make procurement decisions across three tasks of increasing difficulty.

The environment is fully stateful, seeded for reproducibility, and provides dense reward signals at every step so agents can learn from partial progress rather than only binary success/failure.
Agent (inference.py)
|
| POST /reset  ->  start episode, receive inventory
| POST /step   ->  send decision, receive reward + feedback
| GET  /state  ->  inspect full environment state
|
FastAPI Server (port 7860)
|
env.py       — episode state, step logic, reward shaping
tasks.py     — 3 graders: F1 / proximity / coverage+efficiency
models.py    — typed Pydantic: Observation, Action, StepResult

---

## Tasks

### Task 1 — Reorder Threshold Identification (Easy)

**Real-world analog:** The morning stock check. Which medications have fallen below their safety buffer and need to be ordered today?

**Objective:** Identify all medications where stock_level < reorder_point.

**Action format:**
```json
{"items_to_reorder": ["Paracetamol 500mg", "Insulin Regular"]}
```

**Grader:** F1 score with a small recall bonus. Missing a critical item is worse than a false positive.
**Max steps:** 5 | **Success threshold:** 0.80

---

### Task 2 — Reorder Quantity Optimisation (Medium)

**Real-world analog:** The purchase order calculation. How much of each medication should be ordered — enough to last until the next delivery cycle, but not so much that it expires before being sold?

**Objective:** For each medication below threshold, compute the optimal order quantity.

**Formula:** optimal_qty = max(0, min(30, expiry_days) x daily_demand - stock_level)

**Action format:**
```json
{"order_quantities": {"Paracetamol 500mg": 351, "Insulin Regular": 98}}
```

**Grader:** Mean per-item proximity score — peaks at exact optimal, degrades smoothly with distance.
**Max steps:** 8 | **Success threshold:** 0.75

---

### Task 3 — Multi-Supplier Budget Planning (Hard)

**Real-world analog:** The monthly procurement meeting. Three suppliers offer different prices for overlapping catalogues. The budget covers only 70% of full needs — the agent must prioritise.

**Objective:** Select supplier and quantity for each needed medication, maximising coverage within a constrained budget.

**Action format:**
```json
{
  "procurement_plan": [
    {"item": "Paracetamol 500mg", "supplier": "PharmaDirect", "quantity": 300},
    {"item": "Insulin Regular", "supplier": "MedSupply India", "quantity": 80}
  ]
}
```

**Grader:** 0.75 x coverage_score + 0.25 x budget_efficiency. Penalty for exceeding budget.
**Max steps:** 10 | **Success threshold:** 0.70

---

## Reward Function

Rewards are dense — the agent receives a signal on every step, not just at episode end.
raw_score    = grader output, strictly in (0.001, 0.999)
step_penalty = (step_number - 1) x 0.01
final_reward = clamp(raw_score - step_penalty, 0.001, 0.999)

The step efficiency penalty is small (1% per extra step) so it guides agents toward concise solutions without dominating the learning signal.

---

## Baseline Scores

Measured with gpt-4o-mini, seed=42, temperature=0.1:

| Task | Score | Steps |
|------|-------|-------|
| Task 1 — Reorder Identification | 0.77 | 2 |
| Task 2 — Quantity Optimisation | 0.63 | 3 |
| Task 3 — Budget Planning | 0.56 | 4 |
| Overall Average | 0.65 | — |

---

## Setup and Usage

### Docker (recommended)

```bash
git clone https://github.com/Varvyju/med-inventory-env
cd med-inventory-env
docker build -t med-inventory-env .
docker run -p 7860:7860 med-inventory-env

pip install openai httpx
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Hugging Face Space (live now)

```bash
pip install openai httpx
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=sk-...
export ENV_BASE_URL=https://Varun1622-med-inventory-env.hf.space
python inference.py
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| API_BASE_URL | https://api.openai.com/v1 | LLM API endpoint |
| MODEL_NAME | gpt-4o-mini | Model identifier |
| HF_TOKEN | — | API key, no default |
| ENV_BASE_URL | http://localhost:7860 | Environment server URL |
| LOCAL_IMAGE_NAME | — | Docker image for local runs |

---

## Project Structure
med-inventory-env/
├── Dockerfile       — container build for HF Spaces, port 7860
├── openenv.yaml     — OpenEnv spec metadata and task registry
├── inference.py     — baseline inference script (root level)
├── pyproject.toml   — project metadata and dependencies
├── README.md
└── server/
├── main.py      — FastAPI: /reset /step /state /health /tasks
├── env.py       — episode management, step logic, reward shaping
├── tasks.py     — 3 task definitions + deterministic graders
├── models.py    — Pydantic typed models
├── app.py       — server entry point
└── requirements.txt

---

## Why This Environment Matters

India has approximately 1.2 million retail pharmacies. The vast majority operate without software for inventory decisions. The consequences are not abstract:

- Insulin stockouts affect 77 million diabetic patients in India
- Antibiotic stockouts during infections directly increase mortality risk
- Expired medication waste costs the Indian pharmacy sector an estimated Rs 3,000 crore annually

MedInventoryEnv provides a reproducible benchmark that measures whether an AI agent can actually help with these decisions. The tasks reflect real constraints — demand uncertainty, expiry windows, supplier variability, budget limits — that make this hard for humans and harder still for agents.

This is a domain where better AI agents have direct, measurable real-world impact.

---

## License

MIT