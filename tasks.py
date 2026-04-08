"""
tasks.py — 3 tasks with deterministic graders for MedInventoryEnv
Task 1 (Easy):   Identify items below reorder threshold   → F1 score
Task 2 (Medium): Compute optimal reorder quantities       → proximity score
Task 3 (Hard):   Multi-supplier budget planning           → coverage + efficiency score
"""
import json
import random
from typing import List, Dict, Any, Tuple, Optional

# ── Medication catalogue ──────────────────────────────────────────────────────

MEDICATIONS = [
    {"name": "Paracetamol 500mg",   "category": "Analgesic",         "base_demand": 15.0, "base_cost": 2.5},
    {"name": "Amoxicillin 250mg",   "category": "Antibiotic",        "base_demand": 8.0,  "base_cost": 12.0},
    {"name": "Metformin 500mg",     "category": "Antidiabetic",      "base_demand": 12.0, "base_cost": 5.0},
    {"name": "Amlodipine 5mg",      "category": "Antihypertensive",  "base_demand": 10.0, "base_cost": 8.0},
    {"name": "Omeprazole 20mg",     "category": "Antacid",           "base_demand": 9.0,  "base_cost": 6.5},
    {"name": "Cetirizine 10mg",     "category": "Antihistamine",     "base_demand": 7.0,  "base_cost": 4.0},
    {"name": "Azithromycin 500mg",  "category": "Antibiotic",        "base_demand": 5.0,  "base_cost": 18.0},
    {"name": "Atorvastatin 10mg",   "category": "Statin",            "base_demand": 11.0, "base_cost": 9.0},
    {"name": "Ibuprofen 400mg",     "category": "NSAID",             "base_demand": 13.0, "base_cost": 3.5},
    {"name": "Pantoprazole 40mg",   "category": "Antacid",           "base_demand": 8.0,  "base_cost": 7.0},
    {"name": "Vitamin D3 60K IU",   "category": "Supplement",        "base_demand": 6.0,  "base_cost": 15.0},
    {"name": "Insulin Regular",     "category": "Antidiabetic",      "base_demand": 4.0,  "base_cost": 45.0},
]

SUPPLIER_TEMPLATES = [
    {"name": "MedSupply India", "lead_time_days": 2, "reliability_score": 0.95},
    {"name": "PharmaDirect",    "lead_time_days": 1, "reliability_score": 0.88},
    {"name": "GenericPlus",     "lead_time_days": 4, "reliability_score": 0.92},
]

# ── Inventory & supplier generators ──────────────────────────────────────────

def generate_inventory(num_items: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    """Generate a reproducible random pharmacy inventory."""
    rng = random.Random(seed)
    items = rng.sample(MEDICATIONS, min(num_items, len(MEDICATIONS)))
    inventory = []
    for item in items:
        demand = round(item["base_demand"] * rng.uniform(0.7, 1.3), 1)
        # Mix of healthy, borderline, and critically-low stock
        stock_days = rng.choice([rng.uniform(1, 4), rng.uniform(5, 12), rng.uniform(15, 40)])
        stock = max(0, int(demand * stock_days))
        reorder_point = int(demand * rng.uniform(5, 10))  # 5–10 day safety buffer
        expiry_days = rng.randint(30, 365)
        inventory.append({
            "name": item["name"],
            "category": item["category"],
            "stock_level": stock,
            "reorder_point": reorder_point,
            "daily_demand": demand,
            "unit_cost": round(item["base_cost"] * rng.uniform(0.9, 1.1), 2),
            "expiry_days": expiry_days,
        })
    return inventory


def generate_suppliers(inventory: List[Dict], seed: int = 42) -> Tuple[List[Dict], float]:
    """Generate suppliers and a budget that forces prioritization."""
    rng = random.Random(seed + 100)
    suppliers = []
    for template in SUPPLIER_TEMPLATES:
        items_offered = {}
        for item in inventory:
            if rng.random() > 0.25:  # 75% chance supplier carries an item
                price_multiplier = rng.uniform(0.85, 1.20)
                items_offered[item["name"]] = round(item["unit_cost"] * price_multiplier, 2)
        suppliers.append({
            "name": template["name"],
            "lead_time_days": template["lead_time_days"],
            "reliability_score": template["reliability_score"],
            "items": items_offered,
        })

    # Budget = 65–80% of what full optimal coverage would cost (forces prioritization)
    items_needing = [i for i in inventory if i["stock_level"] < i["reorder_point"]]
    full_cost = sum(
        i["unit_cost"] * max(0, int(min(30, i["expiry_days"]) * i["daily_demand"]) - i["stock_level"])
        for i in items_needing
    )
    budget = round(full_cost * rng.uniform(0.65, 0.82), 2)
    return suppliers, max(budget, 500.0)

# ── Task graders ──────────────────────────────────────────────────────────────

def grade_task1(action_message: str, inventory: List[Dict]) -> Tuple[float, str]:
    """
    Task 1: Identify all medications below reorder threshold.
    Expected JSON: {"items_to_reorder": ["Med A", "Med B", ...]}
    Score: F1 between predicted and actual below-threshold items.
    """
    try:
        data = json.loads(action_message)
        predicted = set(data.get("items_to_reorder", []))
    except (json.JSONDecodeError, TypeError, AttributeError):
        return 0.0, (
            "Invalid JSON. Expected: {\"items_to_reorder\": [\"Paracetamol 500mg\", ...]}"
        )

    actual = {item["name"] for item in inventory if item["stock_level"] < item["reorder_point"]}

    if not actual:
        if not predicted:
            return 1.0, "Correct: no items need reordering."
        return 0.3, f"No items need reordering, but you flagged {len(predicted)} incorrectly."

    tp = len(predicted & actual)
    fp = len(predicted - actual)
    fn = len(actual - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    missed   = sorted(actual - predicted)
    false_pos = sorted(predicted - actual)
    feedback = (
        f"Correctly identified: {tp}/{len(actual)}. "
        f"Missed: {missed[:3]}{'...' if len(missed)>3 else ''}. "
        f"False positives: {false_pos[:3]}{'...' if len(false_pos)>3 else ''}. "
        f"F1={f1:.3f}"
    )
    return round(f1, 4), feedback


def grade_task2(action_message: str, inventory: List[Dict]) -> Tuple[float, str]:
    """
    Task 2: Calculate optimal reorder quantities.
    Expected JSON: {"order_quantities": {"Paracetamol 500mg": 450, ...}}
    Score: mean of per-item proximity scores (peaks at exact optimal quantity).
    """
    try:
        data = json.loads(action_message)
        agent_qty = {k: max(0, int(v)) for k, v in data.get("order_quantities", {}).items()}
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return 0.0, (
            "Invalid JSON. Expected: {\"order_quantities\": {\"Paracetamol 500mg\": 450, ...}}"
        )

    items_needing = [i for i in inventory if i["stock_level"] < i["reorder_point"]]
    if not items_needing:
        return 1.0, "No items need reordering."

    scores = []
    parts  = []
    for item in items_needing:
        days_cover = min(30, item["expiry_days"])
        optimal    = max(0, int(days_cover * item["daily_demand"]) - item["stock_level"])
        agent      = agent_qty.get(item["name"], 0)

        if optimal == 0:
            s = 1.0 if agent == 0 else 0.6
        else:
            ratio = agent / optimal
            # Smooth penalty: 0 error → 1.0, doubles the error → 0.2
            s = max(0.0, 1.0 - abs(ratio - 1.0) * 0.8)

        scores.append(s)
        parts.append(f"{item['name']}: you={agent}, optimal≈{optimal}, item_score={s:.2f}")

    avg = round(sum(scores) / len(scores), 4)
    shown = "; ".join(parts[:4]) + (f" [+{len(parts)-4} more]" if len(parts) > 4 else "")
    return avg, f"Avg score={avg:.3f}. Details: {shown}"


def grade_task3(
    action_message: str,
    inventory: List[Dict],
    suppliers: List[Dict],
    budget: float,
) -> Tuple[float, str]:
    """
    Task 3: Multi-supplier procurement within budget.
    Expected JSON:
      {"procurement_plan": [
          {"item": "Paracetamol 500mg", "supplier": "MedSupply India", "quantity": 300},
          ...
      ]}
    Score = 0.75*coverage + 0.25*cost_efficiency  (heavy penalty if over budget)
    """
    try:
        data = json.loads(action_message)
        plan = data.get("procurement_plan", [])
        if not isinstance(plan, list):
            raise ValueError("procurement_plan must be a list")
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return 0.0, (
            "Invalid JSON. Expected: {\"procurement_plan\": "
            "[{\"item\": \"...\", \"supplier\": \"...\", \"quantity\": N}, ...]}"
        )

    needed = {i["name"] for i in inventory if i["stock_level"] < i["reorder_point"]}
    if not needed:
        return 1.0, "No items need reordering."

    # Build price lookup: item → supplier → price
    price_lookup: Dict[str, Dict[str, float]] = {}
    for sup in suppliers:
        for item_name, price in sup.get("items", {}).items():
            price_lookup.setdefault(item_name, {})[sup["name"]] = price

    total_cost   = 0.0
    items_covered = set()

    for entry in plan:
        item_name     = entry.get("item", "")
        supplier_name = entry.get("supplier", "")
        qty           = max(0, int(entry.get("quantity", 0)))

        sup_prices = price_lookup.get(item_name, {})
        if supplier_name in sup_prices and qty > 0:
            total_cost += sup_prices[supplier_name] * qty
            if item_name in needed:
                items_covered.add(item_name)

    over_budget = total_cost > budget * 1.05  # 5% tolerance
    coverage    = len(items_covered) / len(needed) if needed else 0.0

    if over_budget:
        overage_pct = (total_cost - budget) / budget * 100
        penalty     = max(0.0, coverage * 0.35 - overage_pct * 0.005)
        return round(penalty, 4), (
            f"OVER BUDGET by {overage_pct:.1f}%! "
            f"Spent ₹{total_cost:.0f} vs budget ₹{budget:.0f}. "
            f"Covered {len(items_covered)}/{len(needed)} items. Score={penalty:.3f}"
        )

    used_ratio       = min(total_cost / budget, 1.0) if budget > 0 else 0.0
    efficiency_bonus = used_ratio * 0.25 if coverage >= 0.75 else 0.0
    score            = round(min(1.0, 0.75 * coverage + efficiency_bonus), 4)
    return score, (
        f"Covered {len(items_covered)}/{len(needed)} items. "
        f"Spent ₹{total_cost:.0f} / ₹{budget:.0f} ({used_ratio*100:.1f}% of budget). "
        f"Score={score:.3f}"
    )

# ── Task config registry ──────────────────────────────────────────────────────

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "id": "task_1",
        "name": "Reorder Threshold Identification",
        "difficulty": "easy",
        "max_steps": 5,
        "reward_threshold": 0.80,
        "num_items": 10,
        "description": (
            "Identify all medications that are currently BELOW their reorder threshold "
            "(stock_level < reorder_point). Return ONLY valid JSON:\n"
            "{\"items_to_reorder\": [\"Medication Name 1\", \"Medication Name 2\", ...]}\n"
            "Use exact medication names as shown in the inventory."
        ),
    },
    "task_2": {
        "id": "task_2",
        "name": "Reorder Quantity Optimization",
        "difficulty": "medium",
        "max_steps": 8,
        "reward_threshold": 0.75,
        "num_items": 10,
        "description": (
            "For each medication below its reorder point, calculate the OPTIMAL quantity to order.\n"
            "Formula: optimal_qty = max(0, min(30, expiry_days) * daily_demand - stock_level)\n"
            "Return ONLY valid JSON:\n"
            "{\"order_quantities\": {\"Medication Name\": quantity_as_integer, ...}}\n"
            "Include only medications that actually need reordering."
        ),
    },
    "task_3": {
        "id": "task_3",
        "name": "Multi-Supplier Budget Planning",
        "difficulty": "hard",
        "max_steps": 10,
        "reward_threshold": 0.70,
        "num_items": 12,
        "description": (
            "Select the best supplier and quantity for each medication that needs reordering.\n"
            "CONSTRAINT: Total cost must NOT exceed the given budget.\n"
            "GOAL: Cover as many needed items as possible within budget, choosing cheapest supplier per item.\n"
            "Return ONLY valid JSON:\n"
            "{\"procurement_plan\": [\n"
            "  {\"item\": \"Medication Name\", \"supplier\": \"Supplier Name\", \"quantity\": integer},\n"
            "  ...\n"
            "]}\n"
            "Only include items the chosen supplier actually carries. Prioritize high-demand, low-cost items."
        ),
    },
}
