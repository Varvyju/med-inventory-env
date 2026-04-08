#!/usr/bin/env python3
"""
inference.py — MedInventoryEnv Baseline Inference Script
Runs an OpenAI-compatible LLM against all 3 tasks and reports scores.

Required environment variables:
  API_BASE_URL   — LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       — API key / Hugging Face token
  ENV_BASE_URL   — Environment server URL (default: http://localhost:7860)

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=sk-...
  python inference.py
"""

import asyncio
import json
import os
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY      = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE          = 0.1
MAX_TOKENS           = 1000
SUCCESS_SCORE_THRESHOLD = 0.70
BENCHMARK            = "MedInventoryEnv"

TASK_CONFIGS = {
    "task_1": {"name": "Reorder Threshold Identification", "max_steps": 5,  "max_total_reward": 5.0},
    "task_2": {"name": "Reorder Quantity Optimisation",    "max_steps": 8,  "max_total_reward": 8.0},
    "task_3": {"name": "Multi-Supplier Budget Planning",   "max_steps": 10, "max_total_reward": 10.0},
}

SYSTEM_PROMPT = """\
You are an expert pharmaceutical inventory manager for an Indian medical store.
You analyse inventory data and make precise, data-driven procurement decisions.
Always respond with ONLY valid JSON matching the exact format in the task description.
Do not include any explanation, markdown, or extra text — just the raw JSON object."""

# ── Structured stdout logging (DO NOT MODIFY FORMAT) ──────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Truncate action to 120 chars for readability
    action_short = action[:120].replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_short!r} "
        f"reward={reward:.4f} done={done} error={error}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = "[" + ", ".join(f"{r:.4f}" for r in rewards) + "]"
    print(
        f"[END] success={success} steps={steps} score={score:.4f} rewards={rewards_str}",
        flush=True,
    )

# ── Environment HTTP client ────────────────────────────────────────────────────

class EnvClient:
    """Lightweight async HTTP client for MedInventoryEnv REST API."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http    = httpx.AsyncClient(timeout=30.0)

    async def reset(self, task_id: str = "task_1", seed: int = 42) -> dict:
        resp = await self._http.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        resp.raise_for_status()
        return resp.json()

    async def step(self, message: str) -> dict:
        resp = await self._http.post(
            f"{self.base_url}/step",
            json={"message": message},
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._http.aclose()

# ── LLM call ──────────────────────────────────────────────────────────────────

def get_model_message(
    client: OpenAI,
    step: int,
    observation: dict,
    last_reward: float,
    history: List[str],
) -> str:
    """Build a prompt and call the LLM. Returns a JSON string."""
    inventory_text  = json.dumps(observation.get("inventory", []), indent=2)
    task_desc       = observation.get("task_description", "")
    feedback        = observation.get("feedback", "")
    step_number     = observation.get("step_number", step)
    max_steps       = observation.get("max_steps", 5)
    history_text    = "\n".join(history[-3:]) if history else "No previous steps."

    # Build supplier / budget section if present (task_3)
    extra = ""
    if observation.get("suppliers"):
        extra += f"\n\nAvailable Suppliers:\n{json.dumps(observation['suppliers'], indent=2)}"
    if observation.get("budget"):
        extra += f"\n\nProcurement Budget: ₹{observation['budget']:.2f}"

    user_prompt = f"""\
Step {step_number}/{max_steps}  |  Last reward: {last_reward:.4f}
Previous feedback: {feedback}

━━━ TASK ━━━
{task_desc}

━━━ CURRENT INVENTORY ━━━
{inventory_text}{extra}

━━━ RECENT HISTORY ━━━
{history_text}

Respond with ONLY a valid JSON object. No markdown, no explanation."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if model wraps JSON
        if text.startswith("```"):
            lines = text.splitlines()
            start = 1
            end   = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            text  = "\n".join(lines[start:end]).strip()
        return text if text else "{}"
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "{}"

# ── Single-task runner ────────────────────────────────────────────────────────

async def run_task(
    task_id: str,
    env: EnvClient,
    llm: OpenAI,
) -> float:
    config          = TASK_CONFIGS[task_id]
    task_name       = config["name"]
    max_steps       = config["max_steps"]
    max_total_reward = config["max_total_reward"]

    history:  List[str]   = []
    rewards:  List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result      = await env.reset(task_id=task_id, seed=42)
        observation = result["observation"]
        last_reward = 0.0
        done        = result.get("done", False)

        for step in range(1, max_steps + 1):
            if done:
                break

            message = get_model_message(llm, step, observation, last_reward, history)

            result      = await env.step(message)
            observation = result["observation"]
            reward      = float(result.get("reward", 0.0))
            done        = result.get("done", False)
            error       = result.get("info", {}).get("error")

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)
            history.append(
                f"Step {step}: reward={reward:.4f} | "
                f"feedback={str(observation.get('feedback',''))[:80]}"
            )

            if done:
                break

        score   = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} crashed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score

# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"[DEBUG] MedInventoryEnv baseline | model={MODEL_NAME} | env={ENV_BASE_URL}", flush=True)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(base_url=ENV_BASE_URL)

    all_scores: dict = {}

    try:
        for task_id in ["task_1", "task_2", "task_3"]:
            print(f"\n[DEBUG] ═══════ {task_id} ═══════", flush=True)
            score = await run_task(task_id, env, llm)
            all_scores[task_id] = score
            await asyncio.sleep(0.5)

    finally:
        await env.close()

    print("\n[DEBUG] ════════ FINAL SCORES ════════", flush=True)
    for tid, sc in all_scores.items():
        print(f"[DEBUG] {TASK_CONFIGS[tid]['name']}: {sc:.4f}", flush=True)
    overall = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Overall average: {overall:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
