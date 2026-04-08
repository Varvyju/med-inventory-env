"""
main.py — FastAPI server for MedInventoryEnv
Exposes /reset, /step, /state endpoints per OpenEnv spec.
Runs on port 7860 (Hugging Face Spaces default).
"""
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import MedInventoryEnv
from models import Action, ResetResult, StateResult, StepResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedInventoryEnv",
    description=(
        "OpenEnv-compliant environment for medical store inventory management. "
        "AI agents learn to identify reorder needs, optimise quantities, and plan "
        "multi-supplier procurement within budget constraints."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared environment instance
env = MedInventoryEnv()


# ── Request models ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "task_1"
    seed: Optional[int] = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check — confirms the environment is live."""
    return {
        "status":  "ok",
        "env":     "MedInventoryEnv",
        "version": "1.0.0",
        "tasks":   ["task_1", "task_2", "task_3"],
        "docs":    "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.
    - task_id: "task_1" (easy) | "task_2" (medium) | "task_3" (hard)
    - seed: optional int for reproducibility (omit for random)
    """
    try:
        logger.info(f"reset() called — task_id={request.task_id}, seed={request.seed}")
        result = env.reset(task_id=request.task_id or "task_1", seed=request.seed)
        return result
    except Exception as e:
        logger.error(f"reset() error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Send one action to the environment.
    - action.message: JSON string with the agent's inventory decision
    Returns observation, reward (0.0–1.0), done flag, and info dict.
    """
    try:
        logger.info(f"step() called — message length={len(action.message)}")
        result = env.step(action)
        logger.info(f"step() → reward={result.reward:.4f}, done={result.done}")
        return result
    except Exception as e:
        logger.error(f"step() error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResult)
def state():
    """Return the full current environment state (for debugging / validation)."""
    try:
        return env.state()
    except Exception as e:
        logger.error(f"state() error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def list_tasks():
    """List all available tasks with their configuration."""
    from tasks import TASK_CONFIGS
    return {"tasks": list(TASK_CONFIGS.values())}


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
        reload=False,
    )
