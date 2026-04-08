"""
models.py — Typed Pydantic models for MedInventoryEnv (OpenEnv spec compliant)
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Action(BaseModel):
    """Agent action: a JSON string containing the procurement decision."""
    message: str = Field(..., description="JSON string with the agent's inventory decision")


class Observation(BaseModel):
    """Environment observation returned after reset() and step()."""
    echoed_message: str = Field("", description="Last action sent by the agent")
    task_id: str = Field(..., description="Current task identifier")
    task_description: str = Field(..., description="Full task instructions for the agent")
    inventory: List[Dict[str, Any]] = Field(..., description="Current medication inventory state")
    suppliers: Optional[List[Dict[str, Any]]] = Field(None, description="Available suppliers (task_3 only)")
    budget: Optional[float] = Field(None, description="Procurement budget in INR (task_3 only)")
    step_number: int = Field(0, description="Current step within episode")
    max_steps: int = Field(5, description="Maximum steps allowed in this episode")
    feedback: str = Field("", description="Grader feedback from the previous action")


class Reward(BaseModel):
    """Reward model for OpenEnv spec."""
    value: float = Field(..., ge=0.0, le=1.0, description="Reward score between 0.0 and 1.0")
    reason: str = Field("", description="Human-readable reason for the reward")


class StepResult(BaseModel):
    """Result returned by step()."""
    observation: Observation
    reward: float = Field(0.0, ge=0.0, le=1.0)
    done: bool = False
    info: Dict[str, Any] = {}


class ResetResult(BaseModel):
    """Result returned by reset()."""
    observation: Observation
    done: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = {}


class StateResult(BaseModel):
    """Result returned by state()."""
    state: Dict[str, Any]
