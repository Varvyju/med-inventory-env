"""
env.py â€” MedInventoryEnv core environment logic
Implements reset() / step() / state() with clean episode management.
"""
import random
from typing import Dict, Any, Optional, Tuple

from models import Action, Observation, StepResult, ResetResult, StateResult
from tasks import (
    generate_inventory,
    generate_suppliers,
    grade_task1,
    grade_task2,
    grade_task3,
    TASK_CONFIGS,
)


class MedInventoryEnv:
    """
    Medical Store Inventory Management Environment.

    The agent receives pharmacy inventory data and must make procurement
    decisions (reorder identification â†’ quantity optimization â†’ supplier selection).
    Rewards are given on every step (not just at episode end) to provide
    a dense learning signal.
    """

    def __init__(self) -> None:
        self._state: Dict[str, Any] = {}
        self._step_count: int = 0
        self._task_id: str = "task_1"
        self._done: bool = True  # force reset() before first step
        self._episode_seed: int = 42
        self._best_reward: float = 0.0

    # â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def reset(self, task_id: str = "task_1", seed: Optional[int] = None) -> ResetResult:
        """Start a new episode for the given task."""
        if task_id not in TASK_CONFIGS:
            task_id = "task_1"

        self._task_id      = task_id
        self._episode_seed = seed if seed is not None else random.randint(1, 99999)
        self._step_count   = 0
        self._done         = False
        self._best_reward  = 0.0

        config    = TASK_CONFIGS[task_id]
        inventory = generate_inventory(num_items=config["num_items"], seed=self._episode_seed)

        suppliers: Optional[list] = None
        budget:    Optional[float] = None
        if task_id == "task_3":
            suppliers, budget = generate_suppliers(inventory, seed=self._episode_seed)

        self._state = {
            "task_id":      task_id,
            "task_config":  config,
            "inventory":    inventory,
            "suppliers":    suppliers,
            "budget":       budget,
            "episode_seed": self._episode_seed,
            "step_count":   0,
            "done":         False,
            "best_reward":  0.0,
        }

        items_below = sum(
            1 for i in inventory if i["stock_level"] < i["reorder_point"]
        )
        obs = self._build_observation(
            feedback=(
                f"Episode started. Inventory loaded: {len(inventory)} medications, "
                f"{items_below} currently below reorder point. "
                + (f"Budget: â‚¹{budget:.2f}. " if budget else "")
                + "Read the task description carefully and respond with valid JSON."
            )
        )
        return ResetResult(observation=obs, done=False, reward=0.0, info={})

    def step(self, action: Action) -> StepResult:
        """Process one agent action and return reward + next observation."""
        if self._done:
            obs = self._build_observation(
                "Episode is finished. Call /reset to start a new episode."
            )
            return StepResult(observation=obs, reward=0.0, done=True,
                              info={"error": "episode_already_done"})

        self._step_count           += 1
        self._state["step_count"]   = self._step_count

        config    = TASK_CONFIGS[self._task_id]
        max_steps = config["max_steps"]
        threshold = config["reward_threshold"]

        # Grade the agent's action
        raw_reward, feedback = self._grade(action.message)

        # Track best-ever reward this episode
        self._best_reward        = max(self._best_reward, raw_reward)
        self._state["best_reward"] = self._best_reward

        # Determine episode termination
        done = False
        if raw_reward >= threshold:
            done     = True
            feedback = f"âœ… SUCCESS â€” Task solved! {feedback}"
        elif self._step_count >= max_steps:
            done     = True
            feedback = (
                f"â± Max steps reached ({max_steps}). "
                f"Best score this episode: {self._best_reward:.3f}. {feedback}"
            )

        self._done        = done
        self._state["done"] = done

        # Step efficiency penalty: rewards quick solutions
        # Penalty is small (2% per extra step) so it doesn't dominate the signal
        step_penalty    = (self._step_count - 1) * 0.01
        adjusted_reward = round(max(0.001, min(0.999, raw_reward - step_penalty)), 4)

        obs = self._build_observation(feedback, last_action=action.message)
        return StepResult(
            observation=obs,
            reward=adjusted_reward,
            done=done,
            info={
                "raw_reward":   raw_reward,
                "step_penalty": step_penalty,
                "step_count":   self._step_count,
                "best_reward":  self._best_reward,
                "feedback":     feedback,
            },
        )

    def state(self) -> StateResult:
        """Return the full internal state (for debugging / validation)."""
        return StateResult(state=dict(self._state))

    # â”€â”€ Private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _grade(self, message: str) -> Tuple[float, str]:
        inventory = self._state["inventory"]
        if self._task_id == "task_1":
            return grade_task1(message, inventory)
        elif self._task_id == "task_2":
            return grade_task2(message, inventory)
        elif self._task_id == "task_3":
            return grade_task3(
                message, inventory,
                self._state["suppliers"],
                self._state["budget"],
            )
        return 0.0, "Unknown task."

    def _build_observation(
        self,
        feedback: str,
        last_action: Optional[str] = None,
    ) -> Observation:
        config    = TASK_CONFIGS[self._task_id]
        inventory = self._state["inventory"]

        # Filter inventory fields shown to agent based on task difficulty
        display_inventory = []
        for item in inventory:
            entry: Dict[str, Any] = {
                "name":          item["name"],
                "category":      item["category"],
                "stock_level":   item["stock_level"],
                "reorder_point": item["reorder_point"],
                "daily_demand":  item["daily_demand"],
            }
            if self._task_id in ("task_2", "task_3"):
                entry["expiry_days"] = item["expiry_days"]
                entry["unit_cost"]   = item["unit_cost"]
            display_inventory.append(entry)

        return Observation(
            echoed_message=last_action or "",
            task_id=self._task_id,
            task_description=config["description"],
            inventory=display_inventory,
            suppliers=self._state.get("suppliers"),
            budget=self._state.get("budget"),
            step_number=self._step_count,
            max_steps=config["max_steps"],
            feedback=feedback,
        )

