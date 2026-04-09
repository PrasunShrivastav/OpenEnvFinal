"""
OpenEnv‑compatible environment for Customer Support Ticket Triage.

The agent must read a support ticket, classify its intent, set a priority,
draft a reply, and submit — earning dense, partial‑progress rewards.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from models import Action, ActionRecord, Observation, Reward, StepResult
from tasks import Task, TASKS_BY_ID, ALL_TASKS


MAX_STEPS = 10


class SupportTriageEnv:
    """
    Customer Support Ticket Triage environment.

    Implements the OpenEnv interface: reset(), step(), state().
    """

    def __init__(self, task_id: str = "easy") -> None:
        """
        Initialise the environment for the given task.

        Parameters
        ----------
        task_id : str
            One of "easy", "medium", "hard".
        """
        if task_id not in TASKS_BY_ID:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Choose from {list(TASKS_BY_ID)}."
            )
        self._task: Task = TASKS_BY_ID[task_id]
        self._reset_internals()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, task_id: str | None = None) -> Observation:
        """
        Reset the environment, optionally switching to a different task.

        Returns the initial observation.
        """
        if task_id is not None:
            if task_id not in TASKS_BY_ID:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. Choose from {list(TASKS_BY_ID)}."
                )
            self._task = TASKS_BY_ID[task_id]
        self._reset_internals()
        return self._make_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict[str, Any]]:
        """
        Execute one action and return (observation, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode already finished. Call reset().")

        self._current_step += 1
        reward = 0.0
        info: dict[str, Any] = {}

        # --- Validate & execute action --------------------------------
        valid, reason = self._validate_action(action)
        if not valid:
            reward = -0.1
            info["error"] = reason
            self._record_action(action, reward, detail=reason)
        else:
            reward = self._execute_action(action)
            self._record_action(action, reward)

        # --- Check termination ----------------------------------------
        if action.action_type == "submit" and valid:
            self._done = True
            info["reason"] = "submitted"
        elif self._current_step >= MAX_STEPS:
            reward += -0.3  # overtime penalty
            self._done = True
            info["reason"] = "step_limit"

        # Accumulate
        self._cumulative_reward += reward
        self._cumulative_reward = round(
            max(0.0, min(1.0, self._cumulative_reward)), 4
        )
        self._reward_history.append(self._cumulative_reward)

        obs = self._make_observation()
        return obs, round(reward, 4), self._done, info

    def state(self) -> dict[str, Any]:
        """Return the full serialisable internal state."""
        return {
            "task_id": self._task.task_id,
            "current_step": self._current_step,
            "done": self._done,
            "cumulative_reward": self._cumulative_reward,
            "classified_intent": self._classified_intent,
            "assigned_priority": self._assigned_priority,
            "drafted_reply": self._drafted_reply,
            "submitted": self._submitted,
            "actions_taken": [r.model_dump() for r in self._actions_taken],
            "reward_history": list(self._reward_history),
        }

    # ------------------------------------------------------------------
    # Dense reward computation
    # ------------------------------------------------------------------

    def _compute_reward(self, action: Action, *, is_repeat: bool) -> float:
        """
        Compute dense, non‑binary reward for a valid action.

        Component weights
        -----------------
        classify_intent  : +0.2 if matches ground truth, else +0.0
        set_priority     : +0.2 if matches ground truth, else +0.0
        draft_reply      : 0.0 – 0.4 based on keyword / banned‑phrase checks
        submit           : +0.1 if all required actions were taken, else +0.0
        Repeat penalty   : −0.2 if the same action type was already executed
        """
        if is_repeat:
            return -0.2

        reward = 0.0

        if action.action_type == "classify_intent":
            if action.intent == self._task.ground_truth_intent:
                reward = 0.2

        elif action.action_type == "set_priority":
            if action.priority == self._task.ground_truth_priority:
                reward = 0.2

        elif action.action_type == "draft_reply":
            reward = self._score_reply(action.text or "")

        elif action.action_type == "submit":
            # Bonus only when all three required actions have been taken
            if (
                self._classified_intent is not None
                and self._assigned_priority is not None
                and self._drafted_reply is not None
            ):
                reward = 0.1

        return reward

    def _score_reply(self, text: str) -> float:
        """Score reply text on a 0.0 – 0.4 scale."""
        lower = text.lower()
        score = 0.0

        # Required keywords (worth up to 0.2)
        if self._task.required_keywords:
            matched = sum(
                1 for kw in self._task.required_keywords if kw.lower() in lower
            )
            score += 0.2 * (matched / len(self._task.required_keywords))

        # Banned phrases (lose 0.1 if any present)
        has_banned = any(bp.lower() in lower for bp in self._task.banned_phrases)
        if not has_banned:
            score += 0.1

        # Extra checks (up to 0.1)
        if self._task.extra_reply_checks:
            passed = sum(1 for chk in self._task.extra_reply_checks if chk(lower))
            score += 0.1 * (passed / len(self._task.extra_reply_checks))
        else:
            score += 0.1

        return round(min(score, 0.4), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_internals(self) -> None:
        self._current_step: int = 0
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._classified_intent: str | None = None
        self._assigned_priority: int | None = None
        self._drafted_reply: str | None = None
        self._submitted: bool = False
        self._actions_taken: list[ActionRecord] = []
        self._reward_history: list[float] = []
        self._action_types_used: set[str] = set()

    def _validate_action(self, action: Action) -> tuple[bool, str]:
        """Return (is_valid, reason)."""
        at = action.action_type

        if at == "classify_intent":
            if action.intent is None:
                return False, "classify_intent requires 'intent' field."
        elif at == "set_priority":
            if action.priority is None:
                return False, "set_priority requires 'priority' field."
        elif at == "draft_reply":
            if not action.text:
                return False, "draft_reply requires non‑empty 'text' field."
        # submit needs no extra validation

        return True, ""

    def _execute_action(self, action: Action) -> float:
        """
        Apply a validated action to the internal state and return reward.
        """
        at = action.action_type
        is_repeat = at in self._action_types_used
        self._action_types_used.add(at)

        if at == "classify_intent":
            self._classified_intent = action.intent
        elif at == "set_priority":
            self._assigned_priority = action.priority
        elif at == "draft_reply":
            self._drafted_reply = action.text
        elif at == "submit":
            self._submitted = True

        return self._compute_reward(action, is_repeat=is_repeat)

    def _record_action(
        self,
        action: Action,
        reward: float,
        detail: str | None = None,
    ) -> None:
        detail_str = detail
        if detail_str is None:
            if action.action_type == "classify_intent":
                detail_str = action.intent
            elif action.action_type == "set_priority":
                detail_str = str(action.priority)
            elif action.action_type == "draft_reply":
                detail_str = (action.text or "")[:80]
            else:
                detail_str = None

        self._actions_taken.append(
            ActionRecord(
                step=self._current_step,
                action_type=action.action_type,
                detail=detail_str,
                reward=round(reward, 4),
            )
        )

    def _make_observation(self) -> Observation:
        used = set(self._action_types_used)
        available = [
            at
            for at in ["classify_intent", "set_priority", "draft_reply", "submit"]
            if at not in used or at == "submit"
        ]
        return Observation(
            ticket_title=self._task.ticket_title,
            ticket_description=self._task.ticket_description,
            actions_taken=list(self._actions_taken),
            reward_history=list(self._reward_history),
            current_step=self._current_step,
            done=self._done,
            available_actions=available,
        )

    # ------------------------------------------------------------------
    # Agent outputs (for grader)
    # ------------------------------------------------------------------

    def get_agent_outputs(self) -> dict[str, Any]:
        """Return the finalised agent outputs for grading."""
        return {
            "intent": self._classified_intent,
            "priority": self._assigned_priority,
            "reply": self._drafted_reply,
            "submitted": self._submitted,
        }
