"""
Pydantic models for the Customer Support Ticket Triage OpenEnv environment.

Defines the Observation, Action, and Reward data structures used throughout
the environment, ensuring type safety and serialisation support.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Intent(str, Enum):
    """Allowed intent categories for ticket classification."""
    BILLING = "billing"
    TECHNICAL = "technical"
    FEATURE_REQUEST = "feature_request"
    COMPLAINT = "complaint"
    OTHER = "other"


class Priority(int, Enum):
    """Allowed priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """
    Represents a single agent action.

    Exactly one of the action fields must be set per step.

    Action types
    ------------
    - classify_intent : assign an intent label to the ticket.
    - set_priority    : assign a priority level (1‑3).
    - draft_reply     : compose a short reply (≤200 chars).
    - submit          : finalise the episode.
    """

    action_type: Literal[
        "classify_intent",
        "set_priority",
        "draft_reply",
        "submit",
    ]
    intent: Optional[str] = Field(
        default=None,
        description="Intent label; required when action_type == 'classify_intent'.",
    )
    priority: Optional[int] = Field(
        default=None,
        ge=1,
        le=3,
        description="Priority 1‑3; required when action_type == 'set_priority'.",
    )
    text: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Reply text (≤200 chars); required when action_type == 'draft_reply'.",
    )

    @field_validator("intent")
    @classmethod
    def _validate_intent(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            valid = {e.value for e in Intent}
            if v not in valid:
                raise ValueError(f"Invalid intent '{v}'. Must be one of {valid}.")
        return v

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v not in {1, 2, 3}:
            raise ValueError("Priority must be 1, 2, or 3.")
        return v


# ---------------------------------------------------------------------------
# Observation model
# ---------------------------------------------------------------------------

class ActionRecord(BaseModel):
    """A record of a single action taken within an episode."""
    step: int
    action_type: str
    detail: Optional[str] = None
    reward: float


class Observation(BaseModel):
    """
    What the agent sees after each step (or on reset).

    Fields
    ------
    ticket_title       : title of the support ticket.
    ticket_description : body / description of the support ticket.
    actions_taken      : ordered list of actions the agent has executed so far.
    reward_history     : cumulative rewards at each step.
    current_step       : 0‑indexed step counter.
    done               : whether the episode has ended.
    available_actions  : list of action types still valid.
    """

    ticket_title: str
    ticket_description: str
    actions_taken: list[ActionRecord] = Field(default_factory=list)
    reward_history: list[float] = Field(default_factory=list)
    current_step: int = 0
    done: bool = False
    available_actions: list[str] = Field(
        default_factory=lambda: [
            "classify_intent",
            "set_priority",
            "draft_reply",
            "submit",
        ]
    )


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Wraps the scalar reward with optional breakdown for interpretability.
    """

    value: float = Field(ge=-1.0, le=1.0, description="Step reward value.")
    cumulative: float = Field(
        ge=-5.0,
        le=1.0,
        description="Running total of rewards in the episode.",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Optional per‑component breakdown of the reward.",
    )


# ---------------------------------------------------------------------------
# Step result helper
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Convenience wrapper returned by env.step()."""
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
