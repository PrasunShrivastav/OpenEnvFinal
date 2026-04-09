"""
Task definitions and deterministic graders for the Customer Support Triage env.

Each task provides:
  - A support ticket (title + description).
  - Ground‑truth intent, priority, and reply evaluation criteria.
  - A deterministic grader that scores the agent's final output in [0.0, 1.0].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Task:
    """Immutable definition of a single evaluation task."""

    task_id: str
    difficulty: str  # "easy", "medium", "hard"
    ticket_title: str
    ticket_description: str
    ground_truth_intent: str
    ground_truth_priority: int
    required_keywords: list[str] = field(default_factory=list)
    banned_phrases: list[str] = field(default_factory=list)
    extra_reply_checks: list[Callable[[str], bool]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grader(task: Task, agent_outputs: dict) -> float:
    """
    Deterministic grader that evaluates agent performance on a task.

    Parameters
    ----------
    task : Task
        The task definition (includes ground truth).
    agent_outputs : dict
        Must contain keys:
          - "intent"   : str | None
          - "priority" : int | None
          - "reply"    : str | None
          - "submitted": bool

    Returns
    -------
    float  in [0.0, 1.0]
    """
    score = 0.05

    # --- Intent (0.2) -------------------------------------------------
    if agent_outputs.get("intent") == task.ground_truth_intent:
        score += 0.2

    # --- Priority (0.2) -----------------------------------------------
    if agent_outputs.get("priority") == task.ground_truth_priority:
        score += 0.2

    # --- Reply quality (0.0 – 0.4) ------------------------------------
    reply: str = (agent_outputs.get("reply") or "").lower()
    reply_score = 0.0

    if reply:
        # Required keywords
        if task.required_keywords:
            matched = sum(
                1 for kw in task.required_keywords if kw.lower() in reply
            )
            keyword_ratio = matched / len(task.required_keywords)
            reply_score += 0.2 * keyword_ratio

        # Banned phrases
        has_banned = any(bp.lower() in reply for bp in task.banned_phrases)
        if not has_banned:
            reply_score += 0.1

        # Extra checks (each worth equal share of remaining 0.1)
        if task.extra_reply_checks:
            passed = sum(1 for chk in task.extra_reply_checks if chk(reply))
            reply_score += 0.1 * (passed / len(task.extra_reply_checks))
        else:
            reply_score += 0.1  # no extra checks → full marks

    score += min(reply_score, 0.4)

    # --- Submit bonus (0.1) -------------------------------------------
    if agent_outputs.get("submitted", False):
        score += 0.1

    return round(score, 4)


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASK_EASY = Task(
    task_id="easy",
    difficulty="easy",
    ticket_title="Double charge on my last invoice",
    ticket_description=(
        "Hi, I noticed that my credit card was charged twice for the "
        "same subscription renewal last month. The amount is $49.99 each "
        "time. Could you please look into this and refund the duplicate "
        "charge? My account email is alice@example.com."
    ),
    ground_truth_intent="billing",
    ground_truth_priority=2,
    required_keywords=["invoice", "charge"],
    banned_phrases=[],
    extra_reply_checks=[],
)

TASK_MEDIUM = Task(
    task_id="medium",
    difficulty="medium",
    ticket_title="App keeps crashing after update",
    ticket_description=(
        "Since the latest update (v4.2.1) the mobile app crashes every "
        "time I try to open the dashboard. I've tried restarting my phone "
        "but it still happens. Not sure what else to try."
    ),
    ground_truth_intent="technical",
    ground_truth_priority=3,
    required_keywords=["logs", "troubleshoot"],
    banned_phrases=["not our fault", "user error"],
    extra_reply_checks=[
        # Must ask for logs or diagnostics
        lambda r: any(w in r for w in ["logs", "log file", "diagnostics", "crash report"]),
        # Must offer a concrete troubleshooting step
        lambda r: any(
            w in r
            for w in [
                "reinstall",
                "clear cache",
                "update",
                "uninstall",
                "restart",
                "safe mode",
                "troubleshoot",
            ]
        ),
    ],
)

TASK_HARD = Task(
    task_id="hard",
    difficulty="hard",
    ticket_title="Furious – third outage this month, losing customers",
    ticket_description=(
        "This is unacceptable. Your platform has gone down THREE times "
        "this month and each outage lasted over an hour. My e‑commerce "
        "store lost real revenue because of this. I've been a paying "
        "customer for two years and I'm seriously considering switching "
        "to a competitor. What are you going to do about this?"
    ),
    ground_truth_intent="complaint",
    ground_truth_priority=3,
    required_keywords=["apologize", "acknowledge", "credit"],
    banned_phrases=[
        "not our problem",
        "nothing we can do",
        "your fault",
        "we disagree",
        "that's expected",
    ],
    extra_reply_checks=[
        # Must contain an apology
        lambda r: any(w in r for w in ["sorry", "apologize", "apologies", "apology", "regret"]),
        # Must acknowledge the problem
        lambda r: any(
            w in r
            for w in ["outage", "downtime", "disruption", "inconvenience", "issue", "problem"]
        ),
        # Must offer compensation
        lambda r: any(
            w in r
            for w in ["credit", "refund", "discount", "compensation", "complimentary", "free month"]
        ),
    ],
)

# Ordered list for easy iteration
ALL_TASKS: list[Task] = [TASK_EASY, TASK_MEDIUM, TASK_HARD]

TASKS_BY_ID: dict[str, Task] = {t.task_id: t for t in ALL_TASKS}
