"""
Baseline inference script for the Customer Support Triage OpenEnv.

Uses the OpenAI Chat Completions API to drive the agent through all three
tasks. Produces structured stdout logs for automated scoring.

Environment variables
---------------------
API_BASE_URL  : OpenAI‑compatible base URL (default: https://api.openai.com/v1)
MODEL_NAME    : Model to use (default: gpt-4o-mini)
OPENAI_API_KEY: API key (required)
HF_TOKEN      : Optional Hugging Face token
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

from openai import OpenAI

from env import SupportTriageEnv
from models import Action
from tasks import ALL_TASKS, Task

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# Validator uses API_KEY for proxy, but fallback to OPENAI_API_KEY for local runs
api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "dummy-api-key")
HF_TOKEN = os.getenv("HF_TOKEN", "")

try:
    client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
except Exception as e:
    print(f"Warning: OpenAI initialization failed: {e}", file=sys.stderr)
    client = None

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a customer support triage agent. You will be shown a support ticket \
and must take exactly four actions in order:

1. classify_intent – choose exactly one of: billing, technical, feature_request, complaint, other
2. set_priority – choose 1 (low), 2 (medium), or 3 (high)
3. draft_reply – write a short, professional reply (max 200 characters). \
   Include relevant keywords that address the customer's issue.
4. submit – finalise the episode.

Respond with ONLY a JSON object for each action. Examples:
  {"action_type": "classify_intent", "intent": "billing"}
  {"action_type": "set_priority", "priority": 2}
  {"action_type": "draft_reply", "text": "Thank you for reaching out..."}
  {"action_type": "submit"}

Do NOT include any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_message(task: Task, step: int, obs_dict: dict) -> str:
    """Build the user message for the current step."""
    if step == 0:
        return (
            f"Ticket Title: {task.ticket_title}\n"
            f"Ticket Description: {task.ticket_description}\n\n"
            "Begin triage. Reply with your first action as JSON."
        )
    actions_log = "\n".join(
        f"  step {a['step']}: {a['action_type']}({a.get('detail', '')})"
        for a in obs_dict.get("actions_taken", [])
    )
    return (
        f"Actions so far:\n{actions_log}\n\n"
        "Reply with your next action as JSON."
    )


def _parse_action(response_text: str) -> Action | None:
    """Parse the LLM response into an Action, returning None on failure."""
    # Try to extract JSON from the response
    text = response_text.strip()
    # Handle markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
    # Find the first JSON object
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
        return Action(**data)
    except Exception:
        return None


def _format_action_str(action: Action) -> str:
    """Format action for the log line."""
    if action.action_type == "classify_intent":
        return f"classify_intent({action.intent})"
    elif action.action_type == "set_priority":
        return f"set_priority({action.priority})"
    elif action.action_type == "draft_reply":
        preview = (action.text or "")[:50]
        return f"draft_reply({preview}...)"
    else:
        return "submit()"


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(task: Task) -> float:
    """
    Run the agent on a single task. Returns cumulative reward.
    """
    env = SupportTriageEnv(task_id=task.task_id)
    obs = env.reset()
    obs_dict = obs.model_dump()

    print(f"[START]")
    print(f"task: {task.task_id}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    total_reward = 0.0
    step = 0

    while not obs_dict["done"] and step < 10:
        user_msg = _build_user_message(task, step, obs_dict)
        messages.append({"role": "user", "content": user_msg})

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )
            assistant_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [ERROR] LLM call failed: {exc}", file=sys.stderr)
            break

        messages.append({"role": "assistant", "content": assistant_text})

        action = _parse_action(assistant_text)
        if action is None:
            # Fallback: submit to end the episode cleanly
            action = Action(action_type="submit")

        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        total_reward += reward
        step += 1

        action_str = _format_action_str(action)
        print(f"[STEP] step:{step} action:{action_str} reward:{reward}")

        if done:
            break

    # Clamp total reward
    total_reward = round(max(0.0, min(1.0, total_reward)), 4)
    print(f"[END] total_reward:{total_reward} task_score:{total_reward}")
    print()

    return total_reward


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    scores: dict[str, float] = {}

    for task in ALL_TASKS:
        score = run_episode(task)
        scores[task.task_id] = score

    # Summary
    parts = " ".join(f"{tid}={sc}" for tid, sc in scores.items())
    print(f"BASELINE_SCORES: {parts}")


if __name__ == "__main__":
    main()
