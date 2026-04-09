"""
Unit tests for the Customer Support Triage OpenEnv environment.

Run with:  pytest test_env.py -v
"""

from __future__ import annotations

import pytest

from env import SupportTriageEnv
from models import Action, Observation
from tasks import ALL_TASKS, TASKS_BY_ID, grader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["easy", "medium", "hard"])
def env(request: pytest.FixtureRequest) -> SupportTriageEnv:
    """Parametrised fixture: one env per task."""
    return SupportTriageEnv(task_id=request.param)


@pytest.fixture
def easy_env() -> SupportTriageEnv:
    return SupportTriageEnv(task_id="easy")


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env: SupportTriageEnv) -> None:
        obs = env.reset()
        assert isinstance(obs, Observation)
        assert obs.current_step == 0
        assert obs.done is False
        assert len(obs.actions_taken) == 0

    def test_reset_shows_ticket(self, easy_env: SupportTriageEnv) -> None:
        obs = easy_env.reset()
        assert "invoice" in obs.ticket_title.lower() or "charge" in obs.ticket_description.lower()

    def test_reset_with_different_task(self, easy_env: SupportTriageEnv) -> None:
        obs = easy_env.reset(task_id="hard")
        assert "outage" in obs.ticket_description.lower() or "unacceptable" in obs.ticket_description.lower()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------

class TestStep:
    def test_classify_intent_correct(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        obs, reward, done, info = easy_env.step(
            Action(action_type="classify_intent", intent="billing")
        )
        assert reward == pytest.approx(0.2)
        assert done is False
        assert len(obs.actions_taken) == 1

    def test_classify_intent_wrong(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        obs, reward, done, info = easy_env.step(
            Action(action_type="classify_intent", intent="technical")
        )
        assert reward == pytest.approx(0.0)

    def test_set_priority_correct(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="classify_intent", intent="billing"))
        obs, reward, done, info = easy_env.step(
            Action(action_type="set_priority", priority=2)
        )
        assert reward == pytest.approx(0.2)

    def test_draft_reply(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        obs, reward, done, info = easy_env.step(
            Action(
                action_type="draft_reply",
                text="We will review your invoice and refund the duplicate charge.",
            )
        )
        assert reward > 0.0
        assert reward <= 0.4

    def test_submit_ends_episode(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="classify_intent", intent="billing"))
        easy_env.step(Action(action_type="set_priority", priority=2))
        easy_env.step(
            Action(
                action_type="draft_reply",
                text="We will review your invoice and refund the duplicate charge.",
            )
        )
        obs, reward, done, info = easy_env.step(Action(action_type="submit"))
        assert done is True
        assert reward == pytest.approx(0.1)

    def test_invalid_action_penalty(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        obs, reward, done, info = easy_env.step(
            Action(action_type="classify_intent")  # missing intent
        )
        assert reward == pytest.approx(-0.1)
        assert "error" in info

    def test_repeat_action_penalty(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="classify_intent", intent="billing"))
        obs, reward, done, info = easy_env.step(
            Action(action_type="classify_intent", intent="billing")
        )
        assert reward == pytest.approx(-0.2)

    def test_step_after_done_raises(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="submit"))
        with pytest.raises(RuntimeError):
            easy_env.step(Action(action_type="classify_intent", intent="billing"))


# ---------------------------------------------------------------------------
# Step limit
# ---------------------------------------------------------------------------

class TestStepLimit:
    def test_step_limit_ends_episode(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        done = False
        for i in range(10):
            if done:
                break
            # Alternate invalid actions to reach the limit
            obs, reward, done, info = easy_env.step(
                Action(action_type="draft_reply", text=f"msg {i}")
            )
        assert done is True


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TestState:
    def test_state_serialisable(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="classify_intent", intent="billing"))
        state = easy_env.state()
        assert isinstance(state, dict)
        assert state["task_id"] == "easy"
        assert state["current_step"] == 1
        assert state["classified_intent"] == "billing"

    def test_state_after_full_episode(self, easy_env: SupportTriageEnv) -> None:
        easy_env.reset()
        easy_env.step(Action(action_type="classify_intent", intent="billing"))
        easy_env.step(Action(action_type="set_priority", priority=2))
        easy_env.step(
            Action(
                action_type="draft_reply",
                text="We will review your invoice and the charge.",
            )
        )
        easy_env.step(Action(action_type="submit"))
        state = easy_env.state()
        assert state["done"] is True
        assert state["submitted"] is True
        assert state["cumulative_reward"] > 0


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class TestGrader:
    def test_perfect_easy(self) -> None:
        task = TASKS_BY_ID["easy"]
        outputs = {
            "intent": "billing",
            "priority": 2,
            "reply": "We will review your invoice and refund the duplicate charge right away.",
            "submitted": True,
        }
        score = grader(task, outputs)
        assert 0.8 <= score <= 1.0

    def test_zero_score(self) -> None:
        task = TASKS_BY_ID["easy"]
        outputs = {
            "intent": None,
            "priority": None,
            "reply": None,
            "submitted": False,
        }
        score = grader(task, outputs)
        assert score == pytest.approx(0.01)

    def test_partial_medium(self) -> None:
        task = TASKS_BY_ID["medium"]
        outputs = {
            "intent": "technical",
            "priority": 3,
            "reply": "please send us your logs so we can troubleshoot the crash",
            "submitted": True,
        }
        score = grader(task, outputs)
        assert 0.6 <= score <= 1.0

    def test_hard_with_banned_phrase(self) -> None:
        task = TASKS_BY_ID["hard"]
        outputs = {
            "intent": "complaint",
            "priority": 3,
            "reply": "We apologize for the outage. That's not our problem though. Here's a credit.",
            "submitted": True,
        }
        score = grader(task, outputs)
        # Should lose points for the banned phrase
        assert score < 0.9

    def test_all_graders_return_valid_range(self) -> None:
        for task in ALL_TASKS:
            outputs = {
                "intent": task.ground_truth_intent,
                "priority": task.ground_truth_priority,
                "reply": "sorry for the inconvenience, we acknowledge the issue and offer a monthly credit. Please send logs so we can troubleshoot. We will review your invoice and refund the duplicate charge.",
                "submitted": True,
            }
            score = grader(task, outputs)
            assert 0.0 <= score <= 1.0, f"Grader for {task.task_id} returned {score}"
