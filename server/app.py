"""
FastAPI wrapper for the Customer Support Triage OpenEnv environment.

Endpoints
---------
POST /reset   → reset the env (optionally with a task_id), returns Observation.
POST /step    → accept an Action, returns (obs, reward, done, info).
GET  /state   → returns current internal state.
GET  /health  → returns 200 OK.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from env import SupportTriageEnv
from models import Action, Observation

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Support Triage – OpenEnv API",
    version="1.0.0",
    description=(
        "An OpenEnv‑compatible REST API that simulates customer support "
        "ticket triage. The agent classifies intent, sets priority, drafts "
        "a reply, and submits."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single‑tenant for simplicity / HF Space)
_env = SupportTriageEnv(task_id="easy")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(
        default="easy",
        description="Task to load: 'easy', 'medium', or 'hard'.",
    )


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset_env(req: ResetRequest = ResetRequest()) -> Observation:
    """Reset the environment and return the initial observation."""
    try:
        obs = _env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=StepResponse)
def step_env(action: Action) -> StepResponse:
    """Execute one action and return the step result."""
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def get_state() -> dict[str, Any]:
    """Return the full internal state of the environment."""
    return _env.state()


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Simple health / readiness probe."""
    return HealthResponse()


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
