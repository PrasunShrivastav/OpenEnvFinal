---
title: Customer Support Ticket Triage
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
sdk_version: "latest"
app_file: api.py
pinned: false
---

# Customer Support Ticket Triage – OpenEnv

An **OpenEnv‑compatible** reinforcement‑learning environment that simulates
the real‑world task of **customer support ticket triage and response
prioritisation**.

## Real‑World Motivation

Every company with a support channel faces the same bottleneck: incoming
tickets must be **read**, **categorised**, **prioritised**, and **replied to**
as quickly and accurately as possible. Hiring and training human agents is
expensive; an AI agent that can triage tickets — or at least produce a strong
first draft — dramatically reduces response times and improves customer
satisfaction.

This environment lets you train and evaluate such an agent in a controlled,
graded setting.

---

## Action Space (Discrete + Structured)

| Action              | Parameters                               | Description                                |
| ------------------- | ---------------------------------------- | ------------------------------------------ |
| `classify_intent`   | `intent: str` — one of `billing`, `technical`, `feature_request`, `complaint`, `other` | Assign an intent label to the ticket.      |
| `set_priority`      | `priority: int` — 1 (low), 2 (med), 3 (high) | Assign a priority level.                   |
| `draft_reply`       | `text: str` — max 200 characters         | Compose a short reply to the customer.     |
| `submit`            | *(none)*                                 | Finalise the episode.                      |

## Observation Space

| Field                | Type          | Description                                       |
| -------------------- | ------------- | ------------------------------------------------- |
| `ticket_title`       | `str`         | Title of the support ticket.                       |
| `ticket_description` | `str`         | Full body / description of the ticket.             |
| `actions_taken`      | `list[dict]`  | Ordered log of actions (step, type, detail, reward). |
| `reward_history`     | `list[float]` | Cumulative reward at each completed step.          |
| `current_step`       | `int`         | 0‑indexed step counter.                           |
| `done`               | `bool`        | Whether the episode has ended.                     |
| `available_actions`  | `list[str]`   | Action types the agent may still use.              |

## Reward Function (Dense, Partial Progress)

| Component           | Reward     | Condition                                          |
| ------------------- | ---------- | -------------------------------------------------- |
| Correct intent      | +0.2       | Matches ground truth                               |
| Correct priority    | +0.2       | Matches ground truth                               |
| Reply quality       | 0.0 – 0.4 | Keywords present, no banned phrases, extra checks  |
| Submit bonus        | +0.1       | All required actions taken before submit            |
| Invalid action      | −0.1       | Missing required fields                            |
| Repeat action       | −0.2       | Same action type used again                        |
| Step limit exceeded  | −0.3       | More than 10 steps                                 |

Cumulative episode reward is clamped to **[0.0, 1.0]**.

---

## Tasks

### 1. Easy — Billing Question

- **Ticket**: Double charge on a subscription invoice.
- **Ground truth**: intent = `billing`, priority = `2`.
- **Reply criteria**: Must mention "invoice" or "charge".

### 2. Medium — Technical Issue

- **Ticket**: App crashes after update, vague description.
- **Ground truth**: intent = `technical`, priority = `3`.
- **Reply criteria**: Must ask for logs **and** offer a troubleshooting step.

### 3. Hard — Angry Complaint

- **Ticket**: Repeated platform outages causing revenue loss.
- **Ground truth**: intent = `complaint`, priority = `3`.
- **Reply criteria**: Must apologise, acknowledge the problem, and offer
  compensation (e.g., "monthly credit"). Must **not** contain defensive or
  dismissive language.

---

## Installation

```bash
# Clone the repository
git clone <repo-url> && cd customer-support-triage

# Create a virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Baseline

```bash
export OPENAI_API_KEY="sk-..."
export MODEL_NAME="gpt-4o-mini"          # optional, defaults to gpt-4o-mini
export API_BASE_URL="https://api.openai.com/v1"  # optional

python inference.py
```

### Expected output format

```
[START]
task: easy
[STEP] step:1 action:classify_intent(billing) reward:0.2
[STEP] step:2 action:set_priority(2) reward:0.2
[STEP] step:3 action:draft_reply(Thank you... invoice...) reward:0.3
[STEP] step:4 action:submit() reward:0.1
[END] total_reward:0.8 task_score:0.8

...

BASELINE_SCORES: easy=0.XX medium=0.XX hard=0.XX
```

### Baseline Scores (Approximate)

| Task   | Score |
| ------ | ----- |
| Easy   | 0.80  |
| Medium | 0.65  |
| Hard   | 0.40  |

---

## Running with Docker

```bash
# Build the image
docker build -t support-triage .

# Run the container
docker run -p 7860:7860 support-triage
```

### API Endpoints

| Method | Path      | Description                          |
| ------ | --------- | ------------------------------------ |
| POST   | `/reset`  | Reset env; body: `{"task_id": "easy"}` |
| POST   | `/step`   | Send action; body: Action JSON       |
| GET    | `/state`  | Get current internal state            |
| GET    | `/health` | Health check (returns 200)            |

---

## Running Tests

```bash
pytest test_env.py -v
```

## Validating with OpenEnv

```bash
openenv validate
```

---

## Project Structure

```
├── env.py            # OpenEnv environment (reset/step/state)
├── models.py         # Pydantic data models
├── tasks.py          # 3 tasks + deterministic graders
├── openenv.yaml      # OpenEnv metadata
├── inference.py      # Baseline LLM agent script
├── api.py            # FastAPI wrapper for HF Space
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
├── test_env.py       # Unit tests
└── README.md         # This file
```

## License

MIT
