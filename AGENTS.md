# AGENTS.md — Shared AI Context for ShopSense Env

> **READ THIS FIRST.** If you are an AI assistant helping on this project,
> read this entire file before writing a single line of code or making any
> suggestion. This is the single source of truth for the project.

---

## 1. Project Overview

**Competition:** Meta PyTorch OpenEnv Hackathon — Round 1
**Repo:** https://github.com/Viscous106/shopsense-env
**Team:** Yash (Viscous106) + Teammate
**Deadline:** April 8, 2026 11:59 PM IST

### What We Are Building

A **Reinforcement Learning environment** called **ShopSense** that simulates a
shopkeeper's assistant predicting customer purchases. An LLM agent observes a
customer's purchase history and predicts what category they will buy next. The
environment scores predictions and returns a normalized reward.

The environment is served as a **FastAPI HTTP server** and deployed to
**Hugging Face Spaces** using the `openenv` framework.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Framework | `openenv` (Meta's OpenEnv SDK) |
| Server | FastAPI via `openenv` base classes |
| Models | Pydantic (via `openenv.core.env_server.types`) |
| Package manager | `uv` (NOT pip directly) |
| Deployment | Hugging Face Spaces (Docker) |
| Config | `openenv.yaml` |

### Critical: How openenv base classes work

All models inherit from `openenv` base types — do NOT redefine from scratch:

```python
from openenv.core.env_server.types import Action, Observation, State
from openenv.core.env_server.interfaces import Environment
from pydantic import Field
```

- `Action` → base class for `ShopsenseAction`
- `Observation` → base class for `ShopsenseObservation` (has `.done`, `.reward`, `.metadata` built in)
- `State` → base class for episode state (has `.episode_id`, `.step_count` built in)
- `Environment` → base class for `ShopsenseEnvironment` (must implement `reset()`, `step()`, `state`)

---

## 3. Data Models (Issue #2 — `models.py`)

**Status:** ✅ Implemented — on `feature/models` branch
**File:** `models.py` (root of repo)

### Categories (6 total — DO NOT simplify to 4)
```python
CATEGORIES = ["medical", "sports", "stationary", "groceries", "fruits", "generic"]
```

### Implemented models:

```python
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

CATEGORIES = ["medical", "sports", "stationary", "groceries", "fruits", "generic"]

class ShopsenseAction(Action):
    """Agent's prediction of the next purchase category."""
    customer_id: str = Field(..., description="ID of the customer being observed (e.g. 'C001')")
    predicted_category: str = Field(..., description="One of: medical, sports, stationary, groceries, fruits, generic")

class ShopsenseObservation(Observation):
    """What the agent sees after taking an action."""
    customer_id: str = Field(default="", description="Customer ID for this episode")
    purchase_history: list[str] = Field(default_factory=list, description="All purchases so far (including warmup)")
    actual_category: str = Field(default="", description="Ground truth category revealed after prediction")
    score_so_far: float = Field(default=0.0, description="Normalized running score [0.0, 1.0]")
    step: int = Field(default=0, description="Current step number (1-indexed after first step)")
    total_steps: int = Field(default=0, description="Total steps in this episode")
    # NOTE: .reward (0.0 or 1.0), .done (bool), .metadata (dict) come from Observation base class
```

### What NOT to do:
- Do NOT add a `ShopSenseState` class — `State` from openenv base is sufficient
- Do NOT use plain `BaseModel` — always inherit from `Action` / `Observation`
- Do NOT rename `ShopsenseAction` / `ShopsenseObservation` — referenced in `client.py` and `__init__.py`
- Do NOT use 4-category simplification — the original plan uses **6 categories**

---

## 4. Customer Profiles (Issue #8 GH — `data_gen.py`)

**File:** `data_gen.py` (root of repo)
**Branch:** `feature/data-gen`

### 4 Customers with Probability Distributions

> ⚠️ C001 and C004 are **both doctors** but have completely different patterns.
> A naive model can't just memorize "Doctor → medical". It must learn from purchase history.

```python
CUSTOMER_DISTRIBUTIONS = {
    "C001": {"medical": 0.80, "generic": 0.20},                                 # Doctor 1
    "C002": {"sports": 0.65, "groceries": 0.20, "generic": 0.15},              # Athlete
    "C003": {"stationary": 0.40, "groceries": 0.40, "generic": 0.20},          # Teacher
    "C004": {"stationary": 0.40, "fruits": 0.40, "generic": 0.20},             # Doctor 2
}
```

### Functions to implement:

```python
import random

def sample_purchase(customer_id: str) -> str:
    """Sample one category from the customer's distribution using random.choices."""
    dist = CUSTOMER_DISTRIBUTIONS[customer_id]
    return random.choices(list(dist.keys()), weights=list(dist.values()))[0]

def generate_warmup_history(customer_id: str, n: int = 5) -> list[str]:
    """Return n sampled purchases to seed the episode history."""
    return [sample_purchase(customer_id) for _ in range(n)]

def get_all_customer_ids() -> list[str]:
    """Return ['C001', 'C002', 'C003', 'C004']."""
    return list(CUSTOMER_DISTRIBUTIONS.keys())
```

---

## 5. Reward Logic (Issue #9 GH — `reward.py`)

**File:** `reward.py` (root of repo)
**Branch:** `feature/core-env-v2`

```python
def compute_reward(predicted: str, actual: str) -> float:
    """Binary reward: 1.0 if prediction matches ground truth, else 0.0."""
    return 1.0 if predicted.strip().lower() == actual.strip().lower() else 0.0

def normalize_score(correct: int, total: int) -> float:
    return correct / total if total > 0 else 0.0
```

- Reward is always exactly **0.0 or 1.0** — never fractional
- Comparison is **case-insensitive** and **strips whitespace**
- `score_so_far = correct_count / steps_taken` (updated after every step)

---

## 6. Core Environment Logic (Issue #9 GH — `server/shopsense_env_environment.py`)

**File:** `server/shopsense_env_environment.py`
**Branch:** `feature/core-env-v2`

### Current state: Still echo-template — needs full rewrite

### Target implementation outline:

```python
import random
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ShopsenseAction, ShopsenseObservation
    from ..data_gen import sample_purchase, generate_warmup_history
    from ..reward import compute_reward
except ImportError:
    from models import ShopsenseAction, ShopsenseObservation
    from data_gen import sample_purchase, generate_warmup_history
    from reward import compute_reward

class ShopsenseEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def reset(self, customer_ids: list[str] = None, total_steps: int = 20) -> ShopsenseObservation:
        # 1. Pick random customer from customer_ids (or all if None)
        # 2. Generate 5-item warmup via generate_warmup_history()
        # 3. Reset state (step=0, correct=0, score=0.0)
        # 4. Return ShopsenseObservation with purchase_history=warmup, done=False, reward=0.0

    def step(self, action: ShopsenseAction) -> ShopsenseObservation:
        # 1. actual = sample_purchase(self._customer_id)
        # 2. r = compute_reward(action.predicted_category, actual)
        # 3. purchase_history.append(actual), step++, correct += r
        # 4. score_so_far = correct / step
        # 5. done = (step >= total_steps)
        # 6. Return ShopsenseObservation with all fields filled

    @property
    def state(self) -> State:
        return self._state
```

### Key constraints:
- `reset()` may need `**kwargs` — check base class signature before overriding
- `step()` must accept exactly `ShopsenseAction` (not generic `Action`)
- `state` is a `@property`, not a method

---

## 7. Task Definitions (Issue #3 — `tasks/`)

**Branch:** `feature/tasks`

| Task | Customers | Steps | Expected baseline |
|---|---|---|---|
| Easy | C001 only | 20 | ~0.80 |
| Medium | C001 + C002 (random each ep.) | 30 | ~0.70 |
| Hard | All 4 customers | 40 | ~0.55 |

> ⚠️ Hard task: C001 vs C004 disambiguation is the key challenge (both doctors, different patterns)

Each task file exposes a `TaskConfig` with `customer_ids` and `total_steps`.

---

## 8. FastAPI Server (Issue #4 — `server/app.py`)

**Branch:** `feature/api-server`

### Critical endpoints (judges auto-ping these):

| Method | Route | Must return |
|---|---|---|
| GET | `/` | HTTP 200 (health check — **auto-disqualified if fails**) |
| POST | `/reset` | Initial `ShopsenseObservation` |
| POST | `/step` | `ShopsenseObservation` after action |
| GET | `/state` | Current `State` |

Use the `create_app()` factory from openenv:

```python
from openenv.core.env_server.app import create_app
from server.shopsense_env_environment import ShopsenseEnvironment
from models import ShopsenseAction, ShopsenseObservation

app = create_app(
    ShopsenseEnvironment,  # Pass CLASS not instance
    ShopsenseAction,
    ShopsenseObservation,
    max_concurrent_envs=4,
)
```

---

## 9. Inference Script (Issue #5 — `inference.py`)

**File:** `inference.py` — **MUST be in the ROOT of the repo, not in a subfolder**
**Branch:** `feature/inference`

### Env vars (from `.env.example`):
```bash
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN=hf_your_token_here
```

### Full working implementation:

```python
"""
inference.py — LLM Agent for ShopSense Environment
Must be in the ROOT of the repo.
"""

import os
import json
from openai import OpenAI
from shopsense_env import ShopsenseEnv, ShopsenseAction

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8000")

CATEGORIES = ["medical", "sports", "stationary", "groceries", "fruits", "generic"]

llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def predict_category(customer_id: str, purchase_history: list[str]) -> str:
    history_str = ", ".join(purchase_history) if purchase_history else "none"
    prompt = f"""You are predicting the next purchase category for a customer in a shop.

Customer ID: {customer_id}
Purchase history (most recent last): {history_str}

Valid categories: medical, sports, stationary, groceries, fruits, generic

Respond with ONLY the category name. Example: medical"""

    response = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip().lower()
    for cat in CATEGORIES:
        if cat in raw:
            return cat
    return "generic"  # fallback


def run_task(task_name: str, customer_ids: list[str], total_steps: int) -> dict:
    print(f"[START] task_id={task_name}")

    with ShopsenseEnv(base_url=ENV_URL) as env:
        result = env.reset()
        obs = result.observation

        for step_num in range(1, total_steps + 1):
            prediction = predict_category(obs.customer_id, obs.purchase_history)
            action = ShopsenseAction(
                customer_id=obs.customer_id,
                predicted_category=prediction,
            )
            result = env.step(action)
            obs = result.observation

            # EXACT FORMAT — judges parse this strictly
            print(f"[STEP] step={step_num} action={prediction} reward={result.reward} score={obs.score_so_far:.4f}")

        print(f"[END] task_id={task_name} final_score={obs.score_so_far:.4f}")

    return {"task": task_name, "final_score": obs.score_so_far}


def main():
    tasks = [
        {"name": "easy",   "customer_ids": ["C001"],                        "total_steps": 20},
        {"name": "medium", "customer_ids": ["C001", "C002"],                "total_steps": 30},
        {"name": "hard",   "customer_ids": ["C001", "C002", "C003", "C004"], "total_steps": 40},
    ]

    all_results = []
    for task in tasks:
        result = run_task(task["name"], task["customer_ids"], task["total_steps"])
        all_results.append(result)

    overall = sum(r["final_score"] for r in all_results) / len(all_results)
    print(f"\nOVERALL SCORE: {overall:.4f}")
    print(json.dumps({"results": all_results, "overall_score": overall}))


if __name__ == "__main__":
    main()
```

### ⚠️ Critical: Exact stdout log format (judges parse this strictly)
```
[START] task_id=easy
[STEP] step=1 action=medical reward=1.0 score=1.0000
[STEP] step=2 action=generic reward=0.0 score=0.5000
...
[END] task_id=easy final_score=0.8000
```

- `[START]`, `[STEP]`, `[END]` log format is **sacred** — any deviation = wrong score
- Fallback to `generic` if LLM returns a hallucinated category name

---

## 10. Deployment (Issue #6 — Dockerfile + README)

**Branch:** `feature/deployment`

### Full `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### `requirements.txt` to create:
```
openenv
fastapi
uvicorn[standard]
pydantic>=2.0
openai
python-dotenv
```

### `openenv.yaml` (already exists — DO NOT CHANGE):
```yaml
spec_version: 1
name: shopsense_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### Deploy command:
```bash
openenv push --repo-id Viscous106/shopsense-env
```

### HF Spaces env vars (set in Space Settings → Variables):
```
API_BASE_URL = https://api-inference.huggingface.co/v1
MODEL_NAME   = meta-llama/Llama-3.1-8B-Instruct
HF_TOKEN     = hf_your_token_here
```

---

## 11. File Structure (Final Target)

```
shopsense-env/
├── AGENTS.md                          ← You are here
├── README.md
├── openenv.yaml                       ← DO NOT CHANGE
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── inference.py                       ← LLM agent (MUST be in root)
├── models.py                          ← ShopsenseAction, ShopsenseObservation ✅
├── data_gen.py                        ← Customer distributions + sampling
├── reward.py                          ← compute_reward()
├── client.py                          ← DO NOT CHANGE
├── __init__.py                        ← DO NOT CHANGE
├── tasks/
│   ├── __init__.py
│   ├── task_easy.py
│   ├── task_medium.py
│   └── task_hard.py
└── server/
    ├── __init__.py
    ├── app.py                         ← FastAPI app
    └── shopsense_env_environment.py   ← ShopsenseEnvironment class
```

---

## 12. Issue → Branch → Owner Map

| GitHub Issue | Title | Branch | Owner | Status |
|---|---|---|---|---|
| #2 | Data Models (`models.py`) | `feature/models` | **Yash** | 🔄 PR open |
| #8 (GH) | Customer Distributions (`data_gen.py`) | `feature/data-gen` | **Teammate** | ⬜ Not started |
| #9 (GH) | Core Env Logic (`reward.py` + `env.py`) | `feature/core-env-v2` | **Yash** | ⬜ Not started |
| #3 | Task Definitions | `feature/tasks` | **Teammate** | ⬜ Not started |
| #4 | FastAPI Server | `feature/api-server` | **Yash** | ⬜ Not started |
| #5 | `inference.py` | `feature/inference` | **Yash** | ⬜ Not started |
| #6 | Deployment (Dockerfile etc.) | `feature/deployment` | **Teammate** | ⬜ Not started |

---

## 13. Hard Rules — Never Break These

1. **Never rename** `ShopsenseAction`, `ShopsenseObservation`, `ShopsenseEnvironment`
2. **Never edit** `client.py`, `__init__.py`, `openenv.yaml` unless explicitly told to
3. **`inference.py` must live in the repo root** — judges look there
4. **GET `/` must return HTTP 200** — judges auto-ping it; failure = disqualification
5. **Always inherit from openenv base classes** — never use plain `BaseModel` or `dict`
6. **Branch from `main`** — never commit directly to `main`
7. **One issue = one branch = one PR** — don't mix multiple issues in one branch
8. **`uv` is the package manager** — do not run `pip install` directly; use `uv add <pkg>`
9. **6 categories only**: `medical`, `sports`, `stationary`, `groceries`, `fruits`, `generic`
10. **Log format is sacred**: `[START]`, `[STEP]`, `[END]` — exact field names, exact order
