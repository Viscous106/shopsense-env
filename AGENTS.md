# AGENTS.md — Shared AI Context for ShopSense Env

> **READ THIS FIRST.** If you are an AI assistant helping on this project,
> read this entire file before writing a single line of code or making any
> suggestion. This is the single source of truth for the project.

---

## 1. Project Overview

**Competition:** Meta PyTorch OpenEnv Hackathon — Round 1
**Repo:** https://github.com/Viscous106/shopsense-env
**Team:** Yash (Viscous106) + Teammate
**Deadline:** Round 1 submission

### What We Are Building

A **Reinforcement Learning environment** called **ShopSense** that simulates a
shopkeeper predicting customer purchases. An LLM agent observes a customer's
purchase history and predicts what category they will buy next. The environment
scores predictions and returns a normalized reward.

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

**Status:** Scaffolded (currently still echo-template, needs ShopSense rewrite)
**File:** `models.py` (root of repo)
**Branch:** `feature/models`

### Target implementation:

```python
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

CATEGORIES = ["Medicines", "Snacks", "Electronics", "Clothing"]

class ShopsenseAction(Action):
    """Agent's prediction of the next purchase category."""
    customer_id: str = Field(..., description="ID of the customer being predicted for")
    predicted_category: str = Field(..., description="One of: Medicines, Snacks, Electronics, Clothing")

class ShopsenseObservation(Observation):
    """What the agent sees after taking an action."""
    customer_id: str = Field(default="", description="Customer ID for this episode")
    purchase_history: list[str] = Field(default_factory=list, description="All purchases so far (including warmup)")
    actual_category: str = Field(default="", description="Ground truth category revealed after prediction")
    score_so_far: float = Field(default=0.0, description="Normalized running score [0.0, 1.0]")
    step: int = Field(default=0, description="Current step number")
    total_steps: int = Field(default=0, description="Total steps in this episode")
    # NOTE: .reward (0.0 or 1.0), .done (bool), .metadata (dict) come from Observation base class
```

### What NOT to do:
- Do NOT add a `ShopSenseState` class — `State` from openenv base is sufficient
- Do NOT use plain `BaseModel` — always inherit from `Action` / `Observation`
- Do NOT rename `ShopsenseAction` / `ShopsenseObservation` — these names are used throughout `client.py` and `__init__.py`

---

## 4. Customer Profiles (Issue #8 — `data_gen.py`)

**File:** `data_gen.py` (root of repo)
**Branch:** `feature/data-gen`

### 4 Customers with Probability Distributions

```python
CUSTOMER_PROFILES = {
    "C001": {
        "name": "Doctor",
        "weights": {"Medicines": 0.70, "Snacks": 0.15, "Electronics": 0.10, "Clothing": 0.05}
    },
    "C002": {
        "name": "Student",
        "weights": {"Snacks": 0.50, "Electronics": 0.30, "Clothing": 0.15, "Medicines": 0.05}
    },
    "C003": {
        "name": "Office Worker",
        "weights": {"Clothing": 0.40, "Snacks": 0.30, "Electronics": 0.20, "Medicines": 0.10}
    },
    "C004": {
        "name": "Retiree",
        "weights": {"Medicines": 0.50, "Clothing": 0.25, "Snacks": 0.20, "Electronics": 0.05}
    },
}
```

### Functions to implement:

```python
def sample_purchase(customer_id: str) -> str:
    """Sample one category from the customer's distribution using random.choices."""

def generate_warmup_history(customer_id: str, n: int = 5) -> list[str]:
    """Return n sampled purchases to seed the episode history."""

def get_all_customer_ids() -> list[str]:
    """Return ['C001', 'C002', 'C003', 'C004']."""
```

---

## 5. Reward Logic (Issue #9 — `reward.py`)

**File:** `reward.py` (root of repo)
**Branch:** `feature/core-env-v2`

```python
def compute_reward(predicted: str, actual: str) -> float:
    """Binary reward: 1.0 if prediction matches ground truth, else 0.0."""
    return 1.0 if predicted.strip().lower() == actual.strip().lower() else 0.0
```

- Reward is always exactly **0.0 or 1.0** — never fractional
- Comparison is **case-insensitive** and **strips whitespace**
- `score_so_far = correct_count / steps_taken` (updated after every step)

---

## 6. Core Environment Logic (Issue #9 — `server/shopsense_env_environment.py`)

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
        # 1. Pick a random customer from customer_ids (or all if None)
        # 2. Generate 5-item warmup history via generate_warmup_history()
        # 3. Reset episode state (step=0, score=0, correct=0)
        # 4. Return ShopsenseObservation with purchase_history=warmup, done=False, reward=0.0

    def step(self, action: ShopsenseAction) -> ShopsenseObservation:
        # 1. Sample actual purchase: actual = sample_purchase(self._customer_id)
        # 2. Compute reward: r = compute_reward(action.predicted_category, actual)
        # 3. Update: purchase_history.append(actual), step++, correct += r
        # 4. score_so_far = correct / step
        # 5. done = (step >= total_steps)
        # 6. Return ShopsenseObservation with all fields filled

    @property
    def state(self) -> State:
        return self._state  # openenv State object
```

### Key constraints:
- `reset()` signature in openenv may need `**kwargs` — check base class before overriding
- `step()` must accept exactly `ShopsenseAction` (not generic `Action`)
- `state` is a `@property`, not a method

---

## 7. Task Definitions (Issue #3 — `tasks/`)

**Branch:** `feature/tasks`

| Task | Customers | Steps | Expected baseline |
|---|---|---|---|
| Easy | C001 only | 20 | ~0.80 |
| Medium | C001 + C002 (random each ep.) | 30 | ~0.70 |
| Hard | All 4 customers | 40 | ~0.60 |

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

The `openenv` framework provides a `create_app()` factory — use it:

```python
from openenv.core.env_server.app import create_app
from server.shopsense_env_environment import ShopsenseEnvironment
from models import ShopsenseAction, ShopsenseObservation

app = create_app(
    ShopsenseEnvironment,  # Pass CLASS not instance (for concurrent sessions)
    ShopsenseAction,
    ShopsenseObservation,
    max_concurrent_envs=4,
)
```

---

## 9. Inference Script (Issue #5 — `inference.py`)

**File:** `inference.py` — **MUST be in the ROOT of the repo, not in a subfolder**
**Branch:** `feature/inference`

### Env vars it reads:
```bash
API_BASE_URL=<LLM API endpoint>
MODEL_NAME=<e.g. meta-llama/Llama-3-8b-instruct>
HF_TOKEN=<Hugging Face token>
```

### Flow:
1. Connect to LLM via OpenAI-compatible client
2. Run Easy → Medium → Hard tasks sequentially
3. For each step: give LLM the purchase history, ask it to predict next category
4. Print structured stdout logs (judges parse these)

---

## 10. Deployment (Issue #6 — Dockerfile + README)

**Branch:** `feature/deployment`

### Dockerfile target:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### `openenv.yaml` (already exists, do not change):
```yaml
spec_version: 1
name: shopsense_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## 11. File Structure (Final Target)

```
shopsense-env/
├── AGENTS.md                          ← You are here
├── README.md                          ← Human-readable docs
├── openenv.yaml                       ← OpenEnv manifest (DO NOT CHANGE)
├── pyproject.toml                     ← Project metadata
├── requirements.txt                   ← For Docker build
├── Dockerfile                         ← HF Spaces container
├── inference.py                       ← LLM agent (MUST be in root)
├── models.py                          ← ShopsenseAction, ShopsenseObservation
├── data_gen.py                        ← Customer profiles + sampling
├── reward.py                          ← compute_reward()
├── client.py                          ← ShopsenseEnv client (DO NOT CHANGE)
├── __init__.py                        ← Module exports (DO NOT CHANGE)
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

| GitHub Issue | Title | Branch | Owner |
|---|---|---|---|
| #2 | Data Models (`models.py`) | `feature/models` | **Yash** |
| #8 (GH) | Customer Distributions (`data_gen.py`) | `feature/data-gen` | **Teammate** |
| #9 (GH) | Core Env Logic (`reward.py` + `env.py`) | `feature/core-env-v2` | **Yash** |
| #3 | Task Definitions | `feature/tasks` | **Teammate** |
| #4 | FastAPI Server | `feature/api-server` | **Yash** |
| #5 | `inference.py` | `feature/inference` | **Yash** |
| #6 | Deployment (Dockerfile etc.) | `feature/deployment` | **Teammate** |

---

## 13. Hard Rules — Never Break These

1. **Never rename** `ShopsenseAction`, `ShopsenseObservation`, `ShopsenseEnvironment` — they're referenced in `client.py` and `__init__.py`
2. **Never edit** `client.py`, `__init__.py`, `openenv.yaml` unless explicitly told to
3. **`inference.py` must live in the repo root** — judges look there
4. **GET `/` must return HTTP 200** — judges auto-ping it; failure = disqualification
5. **Always inherit from openenv base classes** — never use plain `BaseModel` or `dict`
6. **Branch from `main`** — never commit directly to `main`
7. **One issue = one branch = one PR** — don't mix multiple issues in one branch
8. **`uv` is the package manager** — do not run `pip install` directly; use `uv add <pkg>`
