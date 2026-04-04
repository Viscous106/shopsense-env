---
title: ShopSense Environment Server
emoji: 🛒
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# ShopSense Environment

An RL environment where an LLM agent predicts customer purchase categories based on buying history. The environment simulates a shopkeeper's assistant — customers have latent purchase distributions across 6 categories, and the agent must learn each customer's pattern from their history to predict what they'll buy next.

## Why This Environment

- **Real-world task**: customer behavior prediction is a genuine retail/e-commerce challenge
- **Partial-credit reward**: each step contributes independently; score = correct / total
- **Meaningful difficulty**: easy customers have one dominant category (~65%), hard customers have flatter distributions or share professions (two doctors with opposite buying habits)
- **Dynamic data**: add or remove customers by editing `data.json` — no code changes needed

## Quick Start

```python
from shopsense_env import ShopsenseAction, ShopsenseEnv

with ShopsenseEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(customer_ids=["C001"], total_steps=20)
    obs = result.observation

    for _ in range(20):
        action = ShopsenseAction(
            customer_id=obs.customer_id,
            predicted_category="medical",  # C001's dominant category
        )
        result = env.step(action)
        obs = result.observation
        print(f"Step {obs.step}: predicted=medical, actual={obs.actual_category}, reward={result.reward}")

    print(f"Final score: {obs.score_so_far:.2f}")
```

## Action Space

**ShopsenseAction**:
| Field | Type | Description |
|---|---|---|
| `customer_id` | str | ID of the customer (e.g. `"C001"`) |
| `predicted_category` | str | One of: `medical`, `sports`, `stationary`, `groceries`, `fruits`, `generic` |

## Observation Space

**ShopsenseObservation** (inherits `done`, `reward`, `metadata` from base):
| Field | Type | Description |
|---|---|---|
| `customer_id` | str | Current customer ID |
| `purchase_history` | list[str] | All purchases so far (warmup + revealed actuals) |
| `actual_category` | str | Ground truth revealed after prediction |
| `score_so_far` | float | Running normalized score [0.0, 1.0] |
| `step` | int | Current step number (1-indexed) |
| `total_steps` | int | Total steps in this episode |

## Reward Function

- **Per-step**: binary 1.0 (correct) or 0.0 (wrong), case-insensitive
- **Episode score**: `correct_predictions / total_steps` — always in [0.0, 1.0]
- **Random baseline**: ~1/6 = 0.167 (uniform guess across 6 categories)
- **Good LLM agent**: 0.55 - 0.80 depending on task difficulty

## Tasks

Tasks are built dynamically from `data.json` — difficulty scales with customer pool size and distribution entropy.

| Task | Customers | Steps | Expected Baseline | Description |
|---|---|---|---|---|
| Easy | 1 (highest mode) | 20 | ~0.65 | Single customer with a clear dominant category |
| Medium | 2-5 (easiest subset) | 30 | ~0.50 | Multiple customers; agent must adapt per-episode |
| Hard | All (currently 10) | 40 | ~0.40 | Full pool including tricky pairs (doctors with opposite habits) |

## Customer Profiles

Loaded from `data.json` at runtime. Currently 10 customers across 6 professions. Each has a probability distribution over 6 purchase categories. To add customers, edit `data.json` — environment and tasks adapt automatically.

## Setup & Running

### Using uv (recommended for Docker / submission)

```bash
# Install dependencies
uv sync --extra dev

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Build & run Docker image
docker build -t shopsense-env .
docker run -p 8000:8000 shopsense-env
```

### Using pip + venv (local development on Windows/Mac/Linux)

Requires **Python 3.12** (project supports >=3.10).

```bash
# 1. Create and activate virtual environment
py -3.12 -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

# 2. Install project + dev dependencies
pip install -e ".[dev]"

# 3. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API key and model endpoint
```

**.env** example (using Groq):
```
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.1-8b-instant
HF_TOKEN=gsk_your_groq_or_hf_token_here
```

**.env** example (using Hugging Face):
```
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
HF_TOKEN=hf_your_token_here
```

```bash
# 4. Start the environment server (keep this running in a separate terminal)
python -m shopsense_env.server.app
# Server starts at http://localhost:8000

# 5. Run the LLM inference agent (in another terminal, with venv activated)
python inference.py

# 6. Run tests
pytest
```

**Sample inference output:**
```
[START] task=easy env=shopsense_env model=llama-3.1-8b-instant
[STEP] step=1 action=groceries reward=1.00 done=false error=null
[STEP] step=2 action=medical reward=0.00 done=false error=null
...
[END] success=true steps=20 rewards=1.00,0.00,...

    easy: score=0.3500  steps=20
  medium: score=0.3000  steps=30
    hard: score=0.6250  steps=40
```

## Deploy to Hugging Face Spaces

```bash
openenv push --repo-id Viscous106/shopsense-env
```

## Project Structure

```
shopsense-env/
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml          # Project metadata & dependencies
├── Dockerfile              # Container build
├── requirements.txt        # Pip dependencies for Docker
├── inference.py            # LLM agent inference script
├── models.py               # ShopsenseAction, ShopsenseObservation
├── data_gen.py             # Customer profiles & sampling from data.json
├── data.json               # Customer probability distributions
├── reward.py               # Reward computation & normalization
├── client.py               # ShopsenseEnv WebSocket client
├── __init__.py             # Package exports
├── tasks/
│   └── __init__.py         # TaskConfig definitions (easy/medium/hard)
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI application
│   └── shopsense_env_environment.py  # Core environment logic
└── tests/
    ├── conftest.py
    ├── test_data_gen.py
    ├── test_reward.py
    ├── test_environment.py
    ├── test_models.py
    └── test_tasks.py
```
