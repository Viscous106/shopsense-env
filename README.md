---
title: ShopSense Environment Server
emoji: 🛒
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
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

```bash
# Install dependencies
uv sync --extra dev

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run tests
python -m pytest tests/ -v

# Build Docker image
docker build -t shopsense-env .

# Run container
docker run -p 8000:8000 shopsense-env
```

## Inference Script

```bash
# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_your_token
export ENV_URL=http://localhost:8000

# Run inference
python inference.py
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
