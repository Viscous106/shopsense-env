"""
inference.py -- LLM Agent for ShopSense Environment
====================================================
MANDATORY: Must be in the ROOT of the repo.

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT (sacred -- judges parse this strictly):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import traceback
  
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables being set

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from shopsense_env import ShopsenseEnv, ShopsenseAction
except Exception:
    from client import ShopsenseEnv
    from models import ShopsenseAction
from tasks import TASKS, get_task

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "shopsense_env"
MAX_STEPS = 50
TEMPERATURE = 0.4
MAX_TOKENS = 10

CATEGORIES = ["medical", "sports", "stationary", "groceries", "fruits", "generic"]


def _create_llm_client():
    if OpenAI is None:
        print("[WARN] openai package not installed; using fallback policy", file=sys.stderr)
        return None
    if not API_KEY:
        print("[WARN] Missing API key (HF_TOKEN/API_KEY/OPENAI_API_KEY); using fallback policy", file=sys.stderr)
        return None

    try:
        return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[WARN] Failed to initialize LLM client: {exc}", file=sys.stderr)
        return None


llm = _create_llm_client()


def predict_category(customer_id: str, purchase_history: list[str]) -> str:
    """Ask the LLM to predict the next purchase category."""
    if llm is None:
        return "generic"

    history_str = ", ".join(purchase_history[-20:]) if purchase_history else "none"

    prompt = (
        "You are predicting the next purchase category for a customer in a shop.\n\n"
        f"Customer ID: {customer_id}\n"
        f"Purchase history (most recent last): {history_str}\n\n"
        f"Valid categories: {', '.join(CATEGORIES)}\n\n"
        "Respond with ONLY the category name, nothing else. Example: medical"
    )

    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        raw = response.choices[0].message.content.strip().lower()
        print(f"[DEBUG] customer={customer_id} raw='{raw}'", file=sys.stderr)
        # Exact word match first (prevents 'medical' matching inside longer phrases)
        for cat in CATEGORIES:
            if raw == cat:
                return cat
        # Fallback: substring match
        for cat in CATEGORIES:
            if cat in raw:
                return cat
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}", file=sys.stderr)
    return "generic"  # fallback


def run_task(task_name: str) -> dict:
    """Run a single task episode and emit structured logs."""
    task = get_task(task_name)

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")

    rewards: list[float] = []
    steps_taken = 0
    success = False
    last_error = None

    try:
        with ShopsenseEnv(base_url=ENV_URL).sync() as env:
            result = env.reset(
                customer_ids=task.customer_ids,
                total_steps=task.total_steps,
                warmup_count=task.warmup_count,
            )
            obs = result.observation

            for step_num in range(1, task.total_steps + 1):
                prediction = predict_category(
                    obs.customer_id, obs.purchase_history
                )
                action = ShopsenseAction(
                    customer_id=obs.customer_id,
                    predicted_category=prediction,
                )

                result = env.step(action)
                obs = result.observation
                r = result.reward if result.reward is not None else 0.0
                rewards.append(r)
                steps_taken = step_num
                done = result.done
                error_str = obs.metadata.get("error") if obs.metadata else None

                print(
                    f"[STEP] step={step_num} "
                    f"action={prediction} "
                    f"reward={r:.2f} "
                    f"done={'true' if done else 'false'} "
                    f"error={error_str or 'null'}"
                )

                if done:
                    break

            success = True

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={steps_taken} "
        f"rewards={rewards_str}"
    )

    return {
        "task": task_name,
        "steps": steps_taken,
        "rewards": rewards,
        "success": success,
        "error": last_error,
    }


def main():
    task_names = list(TASKS.keys())
    all_results = []

    for name in task_names:
        result = run_task(name)
        all_results.append(result)

    print(file=sys.stderr)
    for r in all_results:
        avg = sum(r["rewards"]) / len(r["rewards"]) if r["rewards"] else 0.0
        print(f"  {r['task']:>8s}: score={avg:.2f}  steps={r['steps']}", file=sys.stderr)


if __name__ == "__main__":
    main()
