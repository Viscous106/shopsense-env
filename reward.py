"""
ShopSense — Reward Function and Score Normalization.

Reward design
-------------
- Binary per-step reward: 1.0 if the agent's prediction matches the ground truth,
  0.0 otherwise.
- Normalized score: correct_predictions / total_steps_taken → clamped to (0.0, 1.0)
  exclusive (never exactly 0.0 or 1.0) as required by the evaluation platform.
- Trivial random baseline: ~1/6 ≈ 0.167 (6 categories, uniform guess).
- A well-calibrated LLM agent should score 0.55–0.80 depending on the task.

Why this reward is meaningful
------------------------------
- Each step contributes independently → partial credit is natural.
- Score 0.8 = perfectly learned an 80%-dominant customer. Interpretable.
- No reward shaping needed — the distribution is the signal.
"""


def compute_reward(prediction: str, ground_truth: str) -> float:
    """
    Compute the per-step reward.

    Binary: the agent gets 1.0 for a correct prediction, 0.0 otherwise.
    Both strings are lowercased and stripped before comparison so minor
    formatting differences in LLM output don't cause false negatives.

    Args:
        prediction:   Category predicted by the agent.
        ground_truth: Category actually purchased by the customer.

    Returns:
        1.0 if correct, 0.0 if incorrect.
    """
    return 0.99 if prediction.strip().lower() == ground_truth.strip().lower() else 0.01


def normalize_score(correct: int, total: int) -> float:
    """
    Normalize the running score to the open interval (0.0, 1.0) exclusive.

    Simple division: correct_predictions / total_steps_taken, clamped so the
    result is strictly greater than 0 and strictly less than 1 as required by
    the evaluation platform.

    Args:
        correct: Number of correct predictions so far.
        total:   Total number of steps taken so far.

    Returns:
        Float strictly in (0.0, 1.0) — never exactly 0.0 or 1.0.
    """
    if total == 0:
        return 0.01
    score = round(correct / total, 2)
    return max(0.01, min(0.99, score))


def expected_baseline_score(customer_id: str) -> float:
    """
    Return the theoretical maximum score for a perfect agent on a single customer.

    A perfect Bayesian agent that has learned the exact distribution will converge
    to always predicting the most probable category. This IS the theoretical ceiling
    for a deterministic agent (since it can't do better than the mode).

    Args:
        customer_id: A valid customer ID (e.g. from CUSTOMER_IDS).

    Returns:
        The probability of the most likely category (the theoretical score ceiling).
    """
    try:
        from .data_gen import CUSTOMER_DISTRIBUTIONS
    except ImportError:
        from data_gen import CUSTOMER_DISTRIBUTIONS

    dist = CUSTOMER_DISTRIBUTIONS[customer_id]
    return max(dist.values())
