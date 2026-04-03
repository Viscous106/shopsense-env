"""
ShopSense — Customer Purchase Data Generator.

Each customer has a fixed latent profile (e.g. Doctor, Athlete) that determines
their buying probabilities. The data is fully synthetic — purchases are sampled
from these distributions on the fly, so no external dataset is needed.

Customer Profiles
-----------------
C001 — Doctor 1   : 80% medical, 20% generic
C002 — Athlete    : 65% sports, 20% groceries, 15% generic
C003 — Teacher    : 40% stationary, 40% groceries, 20% generic
C004 — Doctor 2   : 40% stationary, 40% fruits, 20% generic

Why C001 vs C004 is hard
-------------------------
Both are doctors but buy completely different things.
A model that just memorises "Doctor → medical" will fail on C004.
It must learn from purchase history, not from the label.
"""

import random
from typing import Literal

# ── Type alias for valid categories ─────────────────────────────────────────
Category = Literal["medical", "sports", "stationary", "groceries", "fruits", "generic"]

# ── All valid customer IDs ───────────────────────────────────────────────────
CUSTOMER_IDS = ("C001", "C002", "C003", "C004")

# ── Fixed probability distributions per customer ─────────────────────────────
# Keys are category names, values are probabilities (must sum to 1.0)
CUSTOMER_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "C001": {
        "medical": 0.80,
        "generic": 0.20,
    },
    "C002": {
        "sports":    0.65,
        "groceries": 0.20,
        "generic":   0.15,
    },
    "C003": {
        "stationary": 0.40,
        "groceries":  0.40,
        "generic":    0.20,
    },
    "C004": {
        "stationary": 0.40,
        "fruits":     0.40,
        "generic":    0.20,
    },
}

# ── Human-readable profile names (for logging / README) ─────────────────────
CUSTOMER_PROFILES: dict[str, str] = {
    "C001": "Doctor 1",
    "C002": "Athlete",
    "C003": "Teacher",
    "C004": "Doctor 2",
}


def sample_purchase(customer_id: str, rng: random.Random | None = None) -> str:
    """
    Sample a single purchase category for the given customer.

    Uses the customer's fixed probability distribution. If an rng instance is
    provided it is used (useful for seeding in tests), otherwise falls back to
    the module-level random.

    Args:
        customer_id: One of C001, C002, C003, C004.
        rng:         Optional seeded Random instance for reproducibility.

    Returns:
        A category string, e.g. "medical".

    Raises:
        ValueError: If customer_id is not recognised.
    """
    if customer_id not in CUSTOMER_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown customer_id '{customer_id}'. "
            f"Must be one of: {list(CUSTOMER_DISTRIBUTIONS.keys())}"
        )

    dist = CUSTOMER_DISTRIBUTIONS[customer_id]
    categories = list(dist.keys())
    weights = list(dist.values())

    _random = rng or random
    return _random.choices(categories, weights=weights, k=1)[0]


def generate_warmup_history(
    customer_id: str,
    n: int = 10,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Generate a warm-up purchase history for a customer.

    Called during reset() to give the agent initial context before it starts
    predicting. 10 purchases gives the LLM enough signal to reliably infer
    the customer's buying pattern before evaluation begins.

    Args:
        customer_id: One of C001, C002, C003, C004.
        n:           Number of warmup purchases to generate (default 10).
        rng:         Optional seeded Random instance.

    Returns:
        List of n category strings sampled from the customer's distribution.
    """
    if n < 10:
        raise ValueError(f"n must be >= 10 for sufficient context. Got n={n}.")
    return [sample_purchase(customer_id, rng=rng) for _ in range(n)]
