"""
ShopSense — Customer Purchase Data Generator.

Each customer has a fixed latent profile (e.g. Doctor, Athlete) that determines
their buying probabilities. The data is fully synthetic — purchases are sampled
from these distributions on the fly, so no external dataset is needed.

Customer profiles are loaded from data.json at import time. Each entry has:
  - name:       Human-readable person name (e.g. "Dr. Emily Chen")
  - profession: Job role (e.g. "Doctor") — useful for prompt context
  - weights:    Purchase probability across all 6 categories (sums to 1.0)
                Every category has at least 1% weight for realism.

To add or remove customers, edit data.json only — this file needs no changes.

Why C001 vs C004 is hard
-------------------------
Both are doctors but buy completely different things.
A model that just memorises "Doctor → medical" will fail on C004.
It must learn from purchase history, not from the label.
"""

import json
import pathlib
import random
from typing import Literal

# ── Load customer profiles from data.json ────────────────────────────────────
_DATA_FILE = pathlib.Path(__file__).parent / "data.json"
_RAW: dict = json.loads(_DATA_FILE.read_text(encoding="utf-8"))["customers"]

# ── Derived constants (all driven by data.json — no hardcoded counts) ────────
CUSTOMER_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    cid: info["weights"] for cid, info in _RAW.items()
}
CUSTOMER_PROFILES: dict[str, str] = {
    cid: info["name"] for cid, info in _RAW.items()
}
CUSTOMER_PROFESSIONS: dict[str, str] = {
    cid: info["profession"] for cid, info in _RAW.items()
}
CUSTOMER_IDS: tuple[str, ...] = tuple(CUSTOMER_DISTRIBUTIONS.keys())

# ── Type alias for valid categories ──────────────────────────────────────────
Category = Literal["medical", "sports", "stationary", "groceries", "fruits", "generic"]


def sample_purchase(customer_id: str, rng: random.Random | None = None) -> str:
    """
    Sample a single purchase category for the given customer.

    Uses the customer's fixed probability distribution loaded from data.json.
    If an rng instance is provided it is used (useful for seeding in tests),
    otherwise falls back to the module-level random.

    Args:
        customer_id: A valid customer ID. Call get_all_customer_ids() to see all
                     available IDs (count is driven by data.json at runtime).
        rng:         Optional seeded Random instance for reproducibility.

    Returns:
        A category string, e.g. "medical".

    Raises:
        ValueError: If customer_id is not recognised.
    """
    if customer_id not in CUSTOMER_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown customer_id '{customer_id}'. "
            f"Must be one of the {len(CUSTOMER_IDS)} customers in data.json: "
            f"{list(CUSTOMER_DISTRIBUTIONS.keys())}"
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
        customer_id: A valid customer ID. Call get_all_customer_ids() to see all
                     available IDs (count driven by data.json at runtime).
        n:           Number of warmup purchases to generate (default 10).
        rng:         Optional seeded Random instance.

    Returns:
        List of n category strings sampled from the customer's distribution.

    Raises:
        ValueError: If n < 10 (insufficient context for the LLM).
    """
    if n < 10:
        raise ValueError(
            f"n must be >= 10 for sufficient LLM context. Got n={n}."
        )
    return [sample_purchase(customer_id, rng=rng) for _ in range(n)]


def get_all_customer_ids() -> list[str]:
    return list(CUSTOMER_IDS)


get_all_customer_ids.__doc__ = (
    f"Return all valid customer IDs loaded from data.json.\n\n"
    f"The count is determined at runtime from data.json — no hardcoding.\n"
    f"Use len(get_all_customer_ids()) or len(CUSTOMER_IDS) to get the total.\n\n"
    f"Returns:\n"
    f"    List of {len(CUSTOMER_IDS)} customer ID strings (currently in data.json).\n"
    f"    e.g. {list(CUSTOMER_IDS)}\n"
)
