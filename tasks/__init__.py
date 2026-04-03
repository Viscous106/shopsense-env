"""
ShopSense Task Definitions.

Tasks configure episode difficulty by controlling which customers appear
and how many prediction steps the agent must make. All customer pools are
derived dynamically from data.json — no hardcoded IDs.

Usage:
    from tasks import TASKS, get_task
    easy = get_task("easy")
    print(easy.customer_ids, easy.total_steps)
"""

from dataclasses import dataclass, field

try:
    from ..data_gen import get_all_customer_ids, CUSTOMER_DISTRIBUTIONS
except ImportError:
    from data_gen import get_all_customer_ids, CUSTOMER_DISTRIBUTIONS


@dataclass(frozen=True)
class TaskConfig:
    """Immutable task configuration."""

    name: str
    customer_ids: list[str] = field(default_factory=list)
    total_steps: int = 20
    warmup_count: int = 10
    description: str = ""


def _sorted_by_difficulty(customer_ids: list[str]) -> list[str]:
    """Sort customers from easiest (highest mode) to hardest (most uniform)."""
    return sorted(
        customer_ids,
        key=lambda cid: max(CUSTOMER_DISTRIBUTIONS[cid].values()),
        reverse=True,
    )


def build_tasks() -> dict[str, TaskConfig]:
    """
    Build task configs dynamically from the current data.json.

    - Easy:   The single easiest customer (highest mode probability), 20 steps.
    - Medium: Top ~30% easiest customers (min 2, max 5), 30 steps.
    - Hard:   ALL customers, 40 steps.
    """
    all_ids = get_all_customer_ids()
    ranked = _sorted_by_difficulty(all_ids)
    n = len(all_ids)

    medium_count = max(2, min(5, n // 3))

    return {
        "easy": TaskConfig(
            name="easy",
            customer_ids=[ranked[0]],
            total_steps=20,
            description=(
                f"Single easiest customer ({ranked[0]}). "
                f"Dominant category has ~{max(CUSTOMER_DISTRIBUTIONS[ranked[0]].values()):.0%} probability."
            ),
        ),
        "medium": TaskConfig(
            name="medium",
            customer_ids=ranked[:medium_count],
            total_steps=30,
            description=(
                f"{medium_count} customers with clearest buying patterns. "
                f"Agent must adapt to different profiles across episodes."
            ),
        ),
        "hard": TaskConfig(
            name="hard",
            customer_ids=list(all_ids),
            total_steps=40,
            description=(
                f"All {n} customers including tricky pairs (e.g. doctors with "
                f"opposite buying habits). Requires learning from history, not labels."
            ),
        ),
    }


TASKS = build_tasks()


def get_task(name: str) -> TaskConfig:
    """Get a task config by name. Raises KeyError if not found."""
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASKS.keys())}")
    return TASKS[name]
