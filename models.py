# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ShopSense Environment.

ShopSense simulates a shopkeeper's assistant that learns individual customer
buying patterns from purchase history and predicts the next purchase category.

Action space  : predicted_category — one of 6 string labels
Observation   : actual_category, reward, running score, full purchase history
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator


# ── Valid purchase categories ────────────────────────────────────────────────
VALID_CATEGORIES = frozenset(
    {"medical", "sports", "stationary", "groceries", "fruits", "generic"}
)


class ShopsenseAction(Action):
    """
    The agent's predicted purchase category for the current customer.

    The agent receives the customer's purchase history and must predict
    what category of item they will buy next.
    """

    predicted_category: str = Field(
        ...,
        description=(
            "Predicted purchase category. "
            "Must be one of: medical, sports, stationary, groceries, fruits, generic"
        ),
    )

    @field_validator("predicted_category", mode="before")
    @classmethod
    def validate_category(cls, v: str) -> str:
        """Lowercase + strip the prediction, then reject anything not in the valid set."""
        cleaned = v.strip().lower()
        if cleaned not in VALID_CATEGORIES:
            raise ValueError(
                f"'{v}' is not a valid category. "
                f"Must be one of: {sorted(VALID_CATEGORIES)}"
            )
        return cleaned


class ShopsenseObservation(Observation):
    """
    Observation returned after each step.

    Contains the ground truth (actual_category), the reward for this step,
    the running normalized score, and the full purchase history so far.
    The agent should use purchase_history as in-context evidence of the
    customer's buying pattern.
    """

    # Ground truth revealed after agent predicts
    actual_category: str = Field(
        default="",
        description="The category the customer actually purchased this step",
    )

    # Running performance
    score_so_far: float = Field(
        default=0.0,
        description="Normalized score so far: correct_predictions / steps_taken, in [0.0, 1.0]",
    )

    # Episode context
    step_number: int = Field(
        default=0,
        description="Current step number (1-indexed)",
    )
    total_steps: int = Field(
        default=0,
        description="Total number of steps in this episode",
    )
    customer_id: str = Field(
        default="",
        description="The current customer's ID",
    )

    # In-context learning signal — grows each step
    purchase_history: list[str] = Field(
        default_factory=list,
        description=(
            "Full purchase history seen so far for this customer. "
            "Includes warmup purchases from reset() plus all revealed actual_category values. "
            "Use this to infer the customer's buying pattern."
        ),
    )
