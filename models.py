# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ShopSense Environment.

The ShopSense environment simulates a shopkeeper's assistant that predicts
what individual customers will buy next, based on their purchase history.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

# All valid purchase categories
CATEGORIES = ["medical", "sports", "stationary", "groceries", "fruits", "generic"]


class ShopsenseAction(Action):
    """
    Agent's prediction for the next customer purchase.

    The agent observes the customer's purchase history and predicts
    which category they will buy next.
    """

    customer_id: str = Field(
        ...,
        description="ID of the customer being observed (e.g. 'C001')",
    )
    predicted_category: str = Field(
        ...,
        description="Predicted next purchase. One of: medical, sports, stationary, groceries, fruits, generic",
    )


class ShopsenseObservation(Observation):
    """
    What the agent sees after taking an action.

    Inherits from Observation base class which provides:
      - reward (float)  : 0.0 or 1.0 for this step
      - done (bool)     : True when episode is complete
      - metadata (dict) : Any extra info
    """

    customer_id: str = Field(
        default="",
        description="Customer ID for this episode",
    )
    purchase_history: list[str] = Field(
        default_factory=list,
        description="All purchases seen so far, including 5-item warmup history",
    )
    actual_category: str = Field(
        default="",
        description="Ground truth category revealed after the agent's prediction",
    )
    score_so_far: float = Field(
        default=0.0,
        description="Normalized running score: correct_predictions / steps_taken [0.0, 1.0]",
    )
    step: int = Field(
        default=0,
        description="Current step number (1-indexed after first step)",
    )
    total_steps: int = Field(
        default=0,
        description="Total number of steps in this episode",
    )
