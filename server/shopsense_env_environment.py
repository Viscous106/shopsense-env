# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ShopSense Environment Implementation.

A reinforcement learning environment where an LLM agent predicts customer
purchase categories based on their buying history. The environment samples
actual purchases from each customer's latent probability distribution and
rewards correct predictions.
"""

import random
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ShopsenseAction, ShopsenseObservation
    from ..data_gen import (
        sample_purchase,
        generate_warmup_history,
        get_all_customer_ids,
    )
    from ..reward import compute_reward, normalize_score
except ImportError:
    from models import ShopsenseAction, ShopsenseObservation
    from data_gen import (
        sample_purchase,
        generate_warmup_history,
        get_all_customer_ids,
    )
    from reward import compute_reward, normalize_score


_DEFAULT_TOTAL_STEPS = 20
_DEFAULT_WARMUP_COUNT = 10


class ShopsenseEnvironment(Environment):
    """
    RL environment for customer purchase prediction.

    On reset(), a customer is randomly selected from the allowed pool and a
    warmup purchase history is generated. On each step(), the environment
    samples the customer's actual next purchase, compares it with the agent's
    prediction, and returns a binary reward (1.0 correct, 0.0 wrong).

    Task difficulty is controlled via reset() kwargs:
      - customer_ids: pool of customers to sample from
      - total_steps:  episode length
      - warmup_count: number of warmup purchases shown before evaluation
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._customer_id: str = ""
        self._purchase_history: list[str] = []
        self._total_steps: int = _DEFAULT_TOTAL_STEPS
        self._correct: int = 0
        self._step_num: int = 0
        self._done: bool = False
        self._rewards: list[float] = []
        self._rng: random.Random = random.Random()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> ShopsenseObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed:        Optional RNG seed for reproducibility.
            episode_id:  Optional episode identifier.
            **kwargs:
                customer_ids (list[str]): Pool of customer IDs to choose from.
                    Defaults to all customers in data.json.
                total_steps (int): Number of prediction steps per episode.
                    Defaults to 20.
                warmup_count (int): Number of warmup purchases to generate.
                    Defaults to 10.

        Returns:
            Initial ShopsenseObservation with warmup history and no prediction yet.
        """
        customer_ids = kwargs.get("customer_ids", get_all_customer_ids())
        total_steps = kwargs.get("total_steps", _DEFAULT_TOTAL_STEPS)
        warmup_count = kwargs.get("warmup_count", _DEFAULT_WARMUP_COUNT)

        if seed is not None:
            self._rng = random.Random(seed)

        self._customer_id = self._rng.choice(customer_ids)
        self._total_steps = total_steps
        self._purchase_history = generate_warmup_history(
            self._customer_id, n=warmup_count, rng=self._rng
        )
        self._correct = 0
        self._step_num = 0
        self._done = False
        self._rewards = []

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        return ShopsenseObservation(
            customer_id=self._customer_id,
            purchase_history=list(self._purchase_history),
            actual_category="",
            score_so_far=0.0,
            step=0,
            total_steps=self._total_steps,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: ShopsenseAction,  # type: ignore[override]
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ShopsenseObservation:
        """
        Execute one prediction step.

        The environment samples the customer's actual purchase, compares it
        with the agent's prediction, and returns the result.

        Args:
            action: ShopsenseAction with customer_id and predicted_category.

        Returns:
            ShopsenseObservation with the revealed actual_category, updated
            purchase_history, running score, and done flag.
        """
        if not self._customer_id:
            return ShopsenseObservation(
                done=True,
                reward=0.0,
                metadata={"error": "No episode in progress. Call reset() first."},
            )

        if self._done:
            return ShopsenseObservation(
                customer_id=self._customer_id,
                purchase_history=list(self._purchase_history),
                actual_category="",
                score_so_far=normalize_score(self._correct, self._step_num),
                step=self._step_num,
                total_steps=self._total_steps,
                done=True,
                reward=0.0,
                metadata={"error": "Episode already finished. Call reset()."},
            )

        actual = sample_purchase(self._customer_id, rng=self._rng)
        reward = compute_reward(action.predicted_category, actual)

        self._step_num += 1
        self._correct += 1 if reward >= 0.5 else 0
        self._purchase_history.append(actual)
        self._rewards.append(reward)
        self._done = self._step_num >= self._total_steps

        score = normalize_score(self._correct, self._step_num)
        self._state.step_count = self._step_num

        return ShopsenseObservation(
            customer_id=self._customer_id,
            purchase_history=list(self._purchase_history),
            actual_category=actual,
            score_so_far=score,
            step=self._step_num,
            total_steps=self._total_steps,
            done=self._done,
            reward=reward,
            metadata={
                "prediction": action.predicted_category,
                "correct": reward >= 0.5,
                "grader_score": score,
            },
        )

    @property
    def state(self) -> State:
        return self._state
