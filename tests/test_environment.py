"""Tests for the core ShopsenseEnvironment — reset/step/state lifecycle."""

import random

import pytest

from models import ShopsenseAction, ShopsenseObservation, CATEGORIES
from server.shopsense_env_environment import ShopsenseEnvironment
from data_gen import CUSTOMER_IDS, get_all_customer_ids


class TestReset:
    def test_returns_observation(self):
        env = ShopsenseEnvironment()
        obs = env.reset()
        assert isinstance(obs, ShopsenseObservation)

    def test_defaults_to_all_customers(self):
        """Without customer_ids kwarg, any customer from data.json is valid."""
        env = ShopsenseEnvironment()
        obs = env.reset(seed=42)
        assert obs.customer_id in CUSTOMER_IDS

    def test_specific_customer(self):
        env = ShopsenseEnvironment()
        obs = env.reset(customer_ids=["C001"])
        assert obs.customer_id == "C001"

    def test_warmup_history_present(self):
        env = ShopsenseEnvironment()
        obs = env.reset(warmup_count=10)
        assert len(obs.purchase_history) == 10
        for cat in obs.purchase_history:
            assert cat in CATEGORIES

    def test_custom_warmup_count(self):
        env = ShopsenseEnvironment()
        obs = env.reset(warmup_count=15)
        assert len(obs.purchase_history) == 15

    def test_initial_state_clean(self):
        env = ShopsenseEnvironment()
        obs = env.reset()
        assert obs.step == 0
        assert obs.score_so_far == 0.0
        assert obs.actual_category == ""
        assert obs.done is False
        assert obs.reward == 0.0

    def test_total_steps_configurable(self):
        env = ShopsenseEnvironment()
        obs = env.reset(total_steps=30)
        assert obs.total_steps == 30

    def test_seed_reproducibility(self):
        env1 = ShopsenseEnvironment()
        obs1 = env1.reset(seed=42, customer_ids=get_all_customer_ids())

        env2 = ShopsenseEnvironment()
        obs2 = env2.reset(seed=42, customer_ids=get_all_customer_ids())

        assert obs1.customer_id == obs2.customer_id
        assert obs1.purchase_history == obs2.purchase_history

    def test_reset_clears_previous_episode(self):
        env = ShopsenseEnvironment()
        env.reset(customer_ids=["C001"], total_steps=5)

        for _ in range(5):
            env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))

        obs = env.reset(customer_ids=["C002"])
        assert obs.step == 0
        assert obs.customer_id == "C002"
        assert obs.done is False


class TestStep:
    def _make_env(self, customer_id="C001", total_steps=5, seed=42):
        env = ShopsenseEnvironment()
        env.reset(seed=seed, customer_ids=[customer_id], total_steps=total_steps)
        return env

    def test_returns_observation(self):
        env = self._make_env()
        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert isinstance(obs, ShopsenseObservation)

    def test_step_increments(self):
        env = self._make_env(total_steps=3)
        for i in range(1, 4):
            obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
            assert obs.step == i

    def test_actual_category_revealed(self):
        env = self._make_env()
        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert obs.actual_category in CATEGORIES
        assert obs.actual_category != ""

    def test_purchase_history_grows(self):
        env = self._make_env()
        initial_len = len(env.reset(seed=42, customer_ids=["C001"], total_steps=5).purchase_history)
        for i in range(3):
            obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
            assert len(obs.purchase_history) == initial_len + i + 1

    def test_reward_binary(self):
        env = self._make_env()
        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert obs.reward in (0.01, 0.99)

    def test_done_at_total_steps(self):
        env = self._make_env(total_steps=3)
        for i in range(3):
            obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))

        assert obs.done is True

    def test_not_done_before_total_steps(self):
        env = self._make_env(total_steps=5)
        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert obs.done is False

    def test_score_in_range(self):
        env = self._make_env(total_steps=10)
        for _ in range(10):
            obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert 0.0 <= obs.score_so_far <= 1.0

    def test_step_after_done_returns_done_obs(self):
        env = self._make_env(total_steps=2)
        env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))

        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert obs.done is True
        assert obs.reward == 0.0
        assert "error" in obs.metadata

    def test_metadata_contains_prediction(self):
        env = self._make_env()
        obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="sports"))
        assert obs.metadata["prediction"] == "sports"
        assert isinstance(obs.metadata["correct"], bool)

    def test_correct_prediction_gives_reward(self):
        """Run many steps; at least some correct predictions should give reward=1.0."""
        env = ShopsenseEnvironment()
        env.reset(seed=42, customer_ids=["C001"], total_steps=50)
        rewards = []
        for _ in range(50):
            obs = env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
            rewards.append(obs.reward)
        # C001 has 65% medical — expect most to be correct
        assert sum(rewards) > 20


class TestState:
    def test_state_has_episode_id(self):
        env = ShopsenseEnvironment()
        env.reset()
        assert env.state.episode_id is not None

    def test_state_step_count_tracks(self):
        env = ShopsenseEnvironment()
        env.reset(customer_ids=["C001"], total_steps=3)
        assert env.state.step_count == 0

        env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert env.state.step_count == 1

        env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        assert env.state.step_count == 2

    def test_reset_resets_step_count(self):
        env = ShopsenseEnvironment()
        env.reset(customer_ids=["C001"], total_steps=3)
        env.step(ShopsenseAction(customer_id="C001", predicted_category="medical"))
        env.reset()
        assert env.state.step_count == 0


class TestFullEpisode:
    """Integration test: run a complete episode end-to-end."""

    def test_full_episode_easy(self):
        env = ShopsenseEnvironment()
        obs = env.reset(seed=100, customer_ids=["C001"], total_steps=20, warmup_count=10)

        assert obs.step == 0
        assert len(obs.purchase_history) == 10

        rewards = []
        for step in range(1, 21):
            action = ShopsenseAction(
                customer_id=obs.customer_id,
                predicted_category="medical",  # C001's dominant category
            )
            obs = env.step(action)
            rewards.append(obs.reward)

            assert obs.step == step
            assert len(obs.purchase_history) == 10 + step

            if step < 20:
                assert obs.done is False
            else:
                assert obs.done is True

        assert obs.score_so_far == pytest.approx(sum(rewards) / 20, abs=0.02)
        # C001 medical ~65%, expect decent score
        assert obs.score_so_far > 0.3

    def test_full_episode_all_customers(self):
        """Run multiple episodes with all customers."""
        env = ShopsenseEnvironment()
        all_ids = get_all_customer_ids()

        for seed in range(5):
            obs = env.reset(seed=seed, customer_ids=all_ids, total_steps=10)
            assert obs.customer_id in all_ids
            for _ in range(10):
                obs = env.step(
                    ShopsenseAction(
                        customer_id=obs.customer_id,
                        predicted_category="generic",
                    )
                )
            assert obs.done is True
