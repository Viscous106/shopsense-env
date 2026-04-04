"""Tests for Pydantic data models — ShopsenseAction and ShopsenseObservation."""

import pytest
from pydantic import ValidationError

from models import CATEGORIES, ShopsenseAction, ShopsenseObservation


class TestCategories:
    def test_six_categories(self):
        assert len(CATEGORIES) == 6

    def test_expected_categories(self):
        expected = {"medical", "sports", "stationary", "groceries", "fruits", "generic"}
        assert set(CATEGORIES) == expected


class TestShopsenseAction:
    def test_valid_action(self):
        action = ShopsenseAction(customer_id="C001", predicted_category="medical")
        assert action.customer_id == "C001"
        assert action.predicted_category == "medical"

    def test_customer_id_required(self):
        with pytest.raises(ValidationError):
            ShopsenseAction(predicted_category="medical")

    def test_predicted_category_required(self):
        with pytest.raises(ValidationError):
            ShopsenseAction(customer_id="C001")

    def test_serialization_roundtrip(self):
        action = ShopsenseAction(customer_id="C001", predicted_category="sports")
        data = action.model_dump()
        restored = ShopsenseAction(**data)
        assert restored.customer_id == action.customer_id
        assert restored.predicted_category == action.predicted_category


class TestShopsenseObservation:
    def test_defaults(self):
        obs = ShopsenseObservation()
        assert obs.customer_id == ""
        assert obs.purchase_history == []
        assert obs.actual_category == ""
        assert obs.score_so_far == 0.0
        assert obs.step == 0
        assert obs.total_steps == 0
        assert obs.done is False
        assert obs.reward is None or obs.reward == 0.0

    def test_full_observation(self):
        obs = ShopsenseObservation(
            customer_id="C001",
            purchase_history=["medical", "generic", "medical"],
            actual_category="medical",
            score_so_far=0.75,
            step=4,
            total_steps=20,
            done=False,
            reward=1.0,
        )
        assert obs.customer_id == "C001"
        assert len(obs.purchase_history) == 3
        assert obs.score_so_far == 0.75

    def test_done_flag(self):
        obs = ShopsenseObservation(done=True)
        assert obs.done is True

    def test_serialization_roundtrip(self):
        obs = ShopsenseObservation(
            customer_id="C002",
            purchase_history=["sports", "fruits"],
            actual_category="sports",
            score_so_far=0.5,
            step=2,
            total_steps=10,
            done=False,
            reward=1.0,
        )
        data = obs.model_dump()
        restored = ShopsenseObservation(**data)
        assert restored.customer_id == obs.customer_id
        assert restored.purchase_history == obs.purchase_history
