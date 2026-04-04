"""Tests for data_gen module — customer profiles, sampling, and warmup."""

import random

import pytest

from data_gen import (
    CUSTOMER_DISTRIBUTIONS,
    CUSTOMER_IDS,
    CUSTOMER_PROFILES,
    CUSTOMER_PROFESSIONS,
    generate_warmup_history,
    get_all_customer_ids,
    sample_purchase,
)
from models import CATEGORIES


class TestDataLoading:
    """Verify data.json is loaded correctly and consistently."""

    def test_customer_ids_not_empty(self):
        assert len(CUSTOMER_IDS) > 0

    def test_all_dicts_have_same_keys(self):
        ids = set(CUSTOMER_IDS)
        assert set(CUSTOMER_DISTRIBUTIONS.keys()) == ids
        assert set(CUSTOMER_PROFILES.keys()) == ids
        assert set(CUSTOMER_PROFESSIONS.keys()) == ids

    def test_get_all_customer_ids_matches_constant(self):
        assert get_all_customer_ids() == list(CUSTOMER_IDS)

    def test_weights_sum_to_one(self):
        for cid, weights in CUSTOMER_DISTRIBUTIONS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{cid} weights sum to {total}"

    def test_all_categories_present_in_weights(self):
        for cid, weights in CUSTOMER_DISTRIBUTIONS.items():
            for cat in CATEGORIES:
                assert cat in weights, f"{cid} missing category '{cat}'"

    def test_all_weights_positive(self):
        for cid, weights in CUSTOMER_DISTRIBUTIONS.items():
            for cat, w in weights.items():
                assert w > 0, f"{cid} has zero weight for '{cat}'"

    def test_dynamic_count(self):
        """Customer count should match data.json, not be hardcoded."""
        count = len(get_all_customer_ids())
        assert count == len(CUSTOMER_IDS)
        assert count >= 1


class TestSamplePurchase:
    def test_returns_valid_category(self):
        rng = random.Random(42)
        for cid in CUSTOMER_IDS:
            cat = sample_purchase(cid, rng=rng)
            assert cat in CATEGORIES

    def test_invalid_customer_raises(self):
        with pytest.raises(ValueError, match="Unknown customer_id"):
            sample_purchase("INVALID")

    def test_seeded_reproducibility(self):
        results_a = [sample_purchase("C001", rng=random.Random(99)) for _ in range(5)]
        results_b = [sample_purchase("C001", rng=random.Random(99)) for _ in range(5)]
        assert results_a == results_b

    def test_distribution_roughly_matches(self):
        """Over many samples, frequencies should approximate the weights."""
        rng = random.Random(123)
        cid = CUSTOMER_IDS[0]
        n = 5000
        counts: dict[str, int] = {}
        for _ in range(n):
            cat = sample_purchase(cid, rng=rng)
            counts[cat] = counts.get(cat, 0) + 1

        dist = CUSTOMER_DISTRIBUTIONS[cid]
        for cat, expected_prob in dist.items():
            actual_freq = counts.get(cat, 0) / n
            assert abs(actual_freq - expected_prob) < 0.05, (
                f"{cid}/{cat}: expected ~{expected_prob:.2f}, got {actual_freq:.2f}"
            )


class TestWarmupHistory:
    def test_default_length(self):
        history = generate_warmup_history("C001")
        assert len(history) == 10

    def test_custom_length(self):
        history = generate_warmup_history("C001", n=15)
        assert len(history) == 15

    def test_minimum_enforced(self):
        with pytest.raises(ValueError, match="n must be >= 10"):
            generate_warmup_history("C001", n=5)

    def test_all_valid_categories(self):
        history = generate_warmup_history("C001", n=20, rng=random.Random(42))
        for cat in history:
            assert cat in CATEGORIES

    def test_seeded_reproducibility(self):
        a = generate_warmup_history("C001", rng=random.Random(7))
        b = generate_warmup_history("C001", rng=random.Random(7))
        assert a == b
