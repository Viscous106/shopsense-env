"""Tests for reward module — compute_reward and normalize_score."""

import pytest

from reward import compute_reward, expected_baseline_score, normalize_score
from data_gen import CUSTOMER_IDS


class TestComputeReward:
    def test_exact_match(self):
        assert compute_reward("medical", "medical") == 0.9999

    def test_mismatch(self):
        assert compute_reward("medical", "sports") == 0.0001

    def test_case_insensitive(self):
        assert compute_reward("Medical", "medical") == 0.9999
        assert compute_reward("SPORTS", "sports") == 0.9999

    def test_whitespace_stripped(self):
        assert compute_reward("  medical  ", "medical") == 0.9999
        assert compute_reward("medical", "  medical\n") == 0.9999

    def test_all_categories_match_self(self):
        from models import CATEGORIES

        for cat in CATEGORIES:
            assert compute_reward(cat, cat) == 0.9999

    def test_empty_strings(self):
        assert compute_reward("", "") == 0.9999
        assert compute_reward("", "medical") == 0.0001


class TestNormalizeScore:
    def test_zero_steps(self):
        assert normalize_score(0, 0) == 0.0001

    def test_perfect_score(self):
        assert normalize_score(10, 10) == 0.9999

    def test_half_score(self):
        assert normalize_score(5, 10) == 0.5

    def test_one_correct(self):
        assert normalize_score(1, 3) == pytest.approx(0.3333, abs=1e-3)

    def test_result_in_range(self):
        for correct in range(21):
            for total in range(1, 21):
                if correct <= total:
                    score = normalize_score(correct, total)
                    assert 0.0 < score < 1.0


class TestExpectedBaselineScore:
    def test_returns_max_weight(self):
        from data_gen import CUSTOMER_DISTRIBUTIONS

        for cid in CUSTOMER_IDS:
            expected = max(CUSTOMER_DISTRIBUTIONS[cid].values())
            assert expected_baseline_score(cid) == expected

    def test_all_customers_have_baseline(self):
        for cid in CUSTOMER_IDS:
            score = expected_baseline_score(cid)
            assert 0.0 < score <= 1.0
