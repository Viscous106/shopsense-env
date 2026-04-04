"""Tests for task definitions — dynamic configuration from data.json."""

import pytest

from tasks import TASKS, TaskConfig, build_tasks, get_task
from data_gen import CUSTOMER_IDS, get_all_customer_ids


class TestTaskConfig:
    def test_three_tasks_defined(self):
        assert len(TASKS) == 3
        assert set(TASKS.keys()) == {"easy", "medium", "hard"}

    def test_all_configs_are_taskconfig(self):
        for task in TASKS.values():
            assert isinstance(task, TaskConfig)

    def test_easy_has_single_customer(self):
        easy = get_task("easy")
        assert len(easy.customer_ids) == 1
        assert easy.customer_ids[0] in CUSTOMER_IDS

    def test_medium_has_multiple_customers(self):
        medium = get_task("medium")
        assert len(medium.customer_ids) >= 2
        for cid in medium.customer_ids:
            assert cid in CUSTOMER_IDS

    def test_hard_has_all_customers(self):
        hard = get_task("hard")
        assert set(hard.customer_ids) == set(get_all_customer_ids())

    def test_step_counts_increase(self):
        assert get_task("easy").total_steps < get_task("medium").total_steps
        assert get_task("medium").total_steps < get_task("hard").total_steps

    def test_easy_steps(self):
        assert get_task("easy").total_steps == 20

    def test_medium_steps(self):
        assert get_task("medium").total_steps == 30

    def test_hard_steps(self):
        assert get_task("hard").total_steps == 40

    def test_warmup_count_default(self):
        for task in TASKS.values():
            assert task.warmup_count == 10

    def test_descriptions_non_empty(self):
        for task in TASKS.values():
            assert task.description

    def test_get_task_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown task"):
            get_task("impossible")


class TestDynamicBehavior:
    """Verify tasks adapt to data.json content."""

    def test_build_tasks_returns_fresh_dict(self):
        t1 = build_tasks()
        t2 = build_tasks()
        assert t1 is not t2

    def test_easy_customer_has_highest_mode(self):
        """Easy task should pick the customer with the most dominant category."""
        from data_gen import CUSTOMER_DISTRIBUTIONS

        easy = get_task("easy")
        easy_cid = easy.customer_ids[0]
        easy_mode = max(CUSTOMER_DISTRIBUTIONS[easy_cid].values())

        for cid in CUSTOMER_IDS:
            cid_mode = max(CUSTOMER_DISTRIBUTIONS[cid].values())
            assert easy_mode >= cid_mode

    def test_hard_customer_count_matches_data(self):
        hard = get_task("hard")
        assert len(hard.customer_ids) == len(get_all_customer_ids())
