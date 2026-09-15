import pytest

from meta_harness.datasets import split_tasks


def test_split_is_deterministic_and_disjoint():
    tasks = [{"input": str(i), "label": "x"} for i in range(10)]
    a_search, a_test = split_tasks(tasks, 0.7, seed=1)
    b_search, b_test = split_tasks(tasks, 0.7, seed=1)
    assert a_search == b_search and a_test == b_test
    assert len(a_search) == 7 and len(a_test) == 3
    assert not [t for t in a_search if t in a_test]


def test_split_rejects_degenerate_fractions():
    tasks = [{"input": "1", "label": "x"}, {"input": "2", "label": "x"}]
    with pytest.raises(ValueError):
        split_tasks(tasks, 0.0)
    with pytest.raises(ValueError):
        split_tasks(tasks, 1.0)


def test_split_keeps_every_task():
    tasks = [{"input": str(i), "label": "x"} for i in range(9)]
    search, test = split_tasks(tasks, 0.5, seed=3)
    assert len(search) + len(test) == 9
