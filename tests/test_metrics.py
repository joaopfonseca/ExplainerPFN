"""Tests for the explanation-agreement metrics."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from explainerpfn.metrics._base import (
    _find_neighbors,
    _get_importance_mask,
    jaccard_similarity,
    kendall_similarity,
    row_wise_jaccard,
    row_wise_kendall,
)


def _scipy_kendall_similarity(a, b):
    tau = kendalltau(a, b).statistic
    if np.isnan(tau):
        return 0.5
    return (tau + 1) / 2


@pytest.mark.parametrize(
    "a,b",
    [
        ([1, 2, 2], [1, 1, 3]),
        ([1, 1, 3], [1, 1, 3]),
        ([1, 2, 3], [3, 2, 1]),
        ([1, 2, 3, 4], [4, 3, 2, 1]),
        ([0.1, 0.9, 0.4], [0.2, 0.8, 0.5]),
    ],
)
def test_kendall_similarity_matches_scipy(a, b):
    assert kendall_similarity(a, b) == pytest.approx(
        _scipy_kendall_similarity(a, b), abs=1e-9
    )


def test_kendall_similarity_constant_ranking_is_neutral():
    # Previously returned 1.0 for any ranking vs a fully tied one.
    assert kendall_similarity([5, 5, 5], [1, 2, 3]) == pytest.approx(0.5)


def test_row_wise_kendall_handles_ties():
    a = pd.Series([0.5, 0.2, 0.2])
    b = pd.Series([0.5, 0.5, 0.2])
    # Compare with the tie-aware reference computed on average ranks.
    assert row_wise_kendall(a, b) == pytest.approx(
        kendall_similarity(
            [3.0, 1.5, 1.5],  # average ranks of a
            [2.5, 2.5, 1.0],  # average ranks of b
        ),
        abs=1e-9,
    )


def test_find_neighbors_selects_closest():
    data = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], float)
    contributions = np.zeros((6, 2))
    rankings = np.arange(1, 7)
    data_neighbors, _ = _find_neighbors(data, rankings, contributions, 0, 3)
    assert sorted(data_neighbors[:, 0]) == [1.0, 2.0, 3.0]


def test_find_neighbors_fewer_candidates_than_requested():
    data = np.array([[0, 0], [1, 0]], float)
    contributions = np.zeros((2, 2))
    rankings = np.array([1, 2])
    data_neighbors, _ = _find_neighbors(data, rankings, contributions, 0, 5)
    assert data_neighbors.shape[0] == 1


def test_importance_mask_selects_minimal_set():
    masks = _get_importance_mask(np.array([0.5, 0.2, 0.3]), 0.8)
    selected = sorted(np.where(np.asarray(masks[0]))[0])
    # 0.5 + 0.3 >= 0.8 is the minimal set.
    assert selected == [0, 2]


def test_importance_mask_accepts_ndarray():
    # Used to raise AttributeError ('numpy.ndarray' has no attribute 'index').
    masks = _get_importance_mask(np.array([0.1, -0.5, 0.3, 0.05]), 0.8)
    assert len(masks) >= 1
    assert sum(np.asarray(masks[0])) >= 1


def test_importance_mask_top_k():
    masks = _get_importance_mask(np.array([0.1, 0.9, 0.4, 0.05]), 2)
    selected = sorted(int(i) for i in np.where(np.asarray(masks[0]))[0])
    # Top-2 by absolute contribution are indices 1 (0.9) and 2 (0.4).
    assert selected == [1, 2]


def test_jaccard_similarity_basic():
    assert jaccard_similarity([1, 2, 3], [2, 3, 4]) == pytest.approx(2 / 4)
    assert jaccard_similarity([1, 2], [1, 2]) == pytest.approx(1.0)


def test_row_wise_jaccard_default_n_features():
    # Used to raise TypeError when dispatched without ``n_features``.
    a = pd.Series([0.5, 0.3, 0.2])
    b = pd.Series([0.4, 0.4, 0.2])
    sim = row_wise_jaccard(a, b)
    assert 0.0 <= sim <= 1.0
