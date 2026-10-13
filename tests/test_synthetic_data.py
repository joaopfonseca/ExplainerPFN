"""Tests for the degenerate-dataset guard in the synthetic data iterator.

Datasets whose exact do-Shapley labels are constant (typically all zero) make
the trainer's ``shap / feature_exp_std`` normalisation divide by zero. They must
be filtered out before they reach the trainer.
"""

import numpy as np
import pandas as pd
import pytest

from explainerpfn.train.synthetic_data import (
    _MIN_SHAP_STD,
    TrainingBatchIterator,
    _usable_feature_exp_std,
)


def test_usable_feature_exp_std_accepts_typical_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = rng.normal(size=50)
    shap = rng.normal(size=(50, 3))
    std = _usable_feature_exp_std(X, y, shap)
    assert std is not None
    assert std == pytest.approx(float(shap.std()))


def test_usable_feature_exp_std_rejects_all_zero_labels():
    # The real failure mode: a dataset whose exact Shapley labels are all zero.
    X = np.ones((10, 3))
    y = np.ones(10)
    shap = np.zeros((10, 3))
    assert _usable_feature_exp_std(X, y, shap) is None


def test_usable_feature_exp_std_rejects_tiny_variance():
    X = np.ones((10, 3))
    y = np.ones(10)
    shap = np.full((10, 3), _MIN_SHAP_STD * 0.5)
    assert _usable_feature_exp_std(X, y, shap) is None


def test_usable_feature_exp_std_rejects_non_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=10)
    shap = rng.normal(size=(10, 3))

    shap_nan = shap.copy()
    shap_nan[0, 0] = np.nan
    assert _usable_feature_exp_std(X, y, shap_nan) is None

    X_inf = X.copy()
    X_inf[0, 0] = np.inf
    assert _usable_feature_exp_std(X_inf, y, shap) is None

    y_nan = y.copy()
    y_nan[0] = np.nan
    assert _usable_feature_exp_std(X, y_nan, shap) is None


def test_preprocess_skips_degenerate_dataset():
    dataset = {
        "X": pd.DataFrame(np.ones((10, 3)), columns=["ind_0", "ind_1", "ind_2"]),
        "y": pd.Series(np.ones(10)),
        "shap": pd.DataFrame(
            np.zeros((10, 3)), columns=["ind_0", "ind_1", "ind_2"]
        ),
        "params": {"seed": 1, "index": 2469, "n_features": 3, "n_samples": 10},
    }
    iterator = TrainingBatchIterator()
    # Returns None (before the throwaway fit) so the worker can skip it.
    assert iterator._preprocess(dataset) is None
