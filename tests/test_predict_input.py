"""Regression tests for predict-time input preparation.

``ExplainerPFN.forward`` routes input through ``_prepare_predict_input``, which
applies the same dtype/text handling as ``fit``. The notebook 6 training loop
uses an instance that was never ``fit`` itself (it borrows pre-fitted
``executor_`` objects from a throwaway ``ExplainerPFN.fit`` call), so the
fit-time encoding attributes are absent. ``_prepare_predict_input`` must fall
back gracefully instead of raising ``AttributeError``/``KeyError``.
"""

import numpy as np
import pandas as pd

from explainerpfn.base import ExplainerPFN


def test_prepare_predict_input_on_unfitted_instance_ndarray():
    xai = ExplainerPFN()
    X = np.random.default_rng(0).standard_normal((5, 3)).astype(np.float32)
    out = xai._prepare_predict_input(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (5, 3)
    assert np.isfinite(out).all()


def test_prepare_predict_input_on_unfitted_instance_dataframe():
    xai = ExplainerPFN()
    X = pd.DataFrame(np.random.default_rng(1).standard_normal((7, 4)))
    out = xai._prepare_predict_input(X)
    assert isinstance(out, np.ndarray)
    assert out.shape == (7, 4)
