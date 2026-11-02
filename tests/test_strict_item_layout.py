"""Tests for the exact one-item-per-feature (strict) preprocessing layout.

The single-pass attribution head consumes one encoder-item state per raw
feature, so the preprocessing must neither add nor drop columns and must not
reorder them. These tests lock that contract plus the order-preserving text/NA
encoding it depends on.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from tabpfn.utils import _fix_dtypes, _get_ordinal_encoder

from explainerpfn.model.encoders import normalize_data, torch_nanstd

from explainerpfn.preprocessing import (
    EnsembleConfig,
    default_explanation_preprocessor_configs,
    strict_item_preprocessor_configs,
)
from explainerpfn.utils import process_text_na_dataframe


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x))


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(_rank(a), _rank(b))[0, 1])


def _mixed_dataset(n: int = 96, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(size=n),
            rng.normal(size=n),
            rng.integers(0, 3, size=n).astype(float),
            rng.normal(size=n),
            np.full(n, 7.0),
        ]
    )
    return X, [2]


def _pipeline(*, strict: bool, random_state: int = 0):
    configs = EnsembleConfig.generate_for_explainability(
        n=1,
        subsample_size=None,
        max_index=96,
        add_fingerprint_feature=not strict,
        polynomial_features="no",
        feature_shift_decoder=None,
        preprocessor_configs=(
            strict_item_preprocessor_configs()
            if strict
            else default_explanation_preprocessor_configs()
        ),
        target_transforms=[None],
        random_state=random_state,
        remove_constant_features=not strict,
    )
    return configs[0].to_pipeline(random_state=random_state)


def test_strict_pipeline_has_no_extra_steps():
    pipe = _pipeline(strict=True)
    names = [type(step).__name__ for step in pipe.steps]
    assert names == ["ReshapeFeatureDistributionsStep", "EncodeCategoricalFeaturesStep"]
    reshape = pipe.steps[0]
    assert reshape.transform_name == "robust"
    assert reshape.global_transformer_name is None
    assert reshape.append_to_original is False
    assert pipe.steps[1].categorical_transform_name == "ordinal"


def test_legacy_pipeline_keeps_extra_steps():
    pipe = _pipeline(strict=False)
    names = [type(step).__name__ for step in pipe.steps]
    assert "RemoveConstantFeaturesStep" in names
    assert "AddFingerprintFeaturesStep" in names
    reshape = next(
        step
        for step in pipe.steps
        if type(step).__name__ == "ReshapeFeatureDistributionsStep"
    )
    assert reshape.global_transformer_name == "svd"


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_strict_pipeline_keeps_width_and_order(seed):
    X, cat_ix = _mixed_dataset()
    pipe = _pipeline(strict=True, random_state=seed)
    Xt, cats = pipe.fit_transform(X, cat_ix)

    # Same width: no appended SVD components, no fingerprint, and the constant
    # column is kept.
    assert Xt.shape == X.shape
    assert cats == cat_ix

    # Every column is a monotone recode of the raw column at the same index.
    for i in range(X.shape[1]):
        assert _rank_corr(Xt[:, i], X[:, i]) > 0.999

    # And the result is seed-independent.
    Xt2, _ = _pipeline(strict=True, random_state=seed + 1).fit_transform(X, cat_ix)
    assert np.array_equal(Xt, Xt2)

    # The transform path (query rows) matches the fit path.
    Xt_test, _ = pipe.transform(X)
    assert np.array_equal(Xt_test, Xt)


def test_process_text_na_dataframe_preserves_column_order():
    rng = np.random.default_rng(0)
    n = 64
    df = pd.DataFrame(
        {
            "num0": rng.normal(size=n),
            "num1": rng.normal(size=n),
            "cat0": rng.integers(0, 4, size=n).astype(float),
            "tex0": rng.choice(["a", "b", "c", None], size=n),
        }
    )
    X = _fix_dtypes(df, cat_indices=[2])
    encoded = process_text_na_dataframe(
        X, ord_encoder=_get_ordinal_encoder(), fit_encoder=True
    )
    assert encoded.shape == (n, 4)

    for i in range(3):
        assert _rank_corr(encoded[:, i], df.iloc[:, i].to_numpy()) > 0.999

    # String column is ordinal-encoded in place, with missing values kept as NaN.
    missing = df["tex0"].isna().to_numpy()
    assert np.isnan(encoded[missing, 3]).all()
    assert np.isfinite(encoded[~missing, 3]).all()


def test_process_text_na_dataframe_matches_encoder_on_transform():
    rng = np.random.default_rng(1)
    n = 48
    df = pd.DataFrame(
        {
            "cat0": rng.integers(0, 3, size=n).astype(float),
            "num0": rng.normal(size=n),
        }
    )
    X = _fix_dtypes(df, cat_indices=[0])
    encoder = _get_ordinal_encoder()
    process_text_na_dataframe(X, ord_encoder=encoder, fit_encoder=True)

    df_test = pd.DataFrame(
        {"cat0": rng.integers(0, 3, size=16).astype(float), "num0": rng.normal(size=16)}
    )
    X_test = _fix_dtypes(df_test, cat_indices=[0])
    encoded = process_text_na_dataframe(X_test, ord_encoder=encoder)
    assert _rank_corr(encoded[:, 0], df_test["cat0"].to_numpy()) > 0.999
    assert np.allclose(encoded[:, 1], df_test["num0"].to_numpy())


def test_constant_feature_has_finite_normalization_gradients():
    """Keeping a constant column must not produce NaN gradients.

    ``torch_nanstd`` takes ``sqrt(0)`` for a constant feature, whose derivative
    is infinite; multiplied by the zero downstream gradient this becomes NaN and
    poisons every parameter of the model.
    """
    torch.manual_seed(0)
    base = torch.randn(64, 1, 3)
    const = torch.full((64, 1, 1), 3.0)
    x = torch.cat([base, const], dim=-1).requires_grad_(True)

    out = normalize_data(x, normalize_positions=32)
    # Forward is unchanged: a constant feature normalizes to zero.
    assert torch.equal(out[..., 3], torch.zeros(64, 1))
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()

    v = x.detach().clone().requires_grad_(True)
    std = torch_nanstd(v[:32], axis=0)
    assert torch.isfinite(std).all()
    std.sum().backward()
    assert torch.isfinite(v.grad).all()
