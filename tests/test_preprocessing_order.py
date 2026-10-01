"""Tests for the explainer preprocessing pipeline.

These cover the order-preserving behaviour that the explainer relies on: the
model score and the feature being explained must remain at columns 0 and 1 of
the transformed data (see ``prepare_explanation_dataset``), even when the
dataset contains categorical features that would otherwise be reordered by the
``ColumnTransformer`` used in the preprocessing steps.
"""

import numpy as np
import pytest

from explainerpfn.model.preprocessing import (
    AddFingerprintFeaturesStep,
    EncodeCategoricalFeaturesStep,
    RemoveConstantFeaturesStep,
    ReshapeFeatureDistributionsStep,
    ShuffleFeaturesStep,
)


def _make_categorical_dataset(n=200, seed=0):
    """Build a dataset laid out as ``prepare_explanation_dataset`` produces it.

    Columns: [score, explained feature, num, cat, num, cat].
    """
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [
            rng.normal(size=n),  # model score
            rng.normal(size=n),  # explained feature
            rng.normal(size=n),  # numeric
            rng.integers(0, 5, size=n),  # categorical
            rng.normal(size=n),  # numeric
            rng.integers(0, 3, size=n),  # categorical
        ],
    ).astype(float)
    return X, [3, 5]


def test_reshape_is_order_preserving():
    from sklearn.preprocessing import RobustScaler

    X, cat_ix = _make_categorical_dataset()
    step = ReshapeFeatureDistributionsStep(
        transform_name="robust",
        append_to_original=False,
        global_transformer_name="svd",
        random_state=0,
    )
    Xt, cat_after = step.fit_transform(X, cat_ix)

    scaler0 = RobustScaler(unit_variance=True).fit(X[:, [0]])
    scaler1 = RobustScaler(unit_variance=True).fit(X[:, [1]])
    assert np.allclose(Xt[:, 0], scaler0.transform(X[:, [0]]).ravel())
    assert np.allclose(Xt[:, 1], scaler1.transform(X[:, [1]]).ravel())
    # Categorical passthrough columns stay in place and stay categorical.
    assert np.allclose(Xt[:, 3], X[:, 3])
    assert np.allclose(Xt[:, 5], X[:, 5])
    assert cat_after == cat_ix

    # Same invariant on the transform (test) path.
    Xt_test, _ = step.transform(X)
    assert np.allclose(Xt_test[:, 0], Xt[:, 0])
    assert np.allclose(Xt_test[:, 1], Xt[:, 1])


def test_encode_categorical_is_order_preserving():
    X, cat_ix = _make_categorical_dataset()
    reshape = ReshapeFeatureDistributionsStep(
        transform_name="robust",
        append_to_original=False,
        global_transformer_name="svd",
        random_state=0,
    )
    X, cat_ix = reshape.fit_transform(X, cat_ix)

    step = EncodeCategoricalFeaturesStep(
        "ordinal_very_common_categories_shuffled", random_state=0
    )
    Xt, cat_after = step.fit_transform(X, cat_ix)

    # Numeric columns untouched and in place; categoricals encoded in place.
    assert np.allclose(Xt[:, 0], X[:, 0])
    assert np.allclose(Xt[:, 1], X[:, 1])
    assert cat_after == cat_ix
    assert Xt.shape == X.shape

    Xt_test, _ = step.transform(X)
    assert np.allclose(Xt_test[:, 0], Xt[:, 0])
    assert np.allclose(Xt_test[:, 1], Xt[:, 1])


def test_remove_constant_features_protects_score_and_explained_feature():
    rng = np.random.default_rng(0)
    n = 50
    X = np.column_stack(
        [
            np.ones(n),  # constant score -- must NOT be dropped
            np.ones(n),  # constant explained feature -- must NOT be dropped
            rng.normal(size=n),
            np.ones(n),  # constant numeric -- dropped
        ],
    )
    step = RemoveConstantFeaturesStep()
    Xt, _ = step.fit_transform(X, [])
    assert Xt.shape[1] == 3
    assert np.all(Xt[:, 0] == 1.0)
    assert np.all(Xt[:, 1] == 1.0)
    assert not np.allclose(Xt[:, 2], 1.0)


def test_full_explainer_pipeline_keeps_score_and_feature_at_front():
    """End-to-end: after every default step, cols 0/1 are score/explained."""
    X, cat_ix = _make_categorical_dataset()

    remove = RemoveConstantFeaturesStep()
    X, cat_ix = remove.fit_transform(X, cat_ix)

    reshape = ReshapeFeatureDistributionsStep(
        transform_name="robust",
        append_to_original=False,
        global_transformer_name="svd",
        random_state=0,
    )
    X, cat_ix = reshape.fit_transform(X, cat_ix)

    encode = EncodeCategoricalFeaturesStep(
        "ordinal_very_common_categories_shuffled", random_state=0
    )
    X, cat_ix = encode.fit_transform(X, cat_ix)

    fingerprint = AddFingerprintFeaturesStep(random_state=0)
    X, cat_ix = fingerprint.fit_transform(X, cat_ix)

    for seed in (0, 1, 7):
        shuffle = ShuffleFeaturesStep(
            shuffle_method="shuffle", shuffle_index=0, random_state=seed
        )
        Xs, cat_shuffled = shuffle.fit_transform(X, cat_ix)
        # Score and explained feature unchanged at positions 0 and 1.
        assert np.allclose(Xs[:, 0], X[:, 0])
        assert np.allclose(Xs[:, 1], X[:, 1])
        # The rest actually gets shuffled.
        assert not np.allclose(Xs[:, 2:], X[:, 2:]) or X.shape[1] <= 2
        # Transform path yields the same layout.
        Xs_test, _ = shuffle.transform(X)
        assert np.allclose(Xs_test[:, 0], Xs[:, 0])
        assert np.allclose(Xs_test[:, 1], Xs[:, 1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"subsample_features": 0.5},
        {"append_to_original": True},
        {"apply_to_categorical": True},
        {"transform_name": "per_feature"},
    ],
)
def test_reshape_unsupported_branches_raise(kwargs):
    X, cat_ix = _make_categorical_dataset()
    step = ReshapeFeatureDistributionsStep(random_state=0, **kwargs)
    with pytest.raises(NotImplementedError):
        step.fit_transform(X, cat_ix)
