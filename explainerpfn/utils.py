"""Utility helpers for ExplainerPFN.

Includes the explanation-correction functions, plus the training-time
checkpoint I/O helpers (``_retry_io``, ``find_latest_checkpoint``) that make
saves robust on flaky cluster filesystems and power ``--auto-resume``. The
latter mirror the DiffusionExplainerPFN training utilities.
"""

import functools
import glob
import os
import re
import time
from typing import Callable, Optional, TypeVar

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

F = TypeVar("F", bound=Callable[..., object])

NA_PLACEHOLDER = "__MISSING__"


def _restore_encoder_column_order(
    X_encoded, ord_encoder, columns: "pd.Index"
) -> np.ndarray:
    """Undo the categorical-first reordering of the ordinal ``ColumnTransformer``."""
    X_encoded = np.asarray(X_encoded)
    if ord_encoder is None or not hasattr(ord_encoder, "transformers_"):
        return X_encoded
    selected = list(ord_encoder.transformers_[0][2])
    if not selected:
        return X_encoded
    position = {col: i for i, col in enumerate(columns)}
    selected_pos = [position[col] for col in selected]
    out_order = selected_pos + [
        i for i in range(len(columns)) if i not in set(selected_pos)
    ]
    return X_encoded[:, np.argsort(np.asarray(out_order))]


def process_text_na_dataframe(
    X: "pd.DataFrame",
    placeholder: str = NA_PLACEHOLDER,
    ord_encoder=None,
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    """Ordinal-encode text/category columns, preserving the column order.

    Mirror of ``tabpfn.utils._process_text_na_dataframe``. The sklearn
    ``ColumnTransformer`` inside the ordinal encoder emits the encoded
    categorical columns *first*, which silently permutes the features.  That is
    harmless for a permutation-invariant regressor but corrupts per-feature
    attribution slots, so the original column order is restored here.
    """
    X = X.copy()
    string_cols = X.select_dtypes(include=["string", "object"]).columns
    if len(string_cols) > 0:
        X[string_cols] = X[string_cols].fillna(placeholder)

    if fit_encoder and ord_encoder is not None:
        X_encoded = ord_encoder.fit_transform(X)
    elif ord_encoder is not None:
        X_encoded = ord_encoder.transform(X)
    else:
        X_encoded = np.asarray(X)

    X_encoded = _restore_encoder_column_order(X_encoded, ord_encoder, X.columns)

    string_cols_ix = [X.columns.get_loc(col) for col in string_cols]
    if len(string_cols_ix) > 0:
        placeholder_mask = (X[string_cols] == placeholder).to_numpy()
        X_encoded[:, string_cols_ix] = np.where(
            placeholder_mask,
            np.nan,
            X_encoded[:, string_cols_ix],
        )
    return np.asarray(X_encoded, dtype=np.float64)


def _retry_io(
    max_retries: int = 4,
    base_delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[F], F]:
    """Retry a file-writing function on transient ``OSError``/``IOError``.

    Cluster/NFS mounts intermittently raise EIO, ENOSPC, EAGAIN, or broken
    pipes during writes; retrying with exponential backoff keeps a long
    training run alive instead of crashing mid-save. Non-filesystem exceptions
    propagate immediately. Each retry is announced with an ``[io-retry]`` prefix.

    Args:
        max_retries: Extra attempts after the first failure (total calls =
            ``max_retries + 1``).
        base_delay: Seconds before the first retry.
        backoff: Multiplier applied to the delay after each retry.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = base_delay
            while True:
                try:
                    return func(*args, **kwargs)
                except (OSError, IOError) as exc:
                    attempt += 1
                    if attempt > max_retries:
                        print(
                            f"[io-retry] {func.__name__} failed after "
                            f"{max_retries + 1} attempts; giving up: {exc!r}",
                            flush=True,
                        )
                        raise
                    print(
                        f"[io-retry] {func.__name__} failed "
                        f"(attempt {attempt}/{max_retries + 1}): {exc!r}; "
                        f"retrying in {delay:.1f}s",
                        flush=True,
                    )
                    time.sleep(delay)
                    delay *= backoff

        return wrapper  # type: ignore[return-value]

    return decorator


def find_latest_checkpoint(save_dir: Optional[str]) -> Optional[str]:
    """Return the highest-step ``checkpoint_*.pt`` in ``save_dir``, else ``None``.

    ``final_model.pt`` is ignored: it is the artifact of a completed run, not a
    periodic resume point. Missing/empty directories also return ``None`` so
    ``--auto-resume`` can fall back to a fresh start.
    """
    if save_dir is None or not os.path.isdir(save_dir):
        return None

    candidates = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not candidates:
        return None

    step_re = re.compile(r"checkpoint_(\d+)\.pt$")
    best_step = -1
    best_path: Optional[str] = None
    for path in candidates:
        m = step_re.search(os.path.basename(path))
        if m is None:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    return best_path


def scores_to_ranking(y, direction=-1):
    """
    Converts an array with scores to a ranking.

    If higher rank values are better, set direction to 1 instead.
    """
    temp = np.argsort(y * direction)
    ranks = np.zeros(y.shape, dtype=int)
    ranks[temp] = np.arange(y.shape[0]) + 1
    return ranks


# def prepare_explanation_dataset(
#     X: np.ndarray,
#     y: np.ndarray,
#     feature_idx: int,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Prepares a dataset for explanation by concatenating the target variable and features,
#     and selecting a specific feature column as the explanation target. The target variable
#     is added as the first column of the feature matrix.
#
#     Parameters
#     ----------
#     X : np.ndarray
#         Feature matrix of shape (n_samples, n_features).
#     y : np.ndarray
#         Target vector of shape (n_samples,).
#     feature_idx : int
#         Index of the feature to be selected from `X`.
#
#     Returns
#     -------
#     X_concat : np.ndarray
#         Concatenated array of shape (n_samples, n_features + 1), where the first column is `y`
#         and the remaining columns are the features from `X`.
#     target_feature : np.ndarray
#         Array of shape (n_samples,) containing the values of the selected feature column from `X`.
#     """
#     target_feature = X[:, feature_idx].copy()
#     X_ = np.delete(X, feature_idx, axis=1)  # Remove the selected feature from X
#     X_concat = np.concatenate(
#         [y.reshape(-1, 1), X_], axis=1
#     )  # Add the original target feature as the first column
#     return X_concat, target_feature


def prepare_explanation_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepares a dataset for explanation by concatenating the target variable and features,
    and selecting a specific feature column as the explanation target. The target variable
    is added as the first column of the feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    feature_idx : int
        Index of the feature to be selected from `X`.

    Returns
    -------
    X_concat : np.ndarray
        Concatenated array of shape (n_samples, n_features), where the first column is the
        selected feature and the remaining columns are the other features from `X`.
    y : np.ndarray
        Array of shape (n_samples,) containing the target variable.
    """
    target_feature = X[:, feature_idx].copy()
    X_ = np.delete(X, feature_idx, axis=1)  # Remove the selected feature from X
    X_concat = np.concatenate(
        [y.reshape(-1, 1), target_feature.reshape(-1, 1), X_], axis=1
    )  # Add the feature to be explained and the original target feature as the first two columns
    return X_concat, target_feature


def multiplicative_correction(
    explanations,
    y_test,
    base_value,
    process_outliers=True,
    std_multiplier=3,
):
    """
    Corrects the explanations to ensure additivity.

    TODO/NOTE: This function is unfinished and does not work properly yet.
    """
    # Apply correction to ensure additivity
    eps = (y_test - base_value) / explanations.sum(axis=1)
    explanations_corrected = explanations * eps.reshape(-1, 1)

    # This approach can lead to outliers, so we replace them with the mean of each feature
    if process_outliers:
        threshold = y_test.std() * std_multiplier
        explanations_corrected = np.where(
            np.abs(
                explanations_corrected
                - explanations_corrected.mean(axis=1).reshape(-1, 1)
            )
            < threshold,
            explanations_corrected,
            np.nan,
        )

        mean = np.nanmean(explanations_corrected, axis=0)
        idx = np.where(np.isnan(explanations_corrected))
        explanations_corrected[idx] = np.take(mean, idx[1])

    return explanations_corrected


def additive_correction(
    explanations: np.ndarray,
    y_test: np.ndarray,
    base_value: float,
):
    """
    Corrects the explanations to ensure additivity.
    """
    eps = (y_test - base_value) - explanations.sum(axis=1)
    explanations_corrected = explanations + (eps.reshape(-1, 1) / explanations.shape[1])

    # Alternative (original) code
    # base_value = np.mean(y_train)
    # error = y_test - (explanations.sum(axis=1) + base_value)
    # error = (
    #     np.repeat(error.reshape(-1, 1), repeats=explanations.shape[1], axis=1)
    #     / explanations.shape[1]
    # )
    # explanations_corrected = explanations + error

    return explanations_corrected


def linear_correction(
    explanations: np.ndarray,
    y_test: np.ndarray,
    base_value: float,
    fit_intercept: bool = False,
):
    """
    Corrects the explanations to ensure additivity using a linear regression.
    """

    # explanations = (explanations.copy() * np.abs(df.corr()["target"].drop("target").values))**3

    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(explanations, y_test - base_value)
    explanations_corrected = explanations * np.abs(model.coef_.reshape(1, -1))

    return explanations_corrected


def statistical_correction(
    explanations: np.ndarray,
    y_test: np.ndarray,
    *args
    # base_value: float,
):
    """
    Corrects the explanations to ensure additivity using statistical measures.
    """
    # if isinstance(explanations, list):
    #     outputs = explanations
    #     explanations = np.array([exp["mean"] for exp in outputs]).T

    explanations_corrected = explanations - explanations.mean()
    explanations_corrected /= explanations.std()
    explanations_corrected *= y_test.std() / np.sqrt(explanations.shape[1])
    return explanations_corrected


def _column_match_score(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two columns under monotone recoding, in [0, 1]."""
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 1.0 if np.allclose(a, b) else -np.inf
    pearson = abs(float(np.corrcoef(a, b)[0, 1]))
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    spearman = abs(float(np.corrcoef(rank_a, rank_b)[0, 1]))
    return max(pearson, spearman)



def fit_item_mapping(X_raw, X_pre, threshold: float = 0.999) -> np.ndarray:
    """Map raw feature columns to preprocessed encoder-item columns.

    Matching uses Pearson and Spearman correlation, so monotone recodes (power
    transforms, ordinal encoding) still match. Returns ``mapping`` with
    ``mapping[i]`` the preprocessed column for raw column ``i``. Raises if a
    raw column cannot be matched, because silent misalignment would corrupt
    the attributions.

    NOTE: the TabPFN pipeline can expand a raw column into several preprocessed
    columns and can shuffle ordinal category codes (destroying monotonicity);
    in those cases this value-based matching is approximate. A dedicated 1:1
    preprocessing mode is required for exact per-feature alignment.
    """
    X_raw = np.asarray(X_raw, dtype=np.float64)
    X_pre = np.asarray(X_pre, dtype=np.float64)
    if X_raw.ndim != 2 or X_pre.ndim != 2:
        raise ValueError("fit_item_mapping expects 2-D arrays")
    if X_raw.shape[0] != X_pre.shape[0]:
        raise ValueError(
            f"row count mismatch: raw {X_raw.shape[0]} vs preprocessed "
            f"{X_pre.shape[0]}"
        )

    n_raw, n_pre = X_raw.shape[1], X_pre.shape[1]
    scores = np.full((n_raw, n_pre), -np.inf)
    for i in range(n_raw):
        for j in range(n_pre):
            scores[i, j] = _column_match_score(X_raw[:, i], X_pre[:, j])

    mapping = np.full(n_raw, -1, dtype=int)
    used: set[int] = set()
    order = np.dstack(
        np.unravel_index(np.argsort(scores, axis=None)[::-1], scores.shape)
    )[0]
    for i, j in order:
        i, j = int(i), int(j)
        if mapping[i] != -1 or j in used:
            continue
        if scores[i, j] < threshold:
            continue
        mapping[i] = j
        used.add(j)

    unmatched = [i for i in range(n_raw) if mapping[i] == -1]
    if unmatched:
        raise ValueError(
            "could not match raw feature columns to preprocessed items: "
            f"{unmatched} (raw d={n_raw}, preprocessed d={n_pre})"
        )
    return mapping
