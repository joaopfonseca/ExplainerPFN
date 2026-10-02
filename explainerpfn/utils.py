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
from sklearn.linear_model import LinearRegression

F = TypeVar("F", bound=Callable[..., object])


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
