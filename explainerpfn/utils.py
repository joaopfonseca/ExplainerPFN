from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression


def scores_to_ranking(y, direction=-1):
    """
    Converts an array with scores to a ranking.

    If higher rank values are better, set direction to 1 instead.
    """
    temp = np.argsort(y * direction)
    ranks = np.zeros(*y.shape, dtype=int)
    ranks[temp] = np.arange(*y.shape) + 1
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
