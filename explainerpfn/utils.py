import numpy as np


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
        Concatenated array of shape (n_samples, n_features + 1), where the first column is `y`
        and the remaining columns are the features from `X`.
    target_feature : np.ndarray
        Array of shape (n_samples,) containing the values of the selected feature column from `X`.
    """
    target_feature = X[:, feature_idx].copy()
    X_ = np.delete(X, feature_idx, axis=1)  # Remove the selected feature from X
    X_concat = np.concatenate(
        [y.reshape(-1, 1), X_], axis=1
    )  # Add the original target feature as the first column
    return X_concat, target_feature


def explanation_correction(
    explanations,
    y_test,
    base_value,
    adjust_std=True,
    process_outliers=True,
    std_multiplier=3,
):
    """
    Corrects the explanations to ensure additivity.

    TODO/NOTE: This function is unfinished and does not work properly yet.
    """
    # Initial adjustment of explanations' std
    if adjust_std:
        exp_ = explanations * y_test.std() / explanations.std()
    else:
        exp_ = explanations

    # Apply correction to ensure additivity (approach 2)
    eps = (y_test - base_value) / exp_.sum(axis=1)
    explanations_corrected = exp_ * eps.reshape(-1, 1)

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
