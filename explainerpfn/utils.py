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
