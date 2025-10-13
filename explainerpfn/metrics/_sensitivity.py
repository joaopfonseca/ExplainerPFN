import numpy as np
from sklearn.utils import check_random_state
from explainerpfn.utils import scores_to_ranking
from ._base import (
    _find_neighbors,
    _find_all_neighbors,
    _get_importance_mask,
    _ROW_WISE_MEASURES,
)


def _pairwise_outcome_sensitivity(
    row_data1,
    row_data2,
    row_cont1,
    row_cont2,
    score_func,
    original_scores,
    threshold,
    n_tests,
    stds,
    rng,
):
    # Find the most important features
    masks = [
        _get_importance_mask(row_cont, threshold) for row_cont in [row_cont1, row_cont2]
    ]
    mask1, mask2 = masks

    # Apply perturbation to the most important features
    perturbations = [rng.normal(loc=0, scale=stds) for _ in range(n_tests)]
    rows_pert1 = np.array([row_data1 + pert * mask1 for pert in perturbations])
    rows_pert2 = np.array([row_data2 + pert * mask2 for pert in perturbations])

    # Compute the prediction gap fidelity
    pert_ranks = []
    for rows_pert in [rows_pert1, rows_pert2]:
        rows_pert_score = score_func(rows_pert)
        rows_pert_rank = np.array(
            [
                scores_to_ranking(np.append(original_scores, score))[-1]
                for score in rows_pert_score
            ]
        )
        pert_ranks.append(rows_pert_rank)

    return np.abs(pert_ranks[0] - pert_ranks[1]).mean()


def row_based_outcome_sensitivity(
    original_data,
    rankings,
    original_scores,
    score_func,
    contributions,
    row_idx,
    threshold=0.8,
    n_neighbors=10,
    n_tests=10,
    std_multiplier=0.2,
    random_state=None,
):
    """
    Compute the sensitivity of the outcome based on the row index by
    comparing it with its neighbors.

    Parameters
    ----------
    original_data : array-like of shape (n_samples, n_features)
        The original dataset.
    rankings : array-like of shape (n_samples,)
        The rankings of the samples.
    original_scores : array-like of shape (n_samples,)
        The original scores of the samples.
    score_func : callable
        The scoring function used to evaluate the outcome.
    contributions : array-like of shape (n_samples, n_features)
        The contributions of each feature to the outcome.
    row_idx : int
        The index of the row for which the sensitivity is computed.
    threshold : float, optional, default=0.8
        The threshold value for the sensitivity computation.
    n_neighbors : int, optional, default=10
        The number of neighbors to consider for the sensitivity computation.
    n_tests : int, optional, default=10
        The number of tests to perform for the sensitivity computation.
    std_multiplier : float, optional, default=0.2
        The multiplier for the standard deviation used in the sensitivity computation.
    random_state : int, RandomState instance or None, optional, default=None
        The seed of the pseudo random number generator to use.

    Returns
    -------
    float
        The mean sensitivity score of the outcome for the given row index.
    """
    rng = check_random_state(random_state)
    # row_cont = np.array(contributions)[row_idx]

    # Select close neighbors
    data_neighbors, cont_neighbors = _find_neighbors(
        original_data, rankings, contributions, row_idx, n_neighbors
    )
    stds = np.std(original_data, axis=0) * std_multiplier

    # Compute distance between the target point and its neighbors
    scores = []
    for i in range(len(data_neighbors)):
        row_cont1 = np.array(contributions)[row_idx]
        row_cont2 = cont_neighbors[i]
        score = _pairwise_outcome_sensitivity(
            np.array(original_data)[row_idx],
            data_neighbors[i],
            row_cont1,
            row_cont2,
            score_func,
            original_scores,
            threshold,
            n_tests,
            stds,
            rng,
        )
        scores.append(score)

    return np.mean(scores)


def outcome_sensitivity(
    original_data,
    score_func,
    contributions,
    threshold=0.8,
    n_neighbors=10,
    n_tests=10,
    std_multiplier=0.2,
    aggregate_results=False,
    random_state=None,
):
    """
    Evaluate the sensitivity of the outcome to perturbations in the data.

    Parameters
    ----------
    original_data : array-like
        The original dataset.
    score_func : callable
        A function that computes scores for the data.
    contributions : array-like
        Contributions of each feature to the score.
    threshold : float, optional
        The threshold for determining sensitivity, by default 0.8.
    n_neighbors : int, optional
        Number of neighbors to consider for perturbations, by default 10.
    n_tests : int, optional
        Number of tests to perform for sensitivity analysis, by default 10.
    std_multiplier : float, optional
        Multiplier for the standard deviation in perturbations, by default 0.2.
    aggregate_results : bool, optional
        Whether to aggregate results into mean and standard error, by default False.
    random_state : int or None, optional
        Seed for random number generator, by default None.

    Returns
    -------
    sensitivities : array-like or tuple
        If `aggregate_results` is False, returns an array of sensitivities
        for each data point.
        If `aggregate_results` is True, returns a tuple containing the mean sensitivity
        and the standard error.
    """
    original_scores = score_func(original_data)
    rankings = scores_to_ranking(original_scores)

    sensitivities = np.vectorize(
        lambda row_idx: row_based_outcome_sensitivity(
            original_data,
            rankings,
            original_scores,
            score_func,
            contributions,
            row_idx,
            threshold,
            n_neighbors,
            n_tests,
            std_multiplier,
            random_state,
        )
    )(np.arange(len(original_data)))
    if aggregate_results:
        return (
            np.mean(sensitivities),
            np.std(sensitivities) / np.sqrt(sensitivities.size),
        )
    else:
        return sensitivities


def row_wise_explanation_sensitivity(
    original_data,
    contributions,
    row_idx,
    rankings,
    n_neighbors=10,
    agg_type="mean",
    measure="kendall",
    similar_outcome=True,
    **kwargs,
):
    """
    Calculate the sensitivity of explanations for a specific row by comparing it
    to its neighbors.

    Parameters
    ----------
    original_data : array-like
        The original dataset.
    contributions : array-like
        The contributions or explanations for each data point.
    row_idx : int
        The index of the row for which to calculate sensitivity.
    rankings : array-like
        The rankings of the data points.
    n_neighbors : int, optional
        The number of neighbors to consider (default is 10).
    agg_type : str, optional
        The type of aggregation to use for the distances ('mean' or 'max', default is 'mean').
    measure : str, optional
        The measure to use for calculating distances (default is 'kendall').
    similar_outcome : bool, optional
        Whether to consider only neighbors with similar outcomes (default is True).
    **kwargs : dict
        Additional keyword arguments to pass to the distance measure function.

    Returns
    -------
    float
        The aggregated distance between the target row and its neighbors, indicating the sensitivity of the explanation.

    Raises
    ------
    ValueError
        If an unknown aggregation type is provided.
    """
    row_cont = np.array(contributions)[row_idx]

    # Select close neighbors
    data_neighbors, cont_neighbors = _find_neighbors(
        original_data, rankings, contributions, row_idx, n_neighbors, similar_outcome
    )

    # Compute Kendall tau distance between the target point and its neighbors
    distances = np.apply_along_axis(
        lambda row: _ROW_WISE_MEASURES[measure](row, row_cont, **kwargs),
        1,
        cont_neighbors,
    )

    if agg_type == "max":
        return np.max(distances)
    elif agg_type == "mean":
        return np.mean(distances)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")


def row_wise_explanation_sensitivity_all_neighbors(
    original_data,
    contributions,
    row_idx,
    rankings,
    threshold=0.1,
    measure="kendall",
    **kwargs,
):
    """
    Calculate the sensitivity of row-wise explanations to all neighbors within a threshold.

    Parameters
    ----------
    original_data : array-like
        The original dataset.
    contributions : array-like
        The contributions or explanations for each data point.
    row_idx : int
        The index of the row for which to calculate sensitivity.
    rankings : array-like
        The rankings of the data points.
    threshold : float, optional
        The distance threshold to consider neighbors, by default 0.1.
    measure : str, optional
        The measure to use for calculating distance (e.g., "kendall"), by default "kendall".
    **kwargs : dict
        Additional keyword arguments to pass to the distance measure function.

    Returns
    -------
    measure_distances : ndarray
        The distances between the target point's contributions and its neighbors' contributions.
    rank_differences : ndarray
        The differences in rankings between the target point and its neighbors.
    feature_distances : ndarray
        The distances between the target point's features and its neighbors' features.
    """
    row_cont = np.array(contributions)[row_idx]
    row_rank = np.array(rankings)[row_idx]

    # Select all neighbors that are under the threshold
    data_neighbors, cont_neighbors, rank_neighbors, feature_distances = (
        _find_all_neighbors(original_data, rankings, contributions, row_idx, threshold)
    )

    # Compute the measure (e.g. Kendall tau) distance
    # between the target point and its neighbors
    measure_distances = np.apply_along_axis(
        lambda row: _ROW_WISE_MEASURES[measure](row, row_cont, **kwargs),
        1,
        cont_neighbors,
    )

    return measure_distances, row_rank - rank_neighbors, feature_distances


def explanation_sensitivity(
    original_data,
    contributions,
    rankings,
    n_neighbors=10,
    agg_type="mean",
    measure="kendall",
    similar_outcome=True,
    **kwargs,
):
    """
    Calculate the sensitivity of explanations for a given dataset.

    Parameters
    ----------
    original_data : array-like
        The original dataset for which explanations are generated.
    contributions : array-like
        The contributions or feature importances for each instance in the dataset.
    rankings : array-like
        The rankings of features for each instance in the dataset.
    n_neighbors : int, optional
        The number of neighbors to consider for sensitivity calculation, by default 10.
    agg_type : str, optional
        The type of aggregation to use for sensitivity calculation, by default "mean".
    measure : str, optional
        The measure to use for sensitivity calculation, by default "kendall".
    similar_outcome : bool, optional
        Whether to consider only neighbors with similar outcomes, by default True.
    **kwargs : dict
        Additional keyword arguments to pass to the row-wise sensitivity function.

    Returns
    -------
    tuple
        A tuple containing the mean sensitivity and the standard error of the mean sensitivity.
    """
    sensitivities = np.vectorize(
        lambda row_idx: row_wise_explanation_sensitivity(
            original_data,
            contributions,
            row_idx,
            rankings,
            n_neighbors,
            agg_type,
            measure,
            similar_outcome,
            **kwargs,
        )
    )(np.arange(len(original_data)))
    return np.mean(sensitivities), np.std(sensitivities) / np.sqrt(sensitivities.size)


def explanation_sensitivity_all_neighbors(
    original_data, contributions, rankings, measure="kendall", threshold=0.1, **kwargs
):
    """
    Calculate the sensitivity of explanations to all neighboring data points.

    Parameters
    ----------
    original_data : array-like
        The original dataset.
    contributions : array-like
        The contributions or feature importances for each data point.
    rankings : array-like
        The rankings of the features for each data point.
    measure : str, optional
        The measure to use for sensitivity calculation. Default is "kendall".
    threshold : float, optional
        The threshold value for sensitivity. Default is 0.1.
    **kwargs : dict
        Additional keyword arguments to pass to the sensitivity calculation function.

    Returns
    -------
    function
        A function that calculates row-wise explanation sensitivity for all neighbors.
    """
    return lambda row_idx: row_wise_explanation_sensitivity_all_neighbors(
        original_data, contributions, row_idx, rankings, threshold, measure, **kwargs
    )
