import numpy as np
from sklearn.utils import check_random_state
from ._base import _MEASURES
from ._sensitivity import _pairwise_outcome_sensitivity


def bootstrapped_explanation_consistency(
    population_result, batch_results, measure="kendall", **kwargs
):
    """
    Computes the mean and standard error of the mean (SEM) of the agreement between
    a population result and multiple batch results using a specified measure.

    Parameters
    ----------
    population_result : array-like
        The result from the entire population to compare against.
    batch_results : list of array-like
        A list of results from different batches to compare with the population result.
    measure : str, optional
        The measure to use for computing agreement. Default is "kendall".
    **kwargs : dict, optional
        Additional keyword arguments to pass to the measure function.

    Returns
    -------
    mean : float
        The mean agreement between the population result and the batch results.
    sem : float
        The standard error of the mean (SEM) of the agreement.
    """
    batch_agreement = []
    for batch_exp in batch_results:
        mean_agreement = _MEASURES[measure](
            population_result, batch_exp, **kwargs
        ).mean()
        batch_agreement.append(mean_agreement)

    batch_agreement = np.array(batch_agreement)
    mean = batch_agreement.mean()
    sem = np.std(batch_agreement) / np.sqrt(batch_agreement.size)
    return mean, sem


# Reviewed
def cross_method_explanation_consistency(
    results1, results2, measure="kendall", **kwargs
):
    """
    Computes the mean and SEM of the agreement between two sets of results
    using a specified measure.

    Parameters
    ----------
    results1 : array-like
        The first set of results to compare.
    results2 : array-like
        The second set of results to compare.
    measure : str, optional
        The measure to use for computing the agreement. Default is "kendall".
    **kwargs : dict
        Additional keyword arguments to pass to the measure function.

    Returns
    -------
    mean : float
        The mean agreement between the two sets of results.
    sem : float
        The standard error of the mean (SEM) of the agreement.
    """
    res_ = _MEASURES[measure](results1, results2, **kwargs)
    mean = res_.mean()
    sem = np.std(res_) / np.sqrt(res_.size)
    return mean, sem


def _row_based_outcome_consistency(
    original_data,
    original_scores,
    score_func,
    contributions1,
    contributions2,
    row_idx,
    threshold,
    n_tests,
    stds,
    rng,
):
    return _pairwise_outcome_sensitivity(
        np.array(original_data)[row_idx],
        np.array(original_data)[row_idx],
        np.array(contributions1)[row_idx],
        np.array(contributions2)[row_idx],
        score_func,
        original_scores,
        threshold,
        n_tests,
        stds,
        rng,
    )


def cross_method_outcome_consistency(
    original_data,
    score_func,
    contributions1,
    contributions2,
    threshold=0.8,
    n_tests=10,
    std_multiplier=0.2,
    random_state=None,
):
    """
    Computes the mean and SEM of the outcome consistency across all rows of data
    based on the contributions from two different methods.

    Parameters
    ----------
    original_data : array-like
        The original dataset for which the consistency is being measured.
    score_func : callable
        A function that computes the score for the original data.
    contributions1 : array-like
        Contributions from the first method.
    contributions2 : array-like
        Contributions from the second method.
    threshold : float, optional
        The threshold for determining consistency, by default 0.8.
    n_tests : int, optional
        The number of tests to perform for each row, by default 10.
    std_multiplier : float, optional
        Multiplier for the standard deviation used in perturbations, by default 0.2.
    random_state : int or None, optional
        Seed for the random number generator, by default None.

    Returns
    -------
    mean_consistency : float
        The mean consistency across all rows.
    sem_consistency : float
        The standard error of the mean (SEM) of the consistency across all rows.
    """
    original_scores = score_func(original_data)
    stds = np.std(original_data, axis=0) * std_multiplier
    rng = check_random_state(random_state)

    consistencies = np.vectorize(
        lambda row_idx: _row_based_outcome_consistency(
            original_data,
            original_scores,
            score_func,
            contributions1,
            contributions2,
            row_idx,
            threshold,
            n_tests,
            stds,
            rng,
        )
    )(np.arange(len(original_data)))
    return np.mean(consistencies), np.std(consistencies) / np.sqrt(consistencies.size)
