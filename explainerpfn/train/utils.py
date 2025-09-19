import numpy as np


def _check_random_state(random_state):
    if isinstance(random_state, int) or random_state is None:
        rng = np.random.default_rng(random_state)
    elif isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        raise ValueError(
            "random_state must be an int, None, or a np.random.Generator instance."
        )
    return rng


def postprocess_synthetic_data(
    df,
    dag_data=None,
    n_features=None,
    exclude_ind_nodes=False,
    target_is_successor=False,
    random_state=None,
):
    """
    Post-process the synthetic data. Selects features and a target variable.
    """
    rng = _check_random_state(random_state)

    # Exclude independent/source nodes if specified
    if dag_data is not None and exclude_ind_nodes:
        df.drop(columns=dag_data["ind_nodes"], inplace=True)

    n_features = (
        df.shape[1] - 1 if n_features is None else min(n_features, df.shape[1] - 1)
    )

    if dag_data is None and target_is_successor:
        raise ValueError(
            f"If `target_is_successor={target_is_successor}`, `dag_data` cannot be None."
        )
    elif dag_data is not None and target_is_successor:
        features = list(rng.choice(df.columns, size=n_features, replace=False))
        candidate_targets = [
            f"dep_{f}"
            for feat in features
            for f in dag_data["dag"].successors(int(feat.split("_")[-1]))
            if f"dep_{f}" not in features
        ]
        if not candidate_targets:
            # TODO: add warning
            return postprocess_synthetic_data(
                df,
                dag_data=dag_data,
                n_features=n_features,
                exclude_ind_nodes=exclude_ind_nodes,
                target_is_successor=False,
                random_state=rng,
            )
        target_col = rng.choice(candidate_targets)
    else:
        target_col = rng.choice(df.columns[df.columns.str.startswith("dep_")])
        features = list(
            rng.choice(df.columns.drop(target_col), size=n_features, replace=False)
        )

    target_col_name = f"target_{target_col.split('_')[-1]}"
    df = df[features + [target_col]].copy()
    df.rename(columns={target_col: target_col_name}, inplace=True)

    df = (df - df.mean()) / df.std(ddof=0)
    # df.columns = [i for i in range(df.shape[1] - 1)]
    df[target_col_name] = (df[target_col_name] > df[target_col_name].mean()).astype(int)

    return df


def kumaraswamy_distortion(x, a, b):
    """
    Cumulative distribution function of the Kumaraswamy distribution. With [0,
    1] scaling.
    """
    x = (x - x.min()) / (x.max() - x.min())
    return 1 - (1 - x**a) ** b
