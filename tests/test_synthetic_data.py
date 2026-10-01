"""Tests for the synthetic data generation used during fine-tuning."""

from explainerpfn.train._directed_acyclical_graphs import (
    generate_synthetic_data,
    redirection_sampling_dag,
)
from explainerpfn.train.utils import postprocess_synthetic_data


def test_independent_nodes_get_ind_prefix():
    dag = redirection_sampling_dag(n_nodes=6, edge_prob=0.3, random_state=1)
    df, dag_data = generate_synthetic_data(
        dag, return_dag_data=True, random_state=1, n_samples=50
    )
    ind_nodes = {int(n) for n in dag_data["ind_nodes"]}
    dep_nodes = {int(n) for n in dag_data["dep_nodes"]}

    for node in ind_nodes:
        assert f"ind_{node}" in df.columns
    for node in dep_nodes:
        assert f"dep_{node}" in df.columns
    # No column is mislabelled (the previous bug labelled every column ``dep_``).
    assert len(ind_nodes) + len(dep_nodes) == len(df.columns)


def test_postprocess_exclude_ind_nodes_and_target_is_dependent():
    dag = redirection_sampling_dag(n_nodes=6, edge_prob=0.3, random_state=1)
    df, dag_data = generate_synthetic_data(
        dag, return_dag_data=True, random_state=1, n_samples=60
    )
    for target_is_successor in (True, False):
        out = postprocess_synthetic_data(
            df,
            dag_data,
            exclude_ind_nodes=True,
            target_is_successor=target_is_successor,
            random_state=0,
        )
        target_name = out.columns[-1]
        target_node = int(target_name.split("_")[-1])
        # The target must be a dependent (non-source) node.
        assert dag_data["dag"].in_degree(target_node) >= 1
        # Independent nodes have been excluded from the selected columns.
        assert not any(c.startswith("ind_") for c in out.columns)
