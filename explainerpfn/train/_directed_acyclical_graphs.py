import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from explainerpfn.train._activations import ACTIVATIONS


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


def random_dag(n_nodes, edge_prob, random_state=None):
    """Generate a random Directed Acyclic Graph (DAG)."""
    rng = _check_random_state(random_state)

    G = nx.gnp_random_graph(n_nodes, edge_prob, directed=True, seed=rng)
    DAG = nx.DiGraph([(u, v) for (u, v) in G.edges() if u < v])

    # Add random edge weights
    for u, v in DAG.edges():
        DAG.edges[u, v]["weight"] = rng.uniform(-1, 1)

    return DAG


def redirection_sampling_dag(n_nodes, edge_prob, random_state=None):
    """Generate a random Directed Acyclic Graph (DAG) using the redirection sampling method."""
    rng = _check_random_state(random_state)

    dag = nx.DiGraph()
    dag.add_node(n_nodes)

    for source in range(n_nodes):
        target = rng.choice(dag.nodes)
        edge_weight = rng.uniform(-1, 1)

        if rng.uniform() < edge_prob or len(list(dag.successors(target))) == 0:
            dag.add_edge(source, target, weight=edge_weight)
        else:
            target = rng.choice(list(dag.successors(target)))
            dag.add_edge(source, target, weight=edge_weight)

    return dag


def random_join(source_dag, target_dag, random_state=None):
    """
    Randomly join two DAGs.
    """
    rng = _check_random_state(random_state)

    mapping = {i: i + len(source_dag.nodes) for i in target_dag.nodes}
    target_dag = nx.relabel_nodes(target_dag, mapping)

    source_node = rng.choice(source_dag.nodes)
    target_node = rng.choice(target_dag.nodes)

    dag_join = nx.disjoint_union(source_dag, target_dag)
    dag_join.add_edge(source_node, target_node, weight=rng.uniform(-1, 1))
    return dag_join


def plot_dag(DAG):
    """Plot the DAG using matplotlib."""
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(DAG)
    nx.draw(
        DAG, pos, with_labels=True, arrows=True, node_size=700, node_color="#A0CBE2"
    )
    edge_labels = nx.get_edge_attributes(DAG, "weight")
    formatted_edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(DAG, pos, edge_labels=formatted_edge_labels)
    # plt.tight_layout()
    plt.show()


def generate_synthetic_data(
    DAG, n_samples=1000, exclude_edge=True, random_state=None, activations=None
):
    """Generate synthetic data based on the DAG structure."""
    if isinstance(random_state, int) or random_state is None:
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    if activations is None:
        activations = ACTIVATIONS

    ind_nodes = [n for n in DAG.nodes if DAG.in_degree(n) == 0]
    dep_nodes = [n for n in nx.topological_sort(DAG) if n not in ind_nodes]
    node_values = {n: rng.normal(size=n_samples) for n in ind_nodes}

    for node in dep_nodes:
        activation = rng.choice(activations)
        source_nodes = list(DAG.predecessors(node))

        # NOTE: This is a linear connection
        # TODO: Add more connection types
        # For non-root nodes, use function of parent values (e.g., sum + some noise)
        values = sum(
            activation(node_values[parent]) * DAG.get_edge_data(parent, node)["weight"]
            for parent in source_nodes
        )

        node_values[node] = values

    df = pd.DataFrame(dict(sorted(node_values.items())))
    if exclude_edge:
        df.drop(columns=ind_nodes, inplace=True)

    return df


def postprocess_synthetic_data(df, n_features=None, random_state=None):
    """Post-process the synthetic data."""
    rng = _check_random_state(random_state)

    if n_features is None:
        n_features = df.shape[1]

    df = df.loc[:, rng.choice(df.columns, size=n_features, replace=False)]
    df = (df - df.mean()) / df.std(ddof=0)
    df.columns = [i for i in range(df.shape[1] - 1)] + ["target"]
    df.iloc[:, -1] = (df.iloc[:, -1] > 0).astype(int)

    return df
