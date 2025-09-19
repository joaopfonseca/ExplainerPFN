import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from explainerpfn.train._activations import ACTIVATIONS
from explainerpfn.train.utils import _check_random_state


def _initialization_sampling(n, rng, init_distr):
    if init_distr == "normal":
        return rng.normal(size=n)
    elif init_distr == "uniform":
        return rng.uniform(-1, 1, size=n)
    elif init_distr == "mixed" or init_distr is None:
        return rng.choice([rng.normal(size=n), rng.uniform(-1, 1, size=n)])
    else:
        raise ValueError(f"Unknown initialization distribution: {init_distr}")


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
    dag.add_node(n_nodes - 1)

    for source in range(n_nodes - 1):
        target = rng.choice(list(dag.nodes))
        edge_weight = rng.uniform(-1, 1)
        successors = list(dag.successors(target))

        if rng.uniform() > edge_prob and len(successors) != 0:
            target = rng.choice(successors)

        dag.add_edge(source, target, weight=edge_weight)

    return dag


def random_join(source_dag, target_dag, edge_prob, random_state=None):
    """
    Randomly join two DAGs.
    """
    rng = _check_random_state(random_state)

    mapping = {i: i + len(source_dag.nodes) for i in target_dag.nodes}
    target_dag = nx.relabel_nodes(target_dag, mapping)

    source_node = rng.choice(source_dag.nodes)
    target_node = rng.choice(target_dag.nodes)
    edge_weight = rng.uniform(-1, 1)
    dag_join = nx.disjoint_union(source_dag, target_dag)

    target_successors = list(dag_join.successors(target_node))
    if rng.uniform() > edge_prob and len(target_successors) != 0:
        target_node = rng.choice(target_successors)

    source_successors = list(dag_join.successors(source_node))
    if rng.uniform() > edge_prob and len(source_successors) != 0:
        source_node = rng.choice(source_successors)

    dag_join.add_edge(source_node, target_node, weight=edge_weight)

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
    DAG,
    noise_std=0,
    n_samples=1000,
    activations=None,
    init_distr="mixed",
    return_dag_data=False,
    random_state=None,
):
    """
    Generate synthetic data based on the DAG structure.

    init_distr: The distribution to use for initializing node values. Can be one
    of:
    - "normal": Gaussian distribution
    - "uniform": Uniform distribution
    - "mixed": Randomly choose between Gaussian and Uniform for each node
    """
    rng = _check_random_state(random_state)

    if activations is None:
        activations = ACTIVATIONS

    ind_nodes = [n for n in DAG.nodes if DAG.in_degree(n) == 0]
    dep_nodes = [n for n in nx.topological_sort(DAG) if n not in ind_nodes]
    node_values = {
        n: _initialization_sampling(n_samples, rng, init_distr) for n in ind_nodes
    }

    for node in dep_nodes:
        activation = rng.choice(activations)
        source_nodes = list(DAG.predecessors(node))

        values = activation(
            sum(
                node_values[parent] * DAG.get_edge_data(parent, node)["weight"]
                for parent in source_nodes
            )
            + rng.normal(0, noise_std, size=n_samples)
        )

        node_values[node] = values

    df = pd.DataFrame(dict(sorted(node_values.items())))
    df.columns = df.columns.astype(str)
    df.rename(
        columns={i: f"ind_{i}" if i in ind_nodes else f"dep_{i}" for i in df.columns},
        inplace=True,
    )

    if return_dag_data:
        dag_data = {"dag": DAG, "ind_nodes": ind_nodes, "dep_nodes": dep_nodes}
        return df, dag_data

    return df
