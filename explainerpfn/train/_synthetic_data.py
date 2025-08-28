import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


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


def _propagate_values(DAG, random_state=None):
    """Propagate values through the DAG."""
    rng = _check_random_state(random_state)

    node_values = {}
    for node in nx.topological_sort(DAG):
        parent_nodes = list(DAG.predecessors(node))
        if not parent_nodes:
            # For root nodes, assign random value
            value = rng.normal()
        else:
            # NOTE: This is a linear connection
            # TODO: Add more connection types
            # For non-root nodes, use function of parent values (e.g., sum + some noise)
            value = sum(
                node_values[parent] * DAG.get_edge_data(parent, node)["weight"]
                for parent in parent_nodes
            )

        node_values[node] = value

    return dict(sorted(node_values.items()))


def uniform_random_dag(n_nodes, edge_prob, random_state=None):
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


def generate_synthetic_data(DAG, num_samples=1000, random_state=None):
    """Generate synthetic data based on the DAG structure."""
    if isinstance(random_state, int) or random_state is None:
        rng = np.random.default_rng(random_state)
    else:
        rng = random_state

    data = [_propagate_values(DAG, random_state=rng) for _ in range(num_samples)]

    return pd.DataFrame(data)
