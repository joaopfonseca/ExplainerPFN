"""Topology-diverse DAG generators for ExplainerPFN.

Adapted from DiffusionExplainerPFN @ commit d5f8258 (branch
``refactor/subpackage-restructure``), module
``diffusionexplainerpfn/data/dag_topologies.py``. Unmodified except for
import flattening/renaming. Used with permission of the author (same
project). See ``explainerpfn/train/do_shapley.py`` for the exact-label
scope of this copy (d <= 15, permutation-MC tier removed).

Implements four graph generation methods adapted from pcalg::randDAG (R):
- Erdős-Rényi (er): uniform edge probability
- Barabási-Albert (barabasi): preferential attachment, power-law degree
- Interconnected ER (interEr): community structure via ER islands
- Watts-Strogatz (watts): small-world (high clustering + short paths)

All generators produce NetworkX DiGraphs with 'weight' edge attributes and
respect a natural topological order (edges only from lower to higher index).
"""

import numpy as np
import networkx as nx

# =============================================================================
# Base utilities
# =============================================================================


def _add_weights(dag, rng, low=-1.0, high=1.0):
    """Add uniform random weights to all edges in a DAG."""
    for u, v in dag.edges():
        dag[u][v]["weight"] = rng.uniform(low, high)
    return dag


def _ensure_single_sink(dag, rng):
    """If multiple sinks exist, add a dummy super-sink node and redirect.

    The super-sink is appended as the highest-index node. All existing
    sinks get an edge to it. If there is already exactly one sink, the
    DAG is returned unchanged.

    Returns the (possibly modified) DAG and the sink node id.
    """
    sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]
    if len(sinks) == 1:
        return dag, sinks[0]

    # Add super-sink as a new node with highest index
    max_node = max(dag.nodes)
    super_sink = max_node + 1
    dag.add_node(super_sink)

    for s in sinks:
        dag.add_edge(s, super_sink, weight=rng.uniform(-1, 1))

    return dag, super_sink


# =============================================================================
# 1. Erdős-Rényi DAG
# =============================================================================


def erdos_renyi_dag(n_nodes, expected_degree, rng=None):
    """Generate a random DAG via Erdős-Rényi model.

    Nodes are ordered 0, 1, ..., n-1. Each directed edge (i, j) with i < j
    is present independently with probability p = expected_degree / (n - 1).

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAG.
    expected_degree : float
        Expected in+out degree per node. Edge probability p = d / (n - 1).
    rng : np.random.Generator, optional
        Random state.

    Returns
    -------
    dag : nx.DiGraph
        Directed acyclic graph with 'weight' attributes on edges.
    sink : int
        The sink node id.
    """
    if rng is None:
        rng = np.random.default_rng()

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    p = expected_degree / max(n_nodes - 1, 1)

    # Vectorized: create upper-triangular edge mask in one numpy call
    edge_mask = rng.uniform(size=(n_nodes, n_nodes)) < p
    edge_mask = np.triu(edge_mask, k=1)  # Only i < j (forward edges)
    u_indices, v_indices = np.where(edge_mask)

    for idx in range(len(u_indices)):
        dag.add_edge(u_indices[idx], v_indices[idx])

    _add_weights(dag, rng)

    dag, sink = _ensure_single_sink(dag, rng)

    return dag, sink


# =============================================================================
# 2. Barabási-Albert DAG
# =============================================================================


def barabasi_albert_dag(n_nodes, expected_degree, rng=None):
    """Generate a random DAG via Barabási-Albert preferential attachment.

    Nodes are added sequentially 0, 1, ..., n-1. Each new node connects to
    m = expected_degree / 2 earlier nodes (in expectation) with probability
    proportional to in-degree^k where k = attachment_param.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the DAG.
    expected_degree : float
        Target expected degree. m = max(1, int(expected_degree / 2)) edges
        are added per new node.
    rng : np.random.Generator, optional
        Random state.

    Returns
    -------
    dag : nx.DiGraph
        Directed acyclic graph with 'weight' attributes.
    sink : int
        The sink node id.
    """
    if rng is None:
        rng = np.random.default_rng()

    dag = nx.DiGraph()

    if n_nodes == 0:
        return dag, None

    # Start with first node
    dag.add_node(0)

    # Number of edges per new node
    m = max(1, int(expected_degree / 2))

    for i in range(1, n_nodes):
        dag.add_node(i)

        if i == 1:
            # Second node connects to first
            dag.add_edge(0, 1)
            continue

        # Candidate earlier nodes
        candidates = list(range(i))

        # Compute attachment probabilities proportional to in-degree^k
        in_degrees = np.array([dag.in_degree(c) for c in candidates], dtype=np.float64)
        # Add 1 to avoid zero probabilities for nodes with in-degree 0
        probs = (in_degrees + 1) ** 1.0
        probs /= probs.sum()

        # Sample m targets without replacement
        n_targets = min(m, i)
        targets = rng.choice(candidates, size=n_targets, replace=False, p=probs)

        for target in targets:
            dag.add_edge(target, i)

    _add_weights(dag, rng)

    dag, sink = _ensure_single_sink(dag, rng)

    return dag, sink


# =============================================================================
# 3. Interconnected ER Islands (interEr)
# =============================================================================


def inter_er_dag(
    n_nodes,
    expected_degree,
    rng=None,
):
    """Generate a DAG with community structure via interconnected ER islands.

    The graph consists of 2 clusters, each an Erdős-Rényi subgraph.
    Inter-cluster edges connect islands with probability proportional to
    0.25.

    Parameters
    ----------
    n_nodes : int
        Total number of nodes. Must be divisible by n_islands.
    expected_degree : float
        Expected degree within each island. Inter-island edges are sparser.
    rng : np.random.Generator, optional
        Random state.

    Returns
    -------
    dag : nx.DiGraph
        Directed acyclic graph with 'weight' attributes.
    sink : int
        The sink node id.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_islands = 2
    inter_connectivity = 0.25

    base_size = n_nodes // n_islands
    remainder = n_nodes % n_islands
    island_sizes = [base_size + 1] * remainder + [base_size] * (n_islands - remainder)

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    # Intra-island edges — vectorized with numpy
    offset = 0
    for island in range(n_islands):
        island_size = island_sizes[island]
        start, end = offset, offset + island_size
        offset = end
        p_intra = expected_degree / max(island_size - 1, 1)

        # Create upper-triangular mask for forward edges only
        mask = rng.uniform(size=(island_size, island_size)) < p_intra
        mask = np.triu(mask, k=1)  # Upper triangle only: i < j
        u_indices, v_indices = np.where(mask)

        for idx in range(len(u_indices)):
            dag.add_edge(start + u_indices[idx], start + v_indices[idx])

    # Precompute island boundaries for inter-island edges
    island_bounds = []
    off = 0
    for sz in island_sizes:
        island_bounds.append((off, off + sz))
        off += sz

    # Inter-island edges — vectorized with numpy for speed
    for i1 in range(n_islands):
        for i2 in range(i1 + 1, n_islands):
            start1, end1 = island_bounds[i1]
            start2, end2 = island_bounds[i2]
            size1, size2 = end1 - start1, end2 - start2

            # Create all possible (u,v) pairs as numpy arrays
            u_grid = np.arange(start1, end1)[:, None]  # (size1, 1)
            v_grid = np.arange(start2, end2)[None, :]  # (1, size2)

            # Sample all edges at once
            mask = rng.uniform(size=(size1, size2)) < inter_connectivity
            u_indices, v_indices = np.where(mask)

            for idx in range(len(u_indices)):
                dag.add_edge(u_grid[u_indices[idx], 0], v_grid[0, v_indices[idx]])

    _add_weights(dag, rng)

    dag, sink = _ensure_single_sink(dag, rng)

    return dag, sink


# =============================================================================
# 4. Watts-Strogatz DAG
# =============================================================================


def watts_strogatz_dag(n_nodes, expected_degree, rng=None):
    """Generate a small-world DAG via Watts-Strogatz model.

    Starts with a regular lattice where each node connects to its k nearest
    forward neighbors (k = expected_degree / 2). Then each edge is rewired
    with probability 0.5 to a random forward target.

    This produces high local clustering (from the lattice) plus short
    average path lengths (from random shortcuts).

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    expected_degree : float
        Target degree. k = int(expected_degree / 2) is the number of
        forward neighbors each node initially connects to.
    rng : np.random.Generator, optional
        Random state.

    Returns
    -------
    dag : nx.DiGraph
        Directed acyclic graph with 'weight' attributes.
    sink : int
        The sink node id.
    """
    if rng is None:
        rng = np.random.default_rng()

    rewiring_prob = 0.5

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    k = max(1, int(expected_degree / 2))

    # Regular lattice: each node i connects to i+1, i+2, ..., i+k
    for i in range(n_nodes):
        for offset in range(1, k + 1):
            j = i + offset
            if j < n_nodes:
                dag.add_edge(i, j)

    # Rewire edges — vectorized
    edges = list(dag.edges())
    n_edges = len(edges)
    rewire_mask = rng.uniform(size=n_edges) < rewiring_prob

    edges_to_remove = []
    edges_to_add = []

    for idx in np.where(rewire_mask)[0]:
        u, v = edges[idx]
        edges_to_remove.append((u, v))
        possible_targets = list(range(u + 1, n_nodes))
        if possible_targets:
            new_v = rng.choice(possible_targets)
            edges_to_add.append((u, new_v))

    for e in edges_to_remove:
        dag.remove_edge(*e)
    for e in edges_to_add:
        dag.add_edge(*e)

    _add_weights(dag, rng)

    dag, sink = _ensure_single_sink(dag, rng)

    return dag, sink


# =============================================================================
# Topology sampler
# =============================================================================

DAG_GENERATORS = {
    "er": erdos_renyi_dag,
    "barabasi": barabasi_albert_dag,
    "interEr": inter_er_dag,
    "watts": watts_strogatz_dag,
}

DEFAULT_TOPOLOGY_PROBS = {
    "er": 0.25,
    "barabasi": 0.30,
    "interEr": 0.25,
    "watts": 0.20,
}


def sample_topology_dag(n_nodes, expected_degree, topology_probs=None, rng=None):
    """Sample a DAG from a mixture of topology generators.

    Parameters
    ----------
    n_nodes : int
        Number of nodes.
    expected_degree : float
        Target expected degree passed to all generators.
    topology_probs : dict, optional
        Mapping from topology name to probability. Must sum to 1.
        Defaults to DEFAULT_TOPOLOGY_PROBS.
    rng : np.random.Generator, optional
        Random state.

    Returns
    -------
    dag : nx.DiGraph
    sink : int
    topology : str
        The chosen topology name.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = topology_probs or DEFAULT_TOPOLOGY_PROBS

    # Normalize probabilities
    total = sum(probs.values())
    normalized = {k: v / total for k, v in probs.items()}

    topologies = list(normalized.keys())
    pvals = [normalized[t] for t in topologies]

    topology = rng.choice(topologies, p=pvals)
    generator = DAG_GENERATORS[topology]

    dag, sink = generator(n_nodes, expected_degree, rng=rng)

    return dag, sink, topology
