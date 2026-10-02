"""DAG-based synthetic data generator for ExplainerPFN.

Adapted from DiffusionExplainerPFN @ commit d5f8258 (branch
``refactor/subpackage-restructure``), module
``diffusionexplainerpfn/data/dag_generators.py``. Changes relative to
upstream:

- intra-package imports flattened (``..attributions.do_shapley`` ->
  ``.do_shapley``; ``..data.*`` -> ``.*``);
- the permutation-Monte-Carlo SHAP estimator (Tier 3) and its
  ``mc_shap_K`` / ``mc_v_samples`` configuration are removed. This copy
  supports exact-label regimes only (d <= 15); see ``do_shapley.py``.
  Introducing MC labels at d > 15 is DiffusionExplainerPFN's scaling
  contribution, not ExplainerPFN's;
- the ``SyntheticDataGenerator = DAGGenerator`` alias is dropped: the
  name is used by the ``SyntheticDataGenerator`` wrapper in
  ``explainerpfn/train/synthetic_data.py``.

This module generates synthetic tabular data from random DAGs (structural causal
models) with known ground-truth SHAP values. The do-Shapley attribution itself
lives in :mod:`explainerpfn.train.do_shapley`; ``DAGGenerator`` delegates to
``DoShapley`` after sampling each dataset.

Reference: Witter et al., 2026. "Exactly Computing do-Shapley Values." arXiv:2602.07203
"""

import hashlib

import numpy as np
import pandas as pd
import networkx as nx
import torch
import math
from scipy.special import betaincinv, ndtr
import logging
from typing import Optional

from .dag_topologies import sample_topology_dag, DEFAULT_TOPOLOGY_PROBS
from .scm import StructuralCausalModel, _sample_feature_value, extract_roots_and_target

logger = logging.getLogger(__name__)

# Salt for the per-dataset Beta-marginal hyper-parameters stream. Kept
# distinct from every other seed derivation so enabling/disabling the
# feature never perturbs ``self._rng`` (and therefore never changes the
# SCM, the ordinary seed stream, or any historical run's data).
_BETA_MARGINAL_SALT = "depfn:beta-marginals:v1"

# Number of grid points in the Beta inverse-CDF lookup table. The same
# table is used to transform the generated data columns and to draw
# off-coalition root values inside the SHAP value functions, so the
# labels integrate against *exactly* the marginal the data came from.
_BETA_LUT_SIZE = 4096

# Probability-space clip so the inverse CDF stays finite at the tails.
_BETA_U_EPS = 1e-6


def _derive_marginal_seed(random_state, salt: str = _BETA_MARGINAL_SALT) -> int:
    """Deterministically derive a seed for the marginal stream.

    Mirrors ``train_pipeline._derive_seed``'s SHA-256 approach so the
    marginal hyper-parameters are a pure function of the dataset seed.
    ``random_state=None`` (unseeded generation) falls back to a fresh
    entropy draw, matching ``np.random.default_rng(None)`` semantics.
    """
    if random_state is None:
        return int(np.random.SeedSequence().entropy) % (2**63)
    digest = hashlib.sha256(f"{random_state}|{salt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _beta_lut(a: float, b: float):
    """Quantile lookup table for a Beta(a, b) marginal.

    Returns ``(u_grid, x_grid, mu, sigma)`` where ``x_grid`` is the
    inverse CDF evaluated on a uniform probability grid and ``(mu,
    sigma)`` are the analytic mean/std used to standardize the transform
    (so the unit-variance root regime expected by the edge weights is
    preserved). The table itself is the marginal *before* standardization;
    callers apply ``(x - mu) / sigma``.
    """
    u = np.linspace(_BETA_U_EPS, 1.0 - _BETA_U_EPS, _BETA_LUT_SIZE)
    x = betaincinv(a, b, u)
    mu = a / (a + b)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    sigma = float(np.sqrt(var))
    return u, x, float(mu), sigma


def _beta_map_from_uniform(u: np.ndarray, spec) -> np.ndarray:
    """Map uniform draws through a precomputed Beta LUT and standardize."""
    u_grid, x_grid, mu, sigma = spec["lut"]
    x = np.interp(u, u_grid, x_grid)
    return (x - mu) / sigma


def _beta_map_from_normal(z: np.ndarray, spec) -> np.ndarray:
    """Map standard-normal draws to the standardized Beta marginal.

    ``u = Phi(z)`` converts the Gaussian draw to a uniform (inverse-
    transform sampling), then the Beta LUT produces a genuine Beta(a, b)
    draw. Consuming exactly one normal variate per output preserves the
    RNG stream of the legacy Gaussian path.
    """
    u = ndtr(z)
    return _beta_map_from_uniform(u, spec)


class DAGGenerator:
    """Simple DAG-based synthetic data generator with exact SHAP computation.

    Generates synthetic tabular data from random DAGs and computes:
    - exact_shap: Ground truth do-Shapley values from the known DAG structure
    - linear_shap: LinearSHAP approximation from a linear surrogate model

    The exact SHAP computation uses a two-tier strategy:
    1. Linear DAGs → closed-form path weights (Tier 1)
    2. Nonlinear DAGs → coalition enumeration (d <= 10) or the irreducible
       sets algorithm (10 < d <= 15) (Tier 2, Witter et al., 2026)

    (The upstream Tier-3 Monte-Carlo SHAP fallback is removed in this copy;
    datasets must stay within the exact-label regime d <= 15.)
    """

    def __init__(
        self,
        n_features_range=[5, 20],
        n_samples_range=[50, 200],
        n_dags_range=[1, 3],
        nodes_per_dag_range=[3, 8],
        edge_prob_range=[0.2, 0.4],
        feature_type_probs=None,
        observation_dependence=None,
        shap_method="auto",
        mc_samples=100,
        sink_sigmoid_steepness_range=(0.5, 3.0),
        sink_linear_prob=0.0,
        linear_dag_prob=0.0,  # Probability all non-root nodes use identity activation
        topology_probs=None,  # NEW: dict of topology -> probability
        mlp_hidden_units_range=(8, 16),  # MLP hidden layer size range
        mlp_prob=0.15,  # Probability of using MLP node function (vs scalar activation)
        mlp_sink_prob=0.0,  # Probability sink node uses MLP activation (vs linear or sigmoid).
        # Forces a non-additive X->y relationship that LinearSHAP
        # cannot approximate. W1/b1/W2/b2 are stored in
        # mlp_params[sink_node] so the existing SHAP paths
        # (closed-form/coalition/irreducible/MC) reconstruct
        # the sink's value at any coalition. Datasets with
        # MLP sinks route to Tier 2/3 SHAP (not closed-form).
        noise_std=0.1,  # Standard deviation of noise injected in data generation
        feature_correlation=0.0,  # Correlation strength (0=independent, >0 = block-correlated features)
        beta_marginals=False,  # Enable Beta base marginals for numerical roots
        beta_prob=0.25,  # Per-feature probability a numerical root gets a Beta marginal
        beta_shape_range=(0.3, 5.0),  # log-uniform (a, b) hyper-range for Beta shapes
        random_state=None,
        use_torch_shap: bool = True,
        shap_device: str = "cuda",
    ):
        self.n_features_range = n_features_range
        self.n_samples_range = n_samples_range
        self.n_dags_range = n_dags_range
        self.nodes_per_dag_range = nodes_per_dag_range
        self.edge_prob_range = edge_prob_range
        self.feature_type_probs = feature_type_probs or [0.6, 0.1, 0.2, 0.1]
        self.observation_dependence = observation_dependence
        self.shap_method = shap_method
        self.mc_samples = mc_samples  # Samples for Monte Carlo integration of v(S)
        self.sink_sigmoid_steepness_range = (
            sink_sigmoid_steepness_range  # (min_c, max_c) for y = σ(c·z)
        )
        self.sink_linear_prob = (
            sink_linear_prob  # Probability sink uses linear activation (identity)
        )
        self.linear_dag_prob = (
            linear_dag_prob  # Probability ALL non-root nodes use identity activation
        )
        self.topology_probs = topology_probs or DEFAULT_TOPOLOGY_PROBS.copy()
        self.mlp_hidden_units_range = (
            mlp_hidden_units_range  # (min_units, max_units) for MLP hidden layer
        )
        self.mlp_prob = mlp_prob  # Probability of using MLP node function
        self.mlp_sink_prob = mlp_sink_prob  # Probability sink uses MLP activation
        self.feature_correlation = feature_correlation  # Correlation strength (0=independent, >0 = block-correlated features)
        self.noise_std = noise_std  # Noise std for data generation
        # Beta base marginals for numerical root features (v32/v33/v34
        # robustness change; config-gated, default OFF so every historical
        # run and the ordinary seed stream stay byte-identical). When ON,
        # each numerical root independently gets a Beta(a, b) marginal with
        # prob ``beta_prob`` (a, b ~ log-uniform ``beta_shape_range``),
        # applied as a rank/quantile transform so the copula (prototype +
        # block correlation) is preserved, then re-standardized to unit
        # variance. The SHAP value function draws off-coalition roots from
        # the same marginal so the labels stay exact. Binary/categorical/
        # ordinal roots are untouched.
        self.beta_marginals = beta_marginals
        self.beta_prob = beta_prob
        self.beta_shape_range = tuple(beta_shape_range)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        # Per-dataset marginal spec: root node -> dict(a, b, lut) for roots
        # that received a Beta marginal, else absent (Gaussian/identity).
        # Repopulated on every dataset generation, like ``_observed_roots``.
        self._root_marginals = {}

        # Stage 4: torch/GPU SHAP value function. When the workload
        # is large enough (r closed sets × n_mc × n_samples > threshold)
        # and CUDA is available, the irreducible-sets SHAP path
        # batches v(S) calls into a single GPU forward pass via
        # dag_value_torch.py. ``shap_device`` can be "cuda" or "cpu"
        # to force a particular path even if the model is on the
        # other device. ``use_torch_shap=False`` disables the GPU
        # path entirely (falls back to the numpy path).
        self.use_torch_shap = use_torch_shap
        # Normalize ``shap_device`` to a real ``torch.device``. This
        # matters in two ways:
        #
        # 1. Thread safety: ``torch.cuda.set_device`` is thread-local.
        #    The SHAP value function runs inside ``AsyncBatchGenerator``
        #    worker threads, which inherit the SYSTEM default device
        #    (cuda:0) rather than any device the caller set. A string
        #    like "cuda" would therefore always resolve to cuda:0 in
        #    those threads. A real ``torch.device("cuda:1")`` is
        #    unambiguous and is honored regardless of the calling
        #    thread's set_device context.
        #
        # 2. Backward compatibility: any string that PyTorch
        #    understands ("cuda", "cuda:0", "cpu", ...) gets
        #    normalized; the original behavior of "cuda" → cuda:0
        #    is preserved.
        self.shap_device = (
            torch.device(shap_device) if isinstance(shap_device, str) else shap_device
        )
        if self.shap_device.type == "cuda" and not torch.cuda.is_available():
            self.shap_device = torch.device("cpu")

        # When an external X matrix is injected via generate_dataset(X_external=...),
        # we store it here so the MC SHAP value function can bootstrap-resample
        # from the empirical distribution instead of assuming Gaussian roots.
        # Set to None for purely synthetic data (the default).
        self._observed_roots: Optional[np.ndarray] = None

        # Activation functions — stored as (function, name) pairs so we can identify linear vs nonlinear
        self.activation_functions = [
            (lambda x: x, "linear"),
            (lambda x: np.maximum(0, x), "relu"),
            (lambda x: np.tanh(x), "tanh"),
            (lambda x: np.sin(x), "sin"),
            (lambda x: x * (1 + np.tanh(x)), "soft_interaction"),
            (lambda x: np.exp(-(x**2)), "gaussian"),
            (lambda x: 1.0 / (1.0 + np.exp(-x)), "sigmoid"),  # placeholder c=1; replaced below
            (lambda x: x, "mlp"),  # sentinel; replaced in _generate_data_from_dag
        ]

        # Sink activation: scaled sigmoid y = σ(c·z) where c is sampled per dataset.
        # This guarantees y ∈ (0, 1) — realistic binary classification probabilities —
        # and creates genuine nonlinearity that LinearSHAP cannot capture.
        # c is sampled log-uniform from sink_sigmoid_steepness_range to vary difficulty:
        #   low c (~0.5) → nearly-linear sigmoid, LinearSHAP does ok
        #   high c (~3.0) → sharp threshold, LinearSHAP fails badly
        self._sink_sigmoid_steepness = None  # Set per dataset in _generate_data_from_dag

    # ========================================================================
    # DAG Creation
    # ========================================================================

    def _create_dag(self, n_nodes, edge_prob):
        """Create a random DAG using topology-aware sampling.

        Samples from a mixture of topology generators (ER, BA, interEr, Watts)
        to produce structurally diverse DAGs. Replaces the old redirection-based
        generator which produced only uniform-degree, shallow DAGs.

        Parameters
        ----------
        n_nodes : int
            Number of nodes in the DAG.
        edge_prob : float
            Edge probability for Erdős-Rényi (legacy parameter). Converted to
            expected_degree = edge_prob * (n_nodes - 1) for topology generators.

        Returns
        -------
        dag : nx.DiGraph
            Directed acyclic graph with 'weight' edge attributes and a single
            sink node (highest index).
        """
        expected_degree = edge_prob * max(n_nodes - 1, 1)
        dag, _, _ = sample_topology_dag(
            n_nodes=n_nodes,
            expected_degree=expected_degree,
            topology_probs=self.topology_probs,
            rng=self._rng,
        )
        return dag

    def _join_dags(self, dag1, dag2, edge_prob):
        """Join two DAGs with a random edge."""
        offset = len(dag1.nodes)
        mapping = {i: i + offset for i in dag2.nodes}
        dag2_offset = nx.relabel_nodes(dag2, mapping)

        combined = nx.compose(dag1, dag2_offset)

        source_node = self._rng.choice(list(dag1.nodes))
        target_node = self._rng.choice(list(dag2_offset.nodes))
        weight = self._rng.uniform(-1, 1)
        combined.add_edge(source_node, target_node, weight=weight)

        return combined

    def _ensure_n_root_features(self, dag, n_features):
        """Augment a DAG with synthetic root nodes so X has the
        requested number of root-feature columns.

        The topology generators (ER, BA, interEr, Watts) can produce
        DAGs with far fewer root nodes than the user requested via
        ``n_features_range`` (e.g. 1-3 roots for a 5-node BA DAG). To
        honor the user's requested root-feature count, this method
        adds additional independent root nodes that are direct
        predecessors of the sink. Each synthetic root has its own
        edge weight in ``[-0.5, 0.5]`` (smaller than typical edge
        weights of ``[-1, 1]`` so the synthetic roots contribute
        modestly to the prediction).

        Args:
            dag: NetworkX DiGraph (modified in place AND returned).
            n_features: The desired number of root features (i.e.
                the size of the X column axis).

        Returns:
            (dag, n_features_actual) — the (possibly augmented)
            DAG and the resulting number of roots.
        """
        # Find current root nodes (in_degree == 0 and not the sink).
        # We use _find_sink_node + a topological-order scan to mirror
        # the convention used elsewhere in this file.
        sink_node = self._find_sink_node(dag)
        topo_order = list(nx.topological_sort(dag))
        current_roots = [n for n in topo_order if dag.in_degree(n) == 0 and n != sink_node]
        n_to_add = n_features - len(current_roots)
        if n_to_add <= 0:
            return dag, len(current_roots)

        # Add new nodes. Each new node is given a fresh integer ID
        # greater than any existing one (we don't reuse node IDs —
        # the SHAP math relies on the node-to-column mapping being
        # one-to-one with topological-order indices, so we re-do
        # the topo sort in _compute_target_and_shap and it picks up
        # the new nodes naturally).
        next_id = max(dag.nodes) + 1
        for _ in range(n_to_add):
            new_id = next_id
            next_id += 1
            dag.add_node(new_id)
            # Synthetic root has a single direct edge to the sink
            # with a small weight. Small weight keeps the synthetic
            # root's marginal contribution to y modest relative to
            # the "real" DAG structure.
            weight = self._rng.uniform(-0.5, 0.5)
            dag.add_edge(new_id, sink_node, weight=weight)

        return dag, len(current_roots) + n_to_add

    def _find_sink_node(self, dag):
        """Find the sink node (no outgoing edges) in the DAG."""
        sinks = [n for n in dag.nodes if dag.out_degree(n) == 0]
        if len(sinks) == 1:
            return sinks[0]
        elif len(sinks) == 0:
            # No sink — use last node in topological order
            return list(nx.topological_sort(dag))[-1]
        else:
            # Multiple sinks — pick the one with highest topological order
            topo = list(nx.topological_sort(dag))
            for node in reversed(topo):
                if node in sinks:
                    return node
            return sinks[0]

    # ========================================================================
    # Data Generation from DAG
    # ========================================================================

    def _generate_feature_value(self, n_samples, feature_type):
        """Generate a single feature value based on its type."""
        return _sample_feature_value(self._rng, n_samples, feature_type)

    def _generate_initialization_data(self, n_samples, n_features):
        """Generate initialization data with controllable observation interdependency
        and optional feature correlation.

        Implements TabPFN-style initialization data sampling:
        - Samples prototypes (random fraction of samples)
        - Generates each sample as weighted combination of prototypes
        - Controls dependence via temperature parameter beta

        When feature_correlation > 0, applies a block correlation structure
        to the root features to simulate real-world correlated feature sets.
        """
        if self.observation_dependence is None and self.feature_correlation <= 0:
            return self._rng.standard_normal((n_samples, n_features))

        if self.observation_dependence is not None:
            prototype_fraction = self._rng.uniform(0.1, 0.4)
            n_prototypes = max(1, int(n_samples * prototype_fraction))
            prototypes = self._rng.standard_normal((n_prototypes, n_features))

            beta = max(self.observation_dependence, 0.1)
            weights = self._rng.dirichlet([beta] * n_prototypes, size=n_samples)

            initialization_data = np.zeros((n_samples, n_features))
            for i in range(n_samples):
                for j in range(n_prototypes):
                    initialization_data[i] += weights[i, j] * prototypes[j]
        else:
            initialization_data = self._rng.standard_normal((n_samples, n_features))

        # Apply feature correlation if requested
        if self.feature_correlation > 0 and n_features > 1:
            # Create block correlation structure: group features into clusters
            # Each cluster shares a common latent factor
            n_clusters = max(2, n_features // 4)  # ~4 features per cluster
            cluster_size = max(1, n_features // n_clusters)

            # Generate cluster latent factors
            cluster_factors = self._rng.standard_normal((n_samples, n_clusters))

            # Mix in correlation: X = sqrt(1-ρ) * X_independent + sqrt(ρ) * X_cluster
            rho = self.feature_correlation
            sqrt_1_minus_rho = np.sqrt(1.0 - rho)
            sqrt_rho = np.sqrt(rho / cluster_size) if cluster_size > 0 else 0

            for j in range(n_features):
                cluster_idx = min(j // max(cluster_size, 1), n_clusters - 1)
                initialization_data[:, j] = (
                    sqrt_1_minus_rho * initialization_data[:, j]
                    + sqrt_rho * cluster_factors[:, cluster_idx]
                )

        return initialization_data

    # ------------------------------------------------------------------
    # Beta base marginals (config-gated; default OFF)
    # ------------------------------------------------------------------

    def _draw_beta_marginals(self, numerical_root_nodes):
        """Pick which numerical roots get a Beta marginal and their shapes.

        Returns ``{node: spec}`` with ``spec = {"a", "b", "lut"}`` for roots
        selected by the per-feature coin. All randomness comes from a salted
        stream keyed on ``random_state`` so ``self._rng`` is never consumed
        here and the legacy (OFF) data stream is unaffected.
        """
        if not self.beta_marginals or self.beta_prob <= 0 or not numerical_root_nodes:
            return {}
        rng = np.random.default_rng(_derive_marginal_seed(self.random_state))
        lo, hi = self.beta_shape_range
        log_lo, log_hi = math.log(lo), math.log(hi)
        spec = {}
        for node in numerical_root_nodes:
            if rng.random() >= self.beta_prob:
                continue  # keep the Gaussian base for this feature
            a = float(math.exp(rng.uniform(log_lo, log_hi)))
            b = float(math.exp(rng.uniform(log_lo, log_hi)))
            spec[node] = {"a": a, "b": b, "lut": _beta_lut(a, b)}
        return spec

    def _apply_beta_marginal(self, z_col: np.ndarray, spec) -> np.ndarray:
        """Rank/quantile-transform a generated Gaussian column to Beta.

        ``z_col`` is the (possibly correlation-mixed) Gaussian base column.
        Mapping through ``Phi`` then the Beta inverse CDF is monotone, so
        the copula (and hence prototype/block-correlation structure) is
        preserved; the output is standardized to unit variance.
        """
        return _beta_map_from_normal(z_col, spec)

    def _generate_data_from_dag(self, dag, n_samples, X_external=None):
        """Generate data from DAG structure with feature type diversity.

        Returns:
            df: DataFrame with generated data
            node_activations: dict mapping node -> activation name ('linear', 'relu', etc.)
            feature_types: dict mapping node -> feature type name
            sink_node: the target node

        When ``X_external`` is provided (an ``(n_samples, n_features)`` numpy
        array), the root features are taken from the external matrix instead of
        being sampled from a Gaussian. All roots are forced to ``numerical``
        type. The external X must be standardized (mean 0, std 1) before
        injection, since edge weights are tuned for unit-variance roots.
        """
        # Use local mlp_params dict to avoid thread-safety issues
        mlp_params = {}
        topo_order = list(nx.topological_sort(dag))
        sink_node = self._find_sink_node(dag)

        # With linear_dag_prob, generate fully linear DAGs where ALL non-root
        # nodes use identity activation. Exact SHAP = LinearSHAP (residual = 0).
        # These serve as cheap regularization — teaching the model when NOT to
        # overcorrect. The inference-time bypass (R² >= 0.95) handles truly
        # linear real-world data; linear DAGs in training calibrate the model
        # for borderline cases (R² ~ 0.90-0.95) where diffusion still runs.
        force_linear = self.linear_dag_prob > 0 and self._rng.random() < self.linear_dag_prob

        # Identify root nodes (needed before feature type assignment
        # so we can force roots to numerical when X_external is provided)
        root_nodes = [n for n in topo_order if dag.in_degree(n) == 0]

        # Assign feature types to each node
        feature_types = {}
        for node in topo_order:
            feature_types[node] = self._rng.choice(
                ["numerical", "binary", "categorical", "ordinal"], p=self.feature_type_probs
            )

        # When X_external is provided, force all roots to numerical and
        # skip random feature type assignment for root nodes.
        if X_external is not None:
            for node in root_nodes:
                feature_types[node] = "numerical"

        # Assign activation functions to each non-root node
        node_activations = {}

        # Initialize node values
        node_values = {}

        if X_external is not None:
            # External X injection: use provided values for root features.
            # Root nodes are in topological order; X_external columns are
            # mapped to root_nodes by index (matching root_to_X_col).
            for idx, node in enumerate(root_nodes):
                if idx < X_external.shape[1]:
                    node_values[node] = X_external[:, idx].copy()
        else:
            # Handle numerical features with observation interdependency
            numerical_root_nodes = [
                node for node in root_nodes if feature_types[node] == "numerical"
            ]
            if numerical_root_nodes:
                numerical_init_data = self._generate_initialization_data(
                    n_samples, len(numerical_root_nodes)
                )
                # Optional Beta base marginals: rank/quantile-transform
                # selected numerical root columns. Drawn from a salted
                # stream so ``self._rng`` (and thus the SCM and the legacy
                # seed stream) is untouched regardless of this branch.
                beta_spec = self._draw_beta_marginals(numerical_root_nodes)
                for idx, node in enumerate(numerical_root_nodes):
                    col = numerical_init_data[:, idx]
                    spec = beta_spec.get(node)
                    if spec is not None:
                        col = self._apply_beta_marginal(col, spec)
                        self._root_marginals[node] = spec
                    node_values[node] = col

            # Assign values to non-numerical root nodes
            for node in root_nodes:
                if node in node_values:
                    continue
                node_values[node] = self._generate_feature_value(n_samples, feature_types[node])

        # Root nodes are always "linear" (no activation applied)
        for node in root_nodes:
            node_activations[node] = "linear"

        # Compute values for non-root nodes
        for node in topo_order:
            if node in node_values:
                continue

            parents = list(dag.predecessors(node))
            if not parents:
                node_values[node] = self._generate_feature_value(n_samples, feature_types[node])
                node_activations[node] = "linear"
                continue

            # Choose activation function
            # Sink node: with sink_linear_prob, use linear activation (identity).
            # With force_linear, ALWAYS use identity (overrides sink_linear_prob).
            # Otherwise use scaled sigmoid y = σ(c·z) with c sampled log-uniform
            # once per dataset.
            if node == sink_node:
                if force_linear or (
                    self.sink_linear_prob > 0 and self._rng.random() < self.sink_linear_prob
                ):
                    # Linear sink: y = z (identity, same as v6/v7/v8 training)
                    activation_fn = lambda x: x
                    activation_name = "linear"
                    self._sink_sigmoid_steepness = None
                elif self.mlp_sink_prob > 0 and self._rng.random() < self.mlp_sink_prob:
                    # MLP sink: a real non-additive X->y relationship. Reuses
                    # the same Xavier-init + tanh + linear-output pattern as
                    # intermediate MLPs (see the mlp branch below). W1/b1/W2/b2
                    # are stored in mlp_params[sink_node] so the existing SHAP
                    # value functions can reconstruct y at any coalition. This
                    # makes LinearSHAP a measurably worse approximation: y is
                    # a non-linear function of multiple parents' values, not
                    # just a sharp sigmoid of one linear combination.
                    n_parents = len(parents)
                    min_h, max_h = self.mlp_hidden_units_range
                    n_hidden = int(self._rng.integers(min_h, max_h + 1))
                    seed = self._rng.integers(0, 2**31)
                    mlp_rng = np.random.default_rng(seed)
                    W1 = mlp_rng.normal(
                        0, np.sqrt(2.0 / (n_parents + n_hidden)), (n_parents, n_hidden)
                    )
                    b1 = mlp_rng.normal(0, 0.1, n_hidden)
                    W2 = mlp_rng.normal(0, np.sqrt(2.0 / (n_hidden + 1)), (n_hidden,))
                    b2 = mlp_rng.normal(0, 0.1)
                    activation_name = f"mlp_sink_h{n_hidden}_s{seed}"
                    mlp_params[node] = (n_parents, W1, b1, W2, b2)
                    activation_fn = None  # Forward pass handled by the 'mlp*' branch below
                    self._sink_sigmoid_steepness = None
                else:
                    # Sigmoid sink: sample steepness log-uniform
                    min_c, max_c = self.sink_sigmoid_steepness_range
                    log_min, log_max = math.log(min_c), math.log(max_c)
                    self._sink_sigmoid_steepness = math.exp(self._rng.uniform(log_min, log_max))
                    c = self._sink_sigmoid_steepness
                    activation_fn = lambda x: 1.0 / (1.0 + np.exp(-c * x))
                    activation_name = f"sigmoid_c{c:.2f}"
            elif force_linear:
                # Fully linear DAG: override all intermediate nodes to identity
                activation_fn = lambda x: x
                activation_name = "linear"
            else:
                # Non-sink intermediate node: choose activation type.
                # With mlp_prob, use an MLP node function; otherwise sample
                # from the scalar activation pool as before.
                if self._rng.random() < self.mlp_prob:
                    # --- MLP node function ---
                    # Small random-weight MLP: takes all parent values as
                    # separate inputs (not a weighted sum), applies 1 hidden
                    # layer with tanh, outputs a scalar. Weights are fixed
                    # (deterministic given seed) so exact SHAP computation
                    # via MC/irreducible sets still works.
                    n_parents = len(parents)
                    min_h, max_h = self.mlp_hidden_units_range
                    n_hidden = int(self._rng.integers(min_h, max_h + 1))
                    seed = self._rng.integers(0, 2**31)
                    mlp_rng = np.random.default_rng(seed)

                    # Xavier-ish init
                    W1 = mlp_rng.normal(
                        0, np.sqrt(2.0 / (n_parents + n_hidden)), (n_parents, n_hidden)
                    )
                    b1 = mlp_rng.normal(0, 0.1, n_hidden)
                    W2 = mlp_rng.normal(0, np.sqrt(2.0 / (n_hidden + 1)), (n_hidden,))
                    b2 = mlp_rng.normal(0, 0.1)

                    activation_name = f"mlp_h{n_hidden}_s{seed}"
                    # Store params so value functions can reconstruct
                    mlp_params[node] = (n_parents, W1, b1, W2, b2)
                    activation_fn = None  # Not used for MLP
                else:
                    activation_fn, activation_name = self._rng.choice(self.activation_functions)
                    # Sigmoid: replace placeholder with random steepness per node
                    # Wider range than sink (0.3–5.0) since intermediates aren't
                    # constrained to produce probabilities.
                    if activation_name == "sigmoid":
                        log_min, log_max = math.log(0.3), math.log(5.0)
                        c = math.exp(self._rng.uniform(log_min, log_max))
                        activation_fn = lambda x, c=c: 1.0 / (1.0 + np.exp(-c * x))
                        activation_name = f"sigmoid_c{c:.2f}"
            node_activations[node] = activation_name

            # Weighted sum of parent values
            weighted_sum = np.zeros(n_samples)
            for parent in parents:
                weight = dag[parent][node]["weight"]
                weighted_sum += node_values[parent] * weight

            # Apply activation
            if activation_name.startswith("mlp_"):
                # MLP forward pass: stack parent values, apply W1/tanh/W2
                n_parents, W1, b1, W2, b2 = mlp_params[node]
                parent_vals = np.column_stack(
                    [node_values[p] for p in parents]
                )  # (n_samples, n_parents)
                hidden = np.tanh(parent_vals @ W1 + b1)  # (n_samples, n_hidden)
                noise = self._rng.normal(0, self.noise_std, size=n_samples)
                node_values[node] = hidden @ W2 + b2 + noise
            else:
                # Scalar activation on weighted sum
                noise = self._rng.normal(0, self.noise_std, size=n_samples)
                node_values[node] = activation_fn(weighted_sum + noise)

        # Create DataFrame
        df_data = {str(node): node_values[node] for node in topo_order}
        df = pd.DataFrame(df_data)

        # Rename columns
        new_columns = {}
        for col in df.columns:
            node_id = int(col)
            if dag.in_degree(node_id) == 0:
                new_columns[col] = f"ind_{node_id}"
            else:
                new_columns[col] = f"dep_{node_id}"
        df = df.rename(columns=new_columns)

        return df, node_activations, feature_types, sink_node, mlp_params

    # ========================================================================
    # do-Shapley attribution (delegated to attributions.DoShapley)
    # ========================================================================

    def _build_shap_computer(self, scm=None):
        """Construct a ``DoShapley`` bound to an SCM's attribution state.

        When ``scm`` is given, its snapshotted per-dataset state
        (``root_marginals``, ``observed_roots``) is used instead of the
        generator's live buffers — required for deferred ``compute_shap``
        calls. When ``scm`` is ``None`` the generator's current state is used
        (the default ``generate_dataset`` path). Building eagerly in
        ``__init__`` is impossible because the marginal/observed-root buffers
        are repopulated on every ``generate_dataset`` call.
        """
        from .do_shapley import DoShapley

        if scm is not None:
            observed_roots = scm.observed_roots
            root_marginals = scm.root_marginals
        else:
            observed_roots = self._observed_roots
            root_marginals = self._root_marginals

        return DoShapley(
            rng=self._rng,
            generate_feature_value=self._generate_feature_value,
            root_marginal_map=_beta_map_from_normal,
            observed_roots=observed_roots,
            root_marginals=root_marginals,
            shap_method=self.shap_method,
            mc_samples=self.mc_samples,
            use_torch_shap=self.use_torch_shap,
            shap_device=self.shap_device,
        )

    def _compute_target_and_shap(
        self, dag, df, node_activations, feature_types, sink_node, mlp_params
    ):
        """Compute target and do-Shapley values via ``DoShapley``."""
        return self._build_shap_computer().compute(
            dag, df, node_activations, feature_types, sink_node, mlp_params
        )

    # ========================================================================
    # Public API
    # ========================================================================

    def generate_dataset(self, X_external=None, include_shap=True):
        """Generate a single synthetic dataset.

        Args:
            X_external: Optional ``(n_samples, n_features)`` numpy array.
                When provided, the root features are taken from this matrix
                instead of being sampled from a Gaussian. The external X
                is standardized (mean 0, std 1) before injection to match
                the scale assumed by edge weights and the sigmoid sink.
                All roots are forced to ``numerical`` type. MC SHAP
                resampling uses empirical bootstrap from the observed X
                instead of Gaussian. ``n_samples`` and ``n_features`` are
                derived from the shape of X_external, overriding the
                generator's range parameters.
            include_shap: When ``True`` (default), compute and return the
                do-Shapley attributions as well (the historical behaviour).

                When ``False``, return only the generative elements and the
                :class:`~diffusionexplainerpfn.data.scm.StructuralCausalModel`
                that produced them. Pass the returned dict to
                :meth:`compute_shap` to obtain the attributions later (or use
                :meth:`~diffusionexplainerpfn.attributions.do_shapley.DoShapley.from_scm`
                directly).

        Returns:
            ``include_shap=True``: dict with keys
                'X': (n_samples, n_features) input features (float32)
                'y': (n_samples,) continuous target values (sink node)
                'p_hat': (n_samples,) model predictions (= y for synthetic data)
                'exact_shap': (n_samples, n_features) exact Shapley values
                'linear_shap': (n_samples, n_features) LinearSHAP approximation
            ``include_shap=False``: dict with keys
                'X': (n_samples, n_features) root features (float64)
                'y': (n_samples,) continuous target values
                'scm': the :class:`StructuralCausalModel` recipe
        """
        # When X_external is provided, derive n_samples and n_features
        # from it and store the standardized X for MC bootstrap resampling.
        # Beta marginals only apply to the generator's own root sampling;
        # external X already carries its own (empirical) marginal, handled
        # by the observed-roots bootstrap.
        self._root_marginals = {}
        if X_external is not None:
            X_external = np.asarray(X_external, dtype=np.float64)
            n_samples = X_external.shape[0]
            n_features = X_external.shape[1]
            # Standardize to mean 0, std 1 (matching the Gaussian assumption
            # of edge weights and sigmoid sink steepness).
            X_mean = X_external.mean(axis=0)
            X_std = X_external.std(axis=0)
            X_std = np.maximum(X_std, 1e-8)  # guard against constant features
            X_external = (X_external - X_mean) / X_std
            # Store for MC bootstrap resampling
            self._observed_roots = X_external.copy()
        else:
            self._observed_roots = None
            # Sample parameters
            n_features_low, n_features_high = self.n_features_range
            if n_features_low == n_features_high:
                n_features = n_features_low
            else:
                n_features = self._rng.integers(n_features_low, n_features_high)

            n_samples_low, n_samples_high = self.n_samples_range
            if n_samples_low == n_samples_high:
                n_samples = n_samples_low
            else:
                n_samples = self._rng.integers(n_samples_low, n_samples_high)
        n_dags_low, n_dags_high = self.n_dags_range
        if n_dags_low == n_dags_high:
            n_dags = n_dags_low
        else:
            n_dags = self._rng.integers(n_dags_low, n_dags_high)

        # Generate DAG(s)
        if n_dags == 1:
            nodes_low, nodes_high = self.nodes_per_dag_range
            n_nodes = (
                nodes_low if nodes_low == nodes_high else self._rng.integers(nodes_low, nodes_high)
            )
            dag = self._create_dag(n_nodes, self._rng.uniform(*self.edge_prob_range))
        else:
            dags = []
            for _ in range(n_dags):
                nodes_low, nodes_high = self.nodes_per_dag_range
                n_nodes = (
                    nodes_low
                    if nodes_low == nodes_high
                    else self._rng.integers(nodes_low, nodes_high)
                )
                sub_dag = self._create_dag(n_nodes, self._rng.uniform(*self.edge_prob_range))
                dags.append(sub_dag)
            dag = dags[0]
            for next_dag in dags[1:]:
                dag = self._join_dags(dag, next_dag, self._rng.uniform(*self.edge_prob_range))

        # Augment with synthetic root nodes if the topology generator
        # produced fewer roots than requested. This honors the
        # n_features_range config option so X.shape[1] actually
        # matches what the user asked for. Each synthetic root is
        # an independent direct predecessor of the sink with a
        # small weight in [-0.5, 0.5].
        dag, _ = self._ensure_n_root_features(dag, n_features)

        # Generate data from DAG
        df, node_activations, feature_types, sink_node, mlp_params = self._generate_data_from_dag(
            dag, n_samples, X_external=X_external
        )

        if not include_shap:
            # Return only the generative elements + the SCM recipe; attribution
            # is left to DoShapley (via ``compute_shap`` or ``from_scm``).
            scm = StructuralCausalModel(
                dag=dag,
                sink_node=sink_node,
                node_activations=node_activations,
                feature_types=feature_types,
                mlp_params=mlp_params,
                root_marginals=dict(self._root_marginals),
                observed_roots=(
                    None if self._observed_roots is None else self._observed_roots.copy()
                ),
                root_marginal_map=_beta_map_from_normal,
                rng=self._rng,
            )
            X_out, y = extract_roots_and_target(dag, df, sink_node)
            return {"X": X_out, "y": y, "scm": scm}

        # Compute target and SHAP values (sink node excluded from X and SHAP)
        X_out, y, p_hat, exact_shap, linear_shap = self._compute_target_and_shap(
            dag, df, node_activations, feature_types, sink_node, mlp_params
        )

        return {
            "X": X_out.astype(np.float32),
            "y": y,
            "p_hat": p_hat,
            "exact_shap": exact_shap,
            "linear_shap": linear_shap,
        }

    def compute_shap(self, dataset):
        """Compute do-Shapley attributions for a dataset from :meth:`generate_dataset`.

        Args:
            dataset: Either the dict returned by
                ``generate_dataset(include_shap=False)`` (keys ``X``, ``y``,
                ``scm``) or a bare
                :class:`~diffusionexplainerpfn.data.scm.StructuralCausalModel`.
                When an SCM is passed, ``dataset["X"]``/``dataset["y"]`` are
                required alongside it.

        Returns:
            dict with keys 'X', 'y', 'p_hat', 'exact_shap', 'linear_shap',
            matching ``generate_dataset(include_shap=True)``. When called
            immediately after ``generate_dataset(include_shap=False)`` (no
            intervening RNG use) the result is bit-identical to the combined
            call.
        """
        if isinstance(dataset, StructuralCausalModel):
            raise TypeError(
                "compute_shap needs the root features too; pass the dict returned "
                "by generate_dataset(include_shap=False) (or a (scm, X, y) triple) "
                "rather than a bare SCM."
            )
        scm = dataset["scm"]
        X = np.asarray(dataset["X"], dtype=np.float64)
        y = np.asarray(dataset["y"], dtype=np.float32)

        _, _, p_hat, exact_shap, linear_shap = self._build_shap_computer(scm).compute_from_X(
            scm.dag,
            X,
            y,
            scm.node_activations,
            scm.feature_types,
            scm.sink_node,
            scm.mlp_params,
        )
        return {
            "X": X.astype(np.float32),
            "y": y,
            "p_hat": p_hat,
            "exact_shap": exact_shap,
            "linear_shap": linear_shap,
        }
