"""do-Shapley attribution computation over known structural causal models.

Adapted from DiffusionExplainerPFN @ commit d5f8258 (branch
``refactor/subpackage-restructure``), module
``diffusionexplainerpfn/attributions/do_shapley.py``. Changes relative to
upstream:

- intra-package import flattened (``.dag_value_torch`` already local);
- the Tier-3 permutation-Monte-Carlo SHAP estimator (``_monte_carlo_shap``)
  and the ``mc_shap_K`` / ``mc_v_samples`` constructor knobs are removed.
  Selecting ``shap_method="monte_carlo"`` or handing the code a dataset with
  more than 15 root features raises ``ValueError``. This copy supports the
  exact-label regimes only (d <= 15): closed-form, coalition enumeration, and
  the irreducible-sets algorithm. Introducing MC labels at d > 15 is
  DiffusionExplainerPFN's scaling contribution, not ExplainerPFN's;
- a couple of dead local bindings (an unused import, ``node_to_idx``,
  ``device``) were dropped to keep the project lint-clean. No behavioral
  change.

Note on "exact": the Shapley *formulas* are exact, but the value function
``v(S) = E[Y | do(X_S = x_S)]`` is still estimated by Monte Carlo integration
over off-coalition root draws (``mc_samples``, default 100). That inner
integration is intentional and unrelated to the removed permutation-MC SHAP
estimator.

``DoShapley`` computes exact do-Shapley values for the root features of a
DAG-generated synthetic dataset, using the strategy originally embedded in
:class:`~explainerpfn.train.dag_generators.DAGGenerator`:

1. Linear DAGs -> closed-form path weights (Tier 1).
2. Nonlinear DAGs -> coalition enumeration (d <= 10) or the irreducible-sets
   algorithm (10 < d <= 15) (Tier 2, Witter et al., 2026).

The GPU path batches ``v(S)`` evaluations via
:mod:`explainerpfn.train.dag_value_torch`.

Data generation and attribution are decoupled: the computer receives the RNG,
the per-dataset root marginals, the observed-roots buffer, the root-value
sampler and the normal->Beta marginal map from the data generator, so the SHAP
value function integrates against exactly the marginal the data was sampled
from.
"""

from __future__ import annotations

import logging
from itertools import combinations
from math import factorial
from typing import Dict

import networkx as nx
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from .dag_value_torch import (
    DagValueMetadata,
    _sample_root_values_torch,
    evaluate_v_batched_chunked_torch,
)

logger = logging.getLogger(__name__)


class DoShapley:
    """Compute do-Shapley values from a known DAG structure.

    Args:
        rng: The data generator's ``numpy.random.Generator``. Shared so the
            Monte Carlo tiers consume the same seed stream as during data
            generation (bit-identical to the pre-extraction code).
        generate_feature_value: ``(n_samples, feature_type) -> ndarray``
            root-value sampler from the data generator.
        root_marginal_map: ``(z, spec) -> ndarray`` normal->Beta marginal map
            from the data generator.
        observed_roots: ``(n_samples, n_features)`` empirical roots for
            bootstrap resampling, or ``None`` for purely synthetic data.
        root_marginals: ``{root_node: beta_spec}`` for roots with a Beta
            marginal, else ``{}``.
        shap_method: ``"auto"`` | ``"closed_form"`` |
            ``"coalition_enumeration"`` | ``"irreducible_sets"``. The
            upstream ``"monte_carlo"`` option is not available in this copy
            (raises ``ValueError``).
        mc_samples: Number of MC draws per ``v(S)`` evaluation. This is the
            inner value-function integration; it is not the removed
            permutation-MC SHAP estimator.
        use_torch_shap: Enable the batched GPU path when CUDA is available.
        shap_device: ``torch.device`` used for the GPU path.
    """

    def __init__(
        self,
        rng,
        generate_feature_value,
        root_marginal_map,
        observed_roots=None,
        root_marginals=None,
        shap_method: str = "auto",
        mc_samples: int = 100,
        use_torch_shap: bool = True,
        shap_device=None,
    ):
        if shap_method == "monte_carlo":
            raise ValueError(
                "shap_method='monte_carlo' is not supported in ExplainerPFN: "
                "training labels must use exact Shapley formulas (d <= 15). "
                "Permutation-MC labels at d > 15 are DiffusionExplainerPFN's "
                "scaling mechanism."
            )
        self._rng = rng
        self._generate_feature_value = generate_feature_value
        self._root_marginal_map = root_marginal_map
        self._observed_roots = observed_roots
        self._root_marginals = root_marginals if root_marginals is not None else {}
        self.shap_method = shap_method
        self.mc_samples = mc_samples
        self.use_torch_shap = use_torch_shap
        self.shap_device = shap_device if shap_device is not None else torch.device("cpu")

    @classmethod
    def from_scm(cls, scm, **shap_config):
        """Build a computer bound to a :class:`~diffusionexplainerpfn.data.scm.StructuralCausalModel`.

        The SCM supplies the RNG, the root-value sampler, the normal->Beta
        marginal map, the per-dataset Beta marginal specs and the observed-roots
        buffer. Any do-Shapley configuration keyword accepted by ``__init__``
        (``shap_method``, ``mc_samples``, ``use_torch_shap``, ``shap_device``)
        may be passed as a keyword and overrides the default.

        This is the standalone attributions-side entry point: given an SCM and
        a root-feature matrix ``X`` (plus target ``y``), call
        :meth:`compute_from_X`.
        """
        root_marginal_map = scm.root_marginal_map
        if root_marginal_map is None and scm.root_marginals:
            raise ValueError(
                "SCM carries Beta root_marginals but no root_marginal_map; "
                "set scm.root_marginal_map (e.g. the generator's normal->Beta map)."
            )
        return cls(
            rng=scm.rng,
            generate_feature_value=scm.sample_feature_value,
            root_marginal_map=root_marginal_map,
            observed_roots=scm.observed_roots,
            root_marginals=scm.root_marginals,
            **shap_config,
        )

    # ========================================================================
    # Tier 1: Closed-Form SHAP for Linear DAGs
    # ========================================================================

    def _closed_form_shap(self, dag, X, sink_node, root_to_X_col=None):
        """Compute exact SHAP values for linear DAGs using path weights.

        For linear activations with independent root features:
            phi_j = w_j * (x_j - E[x_j])
        where w_j = sum of (product of edge weights along each path from j to Y).

        For NON-root nodes (intermediates), the SHAP value is identically zero
        (they have no path to themselves → no causal effect on the prediction
        independent of their parents). Only root features appear in X.

        Args:
            dag: NetworkX DiGraph with 'weight' edge attributes
            X: Feature matrix (n_samples, n_features) — root features only
            sink_node: Target node ID
            root_to_X_col: Optional dict mapping root node ID → X column index.
                          If None, assume X columns are in topological order
                          (preserves backward compat for direct callers).

        Returns:
            shap_values: (n_samples, n_features) exact Shapley values for roots
        """
        n_samples, n_features = X.shape
        topo_order = list(nx.topological_sort(dag))
        root_nodes = [n for n in topo_order if dag.in_degree(n) == 0]

        # Default: X columns are in topological order, roots only.
        # (Matches our new convention — root_to_X_col is the standard now,
        # but we accept the legacy convention for direct callers/tests.)
        if root_to_X_col is None:
            root_to_X_col = {node: i for i, node in enumerate(root_nodes)}

        # Compute total causal effect w_j for each root node
        shap_values = np.zeros_like(X)
        for root in root_nodes:
            col = root_to_X_col[root]
            if col >= n_features:
                continue  # extra roots beyond X columns (shouldn't happen)

            # Find all simple paths from root to sink
            try:
                paths = list(nx.all_simple_paths(dag, root, sink_node))
            except nx.NetworkXNoPath:
                total_effect = 0.0
                shap_values[:, col] = 0.0
                continue

            # Sum of path weight products
            total_effect = 0.0
            for path in paths:
                path_product = 1.0
                for i in range(len(path) - 1):
                    edge_weight = dag[path[i]][path[i + 1]]["weight"]
                    path_product *= edge_weight
                total_effect += path_product

            # SHAP: phi_root = w_root * (x_root - E[x_root])
            feature_mean = X[:, col].mean()
            shap_values[:, col] = total_effect * (X[:, col] - feature_mean)

        return shap_values

    def _evaluate_value_function(
        self,
        dag,
        X,
        node_values_base,
        S,
        sink_node,
        node_activations,
        feature_types,
        mlp_params,
        n_mc_samples=None,
        root_to_X_col=None,
    ):
        """Evaluate v(S) = E[Y | do(X_S = x_S)] using vectorized MC sampling.

        Intervention: set ROOT features in S to their observed values,
        sample remaining root features from their natural distributions,
        propagate through structural equations, return expected Y.

        X contains ONLY root features (intermediate nodes are not intervenable).
        S is a set of ROOT node IDs. root_to_X_col maps root IDs to X column indices.

        Vectorized: all MC samples are processed simultaneously using numpy,
        eliminating the Python loop over MC iterations.
        """
        if n_mc_samples is None:
            n_mc_samples = self.mc_samples

        n_samples = X.shape[0]
        topo_order = list(nx.topological_sort(dag))
        root_nodes = set(n for n in topo_order if dag.in_degree(n) == 0)

        # Build root → X column mapping (default: roots in topo order, no intermediates)
        if root_to_X_col is None:
            sorted_roots = sorted(root_nodes)
            root_to_X_col = {r: i for i, r in enumerate(sorted_roots)}

        # Precompute adjacency: for each non-root node, its parents and weights
        # This avoids repeated dict lookups in the inner loop
        parent_weights = {}
        for node in topo_order:
            if node not in root_nodes:
                parents = list(dag.predecessors(node))
                parent_weights[node] = [(p, dag[p][node]["weight"]) for p in parents]

        # Vectorized MC: process all samples simultaneously
        # node_vals shape: (n_mc_samples, n_samples) per node
        node_vals = {}

        for node in topo_order:
            if node in root_nodes:
                # Root node: intervene (use observed X value) or sample fresh
                col = root_to_X_col.get(node, -1)
                if node in S and col >= 0 and col < X.shape[1]:
                    # Intervened: broadcast observed value across MC samples
                    node_vals[node] = np.tile(X[:, col], (n_mc_samples, 1))  # (n_mc, n_samples)
                else:
                    # Not intervened root: sample from natural distribution.
                    # When observed roots are available (external X injection),
                    # bootstrap-resample from the empirical column instead of
                    # assuming a Gaussian distribution. This ensures correct
                    # interventional expectations for non-Gaussian real data.
                    if (
                        self._observed_roots is not None
                        and col >= 0
                        and col < self._observed_roots.shape[1]
                    ):
                        obs_col = self._observed_roots[:, col]
                        idx = self._rng.integers(0, len(obs_col), size=n_samples * n_mc_samples)
                        vals = obs_col[idx]
                    else:
                        # Beta base marginals: draw a normal variate (same
                        # RNG stream as the legacy Gaussian path) and map it
                        # through Phi -> Beta inverse CDF so the labels
                        # integrate against the same marginal as the data.
                        spec = self._root_marginals.get(node)
                        if spec is not None:
                            z = self._rng.standard_normal(n_samples * n_mc_samples)
                            vals = self._root_marginal_map(z, spec)
                        else:
                            vals = self._generate_feature_value(
                                n_samples * n_mc_samples, feature_types[node]
                            )
                    node_vals[node] = vals.reshape(n_mc_samples, n_samples)
            else:
                # Non-root: compute from structural equation (vectorized across MC samples)
                parents = parent_weights.get(node, [])
                if not parents:
                    vals = self._generate_feature_value(
                        n_samples * n_mc_samples, feature_types.get(node, "numerical")
                    )
                    node_vals[node] = vals.reshape(n_mc_samples, n_samples)
                    continue

                weighted_sum = np.zeros((n_mc_samples, n_samples))
                for parent, weight in parents:
                    weighted_sum += node_vals[parent] * weight

                # Apply activation
                activation_name = node_activations.get(node, "linear")
                if activation_name == "linear":
                    result = weighted_sum
                elif activation_name == "relu":
                    result = np.maximum(0, weighted_sum)
                elif activation_name == "tanh":
                    result = np.tanh(weighted_sum)
                elif activation_name == "sin":
                    result = np.sin(weighted_sum)
                elif activation_name == "soft_interaction":
                    result = weighted_sum * (1 + np.tanh(weighted_sum))
                elif activation_name == "gaussian":
                    result = np.exp(-(weighted_sum**2))
                elif activation_name.startswith("sigmoid_c"):
                    # Extract steepness from name: sigmoid_c2.15 → c = 2.15
                    c = float(activation_name.split("c")[1])
                    result = 1.0 / (1.0 + np.exp(-c * weighted_sum))
                elif activation_name.startswith("mlp_"):
                    # MLP forward pass: stack parent values, apply W1/tanh/W2
                    n_parents, W1, b1, W2, b2 = mlp_params[node]
                    # Stack parent values: (n_mc_samples, n_samples, n_parents)
                    parent_stack = np.stack([node_vals[p] for p, _ in parents], axis=-1)
                    hidden = np.tanh(parent_stack @ W1 + b1)  # (n_mc, n_samples, n_hidden)
                    result = hidden @ W2 + b2  # (n_mc, n_samples)
                else:
                    result = weighted_sum

                # NOTE: No noise added during exact SHAP computation.
                # Noise was already added during data generation. Adding it
                # again in the value function creates randomness that violates
                # the efficiency axiom (v(full) ≠ y). The value function should
                # compute the DETERMINISTIC expected output given the intervention.
                node_vals[node] = result

        # Average over MC samples: (n_mc_samples, n_samples) -> (n_samples,)
        return node_vals.get(sink_node, np.zeros((n_mc_samples, n_samples))).mean(axis=0)

    def _compute_shapley_weight(self, feature_idx, basis_size, closure_size, d):
        """Compute the Shapley class weight w_i(c) in O(d) time.

        From Witter et al., 2026, Equation 5:

        For a class c with basis S and closure S̄:

        w_i(c) = Σ_{ℓ=|S|}^{|S̄|} p_{ℓ-1} · C(|S̄|-|S|, ℓ-|S|)     if i ∈ S
        w_i(c) = -Σ_{ℓ=|S|}^{|S̄|} p_ℓ · C(|S̄|-|S|, ℓ-|S|)        if i ∉ S̄
        w_i(c) = 0                                                   else

        where p_ℓ = 1 / (d · C(d-1, ℓ))

        Args:
            feature_idx: Index of the feature i (unused but kept for API clarity)
            basis_size: |S| (size of the basis)
            closure_size: |S̄| (size of the closure)
            d: Total number of features

        Returns:
            Tuple of (weight_for_in_basis, weight_for_not_in_closure)
        """
        n_free = closure_size - basis_size

        weight_in_basis = 0.0
        weight_not_in_closure = 0.0

        for j in range(n_free + 1):
            s = basis_size + j
            binom_free = self._binom(n_free, j)

            # Weight for i ∈ S: Σ p_{ℓ-1} · C(n_free, ℓ-|S|)
            if s > 0 and s - 1 < d:
                p_s_minus_1 = 1.0 / (d * self._binom(d - 1, s - 1))
                weight_in_basis += p_s_minus_1 * binom_free

            # Weight for i ∉ S̄: -Σ p_ℓ · C(n_free, ℓ-|S|)
            if s < d:
                p_s = 1.0 / (d * self._binom(d - 1, s))
                weight_not_in_closure -= p_s * binom_free

        return weight_in_basis, weight_not_in_closure

    @staticmethod
    def _binom(n, k):
        """Compute binomial coefficient C(n, k)."""
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        k = min(k, n - k)
        result = 1
        for i in range(k):
            result = result * (n - i) // (i + 1)
        return result

    def _find_all_closed_sets(self, dag, sink_node, feature_ids=None):
        """Find all closed sets using the AllClasses algorithm (Algorithm 2).

        A closed set S̄ is one where every node not in S̄ has a directed path
        to Y that doesn't intersect S̄.

        Args:
            dag: NetworkX DiGraph
            sink_node: Target node ID
            feature_ids: Optional set/list of feature node IDs to enumerate over.
                         If None, uses all DAG node IDs (legacy).
                         If provided, only enumerates closed sets that are subsets
                         of feature_ids, which is the correct setting for SHAP over
                         root features only.

        Returns:
            list of frozensets, each representing a closed set
        """
        d = len(dag.nodes)

        # If feature_ids provided, restrict enumeration to subsets of those.
        # This is the correct setting for SHAP over root features.
        if feature_ids is not None:
            feature_ids_set = set(feature_ids)
            universe = feature_ids_set
        else:
            universe = set(range(d))

        # Precompute adjacency lists once for the whole algorithm. The
        # previous implementation called ``G_prime = dag.copy()`` and
        # ``nx.has_path(...)`` inside the inner FindClass loop, which
        # is O(V * (V + E)) per call. We replace it with a single
        # reverse BFS per call, O(V + E).
        parents_of, _children_of = self._build_dag_adjacency(dag)

        # FindClass subroutine — pure set arithmetic, no NetworkX.
        def find_class(S_set):
            """Given a set S of node IDs, find its basis and closure.

            S_set contains DAG node IDs directly.
            """
            # Restrict S to nodes that actually exist in the DAG.
            nodes_in_S = set(S_set) & set(dag.nodes)
            g_prime_ancestors = self._ancestors_of_sink_in_dag_minus_S(
                sink_node, parents_of, nodes_in_S
            )

            # Restrict to universe (feature_ids if provided, else all nodes)
            non_ancestors = universe - g_prime_ancestors
            ancestor_in_universe = g_prime_ancestors & universe

            # Closure S̄ = S ∪ non-ancestors of Y (in the restricted universe)
            closure = frozenset(S_set | non_ancestors)
            # Basis = S ∩ ancestors of Y (in the restricted universe)
            basis = frozenset(S_set & ancestor_in_universe)

            return basis, closure

        # AllClasses algorithm (Algorithm 2)
        full_set = frozenset(universe)
        _, initial_closure = find_class(full_set)

        # BFS to find all closed sets
        closed_sets = set()
        queue = [initial_closure]
        visited = set()

        while queue:
            current_closure = queue.pop(0)
            if current_closure in visited:
                continue
            visited.add(current_closure)

            basis, closure = find_class(current_closure)
            closed_sets.add(closure)

            # For each node in the basis, remove it to get a new closed set
            for j in basis:
                new_set = frozenset(closure - {j})
                if new_set not in visited:
                    queue.append(new_set)

        return list(closed_sets)

    def _compute_shap_from_irreducible_sets(
        self, dag, X, sink_node, node_activations, feature_types, mlp_params, root_to_X_col=None
    ):
        """Compute exact SHAP values using irreducible sets (Witter et al., 2026).

        After finding all closed sets c_1, ..., c_r (over ROOT features only):
            phi_i = sum_{j=1}^{r} v(c_j) * w_i(c_j)

        The class weight w_i(c_j) is computed in O(d) time using Equation 5:
        - If i ∈ basis: w_i = Σ p_{ℓ-1} · C(n_free, ℓ-|S|)
        - If i ∉ closure: w_i = -Σ p_ℓ · C(n_free, ℓ-|S|)
        - Else: w_i = 0

        We compute SHAP for ROOT features only (d = n_roots). The Witter
        algorithm requires enumeration over all features, so we restrict the
        closed-set enumeration to root node IDs.

        Total runtime: O(r(d + e + T)) where T = cost of one v() evaluation.

        Stage 4 dispatch:
            * When ``self.use_torch_shap`` is True, the GPU is available,
              and the workload is large enough (R * n_mc * n_samples
              exceeds ~1e7), all v(c) calls are batched into a single
              GPU forward pass via ``dag_value_torch.py``.
            * Otherwise the numpy path is used (unchanged behaviour).

        Args:
            dag: NetworkX DiGraph
            X: Feature matrix (n_samples, n_features) — root features only
            sink_node: Target node
            node_activations: Dict mapping node -> activation name
            feature_types: Dict mapping node -> feature type
            root_to_X_col: Dict mapping root node ID → X column index

        Returns:
            shap_values: (n_samples, n_features) exact SHAP for root features
        """
        n_samples, n_features = X.shape
        shap_values = np.zeros_like(X)

        if root_to_X_col is None:
            topo_order = list(nx.topological_sort(dag))
            root_to_X_col = {n: i for i, n in enumerate(topo_order) if dag.in_degree(n) == 0}

        # Universe for closed-set enumeration: root node IDs only
        feature_ids = sorted(root_to_X_col.keys())
        d = len(feature_ids)  # SHAP dimension = number of root features

        # Find all closed sets (equivalence classes) — over root features only
        closed_sets = self._find_all_closed_sets(dag, sink_node, feature_ids=feature_ids)
        r = len(closed_sets)

        # If too many closed sets, fall back to exact coalition
        # enumeration. At d <= 15 this branch is structurally unreachable
        # (r <= 2^d <= 2^15 = 32768 < 100000), but it is kept as a safe
        # exact fallback. (Upstream routed d > 15 to Monte Carlo here.)
        if r > 100000:
            return self._enumerate_coalitions_shap(
                dag,
                X,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
            )

        # Stage 4: decide whether to use the GPU path.
        n_mc = self.mc_samples
        gpu_workload = r * n_mc * max(n_samples, 1)
        # Threshold 0 = "always use GPU when available" (per user
        # decision: even small workloads benefit from the chunked
        # GPU forward pass over the per-call CPU loop, because the
        # CPU loop is also pure Python with non-trivial overhead per
        # call). If we find this is wrong for very small d in
        # practice, we can re-introduce a non-zero threshold.
        GPU_THRESHOLD = 0
        use_gpu = (
            self.use_torch_shap
            and self.shap_device.type == "cuda"
            and torch.cuda.is_available()
            and gpu_workload >= GPU_THRESHOLD
        )

        if use_gpu:
            return self._compute_shap_from_irreducible_sets_torch(
                dag,
                X,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
                closed_sets=closed_sets,
                feature_ids=feature_ids,
                d=d,
            )

        # For each closed set, compute v(c) and add its weighted contribution
        for c_bar in closed_sets:
            # Find the basis of this closed set (in the full DAG, not just roots)
            basis = self._find_class_basis(dag, c_bar, sink_node)
            closure_size = len(c_bar)
            basis_size = len(basis)

            # Evaluate v(c) = E[Y | do(X_S = x_S)] where S = basis (root IDs only)
            v_c = self._evaluate_value_function(
                dag,
                X,
                {},
                basis,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
            )

            # Compute Shapley weights using Equation 5 (O(d) per feature, d = n_roots)
            w_in_basis, w_not_in_closure = self._compute_shapley_weight(
                None, basis_size, closure_size, d
            )

            # Apply weights to root features (the SHAP universe is now root IDs)
            for i in feature_ids:
                col = root_to_X_col[i]
                if i in basis:
                    shap_values[:, col] += w_in_basis * v_c
                elif i not in c_bar:
                    shap_values[:, col] += w_not_in_closure * v_c
                # else: i in closure but not in basis → weight = 0

        return shap_values

    def _batched_v_for_coalitions(
        self,
        dag,
        X,
        sink_node,
        node_activations,
        feature_types,
        mlp_params,
        root_to_X_col,
        coalitions,
        max_chunk_r=64,
        n_mc=None,
    ):
        """Shared GPU helper used by both coalition_enumeration (Tier 2a)
        and irreducible_sets (Tier 2b) for batched v(S) evaluation.

        Args:
            dag: NetworkX DiGraph.
            X: (n_samples, n_roots) root feature values, numpy float64.
            sink_node: Target node ID.
            node_activations: Dict mapping node -> activation name.
            feature_types: Dict mapping node -> feature type.
            mlp_params: Dict mapping node -> (n_parents, W1, b1, W2, b2) for MLP nodes.
            root_to_X_col: Dict mapping root node ID -> X column index.
            coalitions: Iterable of coalitions. Each coalition is a
                set/frozenset of root node IDs (the ones to mark as
                "intervened" in the mask). May also be a list of
                frozensets of arbitrary node IDs, as long as each
                node in the coalition has an entry in
                ``root_to_X_col`` (the helper silently ignores nodes
                that are not root-mapped, so callers can pass e.g.
                the basis sets produced by ``_find_class_basis``).
            max_chunk_r: Chunk size for the GPU forward pass. Memory
                scales as ``O(max_chunk_r * n_mc * n_samples *
                n_parents)``. For 12 GB VRAM and (n_mc=200,
                n_samples=2000, n_parents=4), use max_chunk_r=8
                for 2^15 enumeration workloads.
            n_mc: Number of MC samples for the v() estimator. If
                ``None`` (default), uses ``self.mc_samples`` (which
                is typically 200). The Monte Carlo SHAP tier passes
                a smaller value (~50) here to keep wall-time bounded
                for d=15-30, matching the numpy MC SHAP fallback.

        Returns:
            v_cache: dict mapping ``frozenset(coalition)`` to
                ``(n_samples,)`` numpy float32 array — v(S) for each
                coalition. Keys are exactly the values in
                ``coalitions`` (as frozensets).
        """
        # Wrap the entire GPU body in torch.no_grad() so the
        # autograd graph is never built (SHAP values are
        # targets-of-comparison, not optimisation parameters).
        # This is a small per-iteration CPU/memory saving, but the
        # main benefit is that ``empty_cache`` below can release
        # ALL of the intermediate buffers without us worrying about
        # stale graph nodes.
        with torch.no_grad():
            n_samples, _ = X.shape
            if n_mc is None:
                n_mc = self.mc_samples
            device = self.shap_device

            # Materialize coalitions as a list of frozensets (and remember
            # the original keys for the output dict).
            coal_list = [frozenset(c) for c in coalitions]
            n_coal = len(coal_list)
            n_roots = len(root_to_X_col)

            # Build the (R, n_roots) coalition mask. Roots not present
            # in a coalition are sampled from mc_X; roots present are
            # set to their observed value X[:, root_to_X_col[root]].
            # Roots not in root_to_X_col are ignored (defensive: callers
            # may pass sets that include non-root nodes, e.g. the
            # basis from _find_class_basis, which may include
            # non-root ancestors of the sink).
            col_for_root = root_to_X_col  # alias for clarity below
            coalition_mask_np = np.zeros((n_coal, n_roots), dtype=bool)
            for j, S in enumerate(coal_list):
                for root_id in S:
                    if root_id in col_for_root:
                        coalition_mask_np[j, col_for_root[root_id]] = True

            # Build metadata once.
            meta_cpu = DagValueMetadata.from_dag(
                dag, sink_node, node_activations, mlp_params, root_to_X_col
            )
            meta = meta_cpu.to(device)

            # Sample mc_X on the device. (n_mc, n_samples, n_roots).
            feature_types_root = [feature_types[meta.topo_order[i]] for i in meta.root_indices]
            # Align the per-root marginal spec with the torch root-index
            # order (specs are keyed by node id from the data-generation
            # walk; None for Gaussian/identity roots).
            root_marginals = [
                self._root_marginals.get(meta.topo_order[i]) for i in meta.root_indices
            ]
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self._rng.integers(0, 2**31)))
            mc_X = _sample_root_values_torch(
                feature_types_root,
                n_mc,
                n_samples,
                n_roots,
                device,
                gen,
                rng=self._rng,
                observed_roots=self._observed_roots,
                root_marginals=root_marginals,
            )

            # Move X to GPU.
            X_gpu = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
            mask = torch.from_numpy(coalition_mask_np).to(device)

            # Single (chunked) GPU forward pass: returns (R, n_samples).
            v_all = (
                evaluate_v_batched_chunked_torch(meta, X_gpu, mask, mc_X, max_chunk_r=max_chunk_r)
                .detach()
                .cpu()
                .numpy()
            )

            # Build the output dict, keyed by frozenset(coalition) so
            # callers can look up by the same key they passed in.
            v_cache: Dict = {}
            for j, S in enumerate(coal_list):
                v_cache[S] = v_all[j]

        # Force the PyTorch CUDA caching allocator to release the
        # free pool back to the OS. Without this, the peak per-call
        # allocation (mc_X + X_gpu + mask + meta + chunked node_vals)
        # accumulates across the 4 datasets/iter of v25 pretraining
        # and the process OOMs after a few hundred iterations. The
        # cost is a single allocator sweep (~1 ms); the alternative
        # (relying on refcount + leaving the cache to grow) was the
        # original OOM root cause.
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return v_cache

    def _compute_shap_from_irreducible_sets_torch(
        self,
        dag,
        X,
        sink_node,
        node_activations,
        feature_types,
        mlp_params,
        root_to_X_col,
        closed_sets,
        feature_ids,
        d,
    ):
        """Stage-4 GPU implementation of the irreducible-sets SHAP path.

        Batches the r v(c) calls into a single GPU forward pass. The
        per-coalition cost is ~0.1-0.5 ms on RTX 5070 Ti versus ~5-20 ms
        in the numpy path, so for r=1000 the speedup is 100-500x.

        The math is identical to the numpy path, but with a
        variance-reduction trick: mc_X is sampled once and shared
        across all r coalitions. This is a standard SHAP variance
        reduction technique; the estimator remains unbiased because
        the v(S) expectation is over the MC distribution, not over
        the coalition set.
        """
        # Get v(c) for every closed set's basis, in one GPU pass.
        coalitions = [self._find_class_basis(dag, c_bar, sink_node) for c_bar in closed_sets]
        v_cache = self._batched_v_for_coalitions(
            dag,
            X,
            sink_node,
            node_activations,
            feature_types,
            mlp_params,
            root_to_X_col,
            coalitions,
            max_chunk_r=64,
        )

        # Apply the Shapley weights. This part is still numpy
        # (it's pure scalar arithmetic on r closed sets, O(r * d) — cheap).
        shap_values = np.zeros_like(X)
        for j, c_bar in enumerate(closed_sets):
            basis = self._find_class_basis(dag, c_bar, sink_node)
            closure_size = len(c_bar)
            basis_size = len(basis)
            w_in_basis, w_not_in_closure = self._compute_shapley_weight(
                None, basis_size, closure_size, d
            )
            v_c = v_cache[frozenset(basis)]
            for i in feature_ids:
                col = root_to_X_col[i]
                if i in basis:
                    shap_values[:, col] += w_in_basis * v_c
                elif i not in c_bar:
                    shap_values[:, col] += w_not_in_closure * v_c
        return shap_values

    @staticmethod
    def _build_dag_adjacency(dag):
        """Precompute parent/child adjacency lists for a DAG.

        Returns ``(parents_of, children_of)`` where each is a dict
        ``{node: [list of neighbours]}``. Cost: O(V + E). Built once
        per SHAP computation and reused across every FindClass /
        basis query in the inner loop, which is the hot path.
        """
        parents_of = {v: [] for v in dag.nodes}
        children_of = {v: [] for v in dag.nodes}
        for u, v in dag.edges:
            children_of[u].append(v)
            parents_of[v].append(u)
        return parents_of, children_of

    @staticmethod
    def _ancestors_of_sink_in_dag_minus_S(sink_node, parents_of, nodes_in_S):
        """Set of DAG nodes v such that v can reach ``sink_node`` in
        ``dag`` *with the incoming edges of every node in
        ``nodes_in_S`` removed*.

        Computed by a single reverse BFS from ``sink_node`` that
        treats nodes in ``nodes_in_S`` as forbidden. Cost: O(V + E).
        This replaces the previous NetworkX
        ``G_prime = dag.copy(); ...; nx.has_path(G_prime, v, sink)``
        pattern, which was O(V * (V + E)) per FindClass call.
        """
        ancestors = set()
        stack = [sink_node]
        forbidden = nodes_in_S
        while stack:
            v = stack.pop()
            if v in ancestors:
                continue
            ancestors.add(v)
            # v is forbidden iff it is in nodes_in_S. If v is forbidden
            # then the edge from any parent p -> v has been removed
            # in G', so v's parents cannot reach v through the removed
            # edge. We do NOT recurse into v's parents in that case.
            if v not in forbidden:
                for p in parents_of[v]:
                    if p not in ancestors:
                        stack.append(p)
        return ancestors

    def _find_class_basis(self, dag, S_bar, sink_node):
        """Find the basis of a closed set S_bar.

        The basis is the set of nodes in S_bar that are still
        connected to Y after intervention.
        """
        # S_bar contains DAG node IDs directly.
        S_bar_set = set(S_bar) if not isinstance(S_bar, frozenset) else set(S_bar)
        # Restrict to nodes actually in the DAG (defensive)
        nodes_in_S = S_bar_set & set(dag.nodes)

        # Precompute once (constant in S_bar); this is the work that
        # the old G_prime = dag.copy() did per call.
        parents_of, _children_of = self._build_dag_adjacency(dag)
        ancestors_of_Y = self._ancestors_of_sink_in_dag_minus_S(sink_node, parents_of, nodes_in_S)

        # Basis = S_bar ∩ ancestors of Y
        basis = S_bar_set & ancestors_of_Y

        return frozenset(basis)

    def _enumerate_coalitions_shap(
        self, dag, X, sink_node, node_activations, feature_types, mlp_params, root_to_X_col=None
    ):
        """Compute exact SHAP by enumerating all 2^d coalitions.

        Only feasible for d ≤ ~15-20. Operates on ROOT features (d = n_features).
        S is a set of root node IDs; X has one column per root.

        Stage 4 dispatch: when CUDA is available and ``use_torch_shap``
        is True, the 2^d v(S) calls are batched into a single GPU
        forward pass via ``_batched_v_for_coalitions``. For d=8-15
        (the v25 hot path) this is 30-300x faster than the per-call
        CPU loop. Below the ``GPU_THRESHOLD``, falls back to the
        numpy path.

        Args:
            dag: NetworkX DiGraph
            X: Feature matrix (n_samples, n_features) — root features only
            sink_node: Target node
            node_activations: Dict mapping node -> activation name
            feature_types: Dict mapping node -> feature type
            root_to_X_col: Dict mapping root node ID → X column index

        Returns:
            shap_values: (n_samples, n_features) exact SHAP for root features
        """
        n_samples, n_features = X.shape

        # Use root node IDs as coalition space (not column indices)
        # Coalition S ⊆ root_nodes
        topo_order = list(nx.topological_sort(dag))
        if root_to_X_col is None:
            root_to_X_col = {n: i for i, n in enumerate(topo_order) if dag.in_degree(n) == 0}
        root_ids = sorted(root_to_X_col.keys())  # canonical list of root IDs
        d = len(root_ids)

        # Stage 4 dispatch.
        n_mc = self.mc_samples
        gpu_workload = (1 << d) * n_mc * max(n_samples, 1)
        # Threshold 0 = "always use GPU when available" (per user
        # decision: even tiny d is faster on GPU than a 2^d Python loop).
        GPU_THRESHOLD = 0
        use_gpu = (
            self.use_torch_shap
            and self.shap_device.type == "cuda"
            and torch.cuda.is_available()
            and gpu_workload >= GPU_THRESHOLD
        )

        if use_gpu:
            return self._enumerate_coalitions_shap_torch(
                dag,
                X,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
                root_ids=root_ids,
                d=d,
            )

        # CPU path: precompute v(S) for all 2^d subsets of root IDs
        v_cache = {}
        for subset_size in range(d + 1):
            for subset in combinations(root_ids, subset_size):
                S = frozenset(subset)
                v_cache[S] = self._evaluate_value_function(
                    dag,
                    X,
                    {},
                    S,
                    sink_node,
                    node_activations,
                    feature_types,
                    mlp_params,
                    root_to_X_col=root_to_X_col,
                )

        # Compute Shapley values using the standard formula
        shap_values = np.zeros_like(X)
        for i in root_ids:
            col = root_to_X_col[i]
            other_roots = [j for j in root_ids if j != i]
            for subset_size in range(d):
                for subset in combinations(other_roots, subset_size):
                    S = frozenset(subset)
                    S_with_i = frozenset(subset + (i,))

                    s = len(S)
                    weight = factorial(s) * factorial(d - s - 1) / factorial(d)
                    marginal = v_cache[S_with_i] - v_cache[S]
                    shap_values[:, col] += weight * marginal

        return shap_values

    def _enumerate_coalitions_shap_torch(
        self,
        dag,
        X,
        sink_node,
        node_activations,
        feature_types,
        mlp_params,
        root_to_X_col,
        root_ids,
        d,
    ):
        r"""Stage-4 GPU implementation of the coalition_enumeration SHAP path.

        Batches all 2^d v(S) calls into a single (chunked) GPU forward
        pass via ``_batched_v_for_coalitions``. For d=8-15, this is
        30-300x faster than the per-call CPU loop.

        The math is identical to the CPU path:
            phi_i = sum_{S subset of roots\{i}} [|S|! (d-|S|-1)! / d!]
                                    * (v(S union {i}) - v(S))

        The variance-reduction trick (mc_X sampled once and shared
        across all 2^d coalitions) is the same as in the irreducible
        Sets tier; it's a standard SHAP variance-reduction technique.

        Memory: for d=15, 2^15=32768 coalitions. We chunk over the
        coalition axis with max_chunk_r=8 so peak memory stays at
        ~13 MB per chunk (well under 12 GB VRAM). Total of 4096
        GPU kernel launches per dataset for d=15; for smaller d the
        chunk count is correspondingly lower.
        """
        # Build the 2^d coalition list (frozensets of root IDs).
        coalitions = []
        for subset_size in range(d + 1):
            for subset in combinations(root_ids, subset_size):
                coalitions.append(frozenset(subset))

        # Single GPU call: returns v_cache[S] for every S.
        v_cache = self._batched_v_for_coalitions(
            dag,
            X,
            sink_node,
            node_activations,
            feature_types,
            mlp_params,
            root_to_X_col,
            coalitions,
            max_chunk_r=64,
        )

        # Standard Shapley marginal-contribution formula. The
        # v_cache dict is the only data structure the loop needs.
        shap_values = np.zeros_like(X)
        for i in root_ids:
            col = root_to_X_col[i]
            other_roots = [j for j in root_ids if j != i]
            for subset_size in range(d):
                for subset in combinations(other_roots, subset_size):
                    S = frozenset(subset)
                    S_with_i = frozenset(subset + (i,))
                    s = len(S)
                    weight = factorial(s) * factorial(d - s - 1) / factorial(d)
                    marginal = v_cache[S_with_i] - v_cache[S]
                    shap_values[:, col] += weight * marginal
        return shap_values

    def _compute_linear_shap(self, X, p_hat):
        """Compute LinearSHAP approximation from a linear surrogate model.

        Fits on standardized X and p_hat so beta depends only on correlation,
        not on raw feature scales. This ensures consistent linear_shap_norm
        scale across synthetic and real data.

        Args:
            X: Feature matrix (n_samples, n_features)
            p_hat: Model predictions (n_samples,)

        Returns:
            linear_shap: (n_samples, n_features) LinearSHAP approximation in original scale
        """
        n_samples, n_features = X.shape

        if n_samples < 2 or n_features < 1:
            return np.zeros_like(X)

        try:
            # Standardize X and p_hat (same as DatasetNormalizer)
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0).clip(min=0.5)
            X_norm = (X - X_mean) / X_std

            p_hat_mean = p_hat.mean()
            p_hat_std = max(p_hat.std(), 1e-8)
            p_hat_norm = (p_hat - p_hat_mean) / p_hat_std

            # Fit linear surrogate on normalized data
            reg = LinearRegression()
            reg.fit(X_norm, p_hat_norm)

            # Center normalized X for LinearSHAP computation
            X_norm_centered = X_norm - X_norm.mean(axis=0)

            # LinearSHAP in original scale: phi = beta_norm * X_norm_centered * p_hat_std
            # This gives SHAP values in the same units as p_hat (change in prediction).
            # The training pipeline will normalize by p_hat_std to get dimensionless space.
            linear_shap = X_norm_centered * reg.coef_ * p_hat_std

            return linear_shap.astype(np.float32)
        except Exception:
            return np.zeros_like(X, dtype=np.float32)

    # ========================================================================
    # Public API
    # ========================================================================

    def compute(self, dag, df, node_activations, feature_types, sink_node, mlp_params):
        """Compute target variable and SHAP explanations from known DAG structure.

        Thin wrapper around :meth:`compute_from_X` that extracts the root
        feature matrix ``X`` and target ``y`` from an all-node DataFrame (the
        representation produced by ``DAGGenerator``).

        Args:
            dag: NetworkX DiGraph
            df: DataFrame with generated data
            node_activations: Dict mapping node -> activation name
            feature_types: Dict mapping node -> feature type
            sink_node: Target node ID
            mlp_params: Dict mapping node -> MLP parameters

        Returns:
            X, y, p_hat, exact_shap, linear_shap
            (see :meth:`compute_from_X`)
        """
        # Build X using ONLY root-node columns. Each X column is identified
        # by its root node ID (the column position in X corresponds to the
        # root's position in topo order; SHAP methods index by root node ID
        # and we provide an explicit root_to_X_col mapping for clarity).
        topo_order = list(nx.topological_sort(dag))
        root_nodes = [n for n in topo_order if dag.in_degree(n) == 0]
        root_to_X_col = {node: i for i, node in enumerate(root_nodes)}

        # Build X as (n_samples, n_roots) array from df, using root columns
        X = np.zeros((df.shape[0], len(root_nodes)), dtype=np.float64)
        for root in root_nodes:
            col_name = f"ind_{root}" if f"ind_{root}" in df.columns else f"dep_{root}"
            X[:, root_to_X_col[root]] = df[col_name].values.astype(np.float64)

        # Target y = sink node value from the DAG
        sink_col = f"dep_{sink_node}" if f"dep_{sink_node}" in df.columns else f"ind_{sink_node}"
        if sink_col in df.columns:
            y = df[sink_col].values.astype(np.float32)
        else:
            # Fallback: last column in df
            y = df.values[:, -1].astype(np.float32)

        return self.compute_from_X(
            dag, X, y, node_activations, feature_types, sink_node, mlp_params
        )

    def compute_from_X(self, dag, X, y, node_activations, feature_types, sink_node, mlp_params):
        """Compute do-Shapley attributions for a precomputed root matrix.

        This is the attribution core shared by the DataFrame-based
        :meth:`compute` and the SCM-based :meth:`from_scm` entry point.

        Only ROOT nodes are used as features (X). Intermediate nodes are
        deterministic (or noisy) functions of their parents and are part of
        the model's mechanism — not input features. The sink node is the
        target and is likewise excluded.

        Args:
            dag: NetworkX DiGraph
            X: (n_samples, n_features) root-feature matrix (float64)
            y: (n_samples,) continuous target values (sink node)
            node_activations: Dict mapping node -> activation name
            feature_types: Dict mapping node -> feature type
            sink_node: Target node ID
            mlp_params: Dict mapping node -> MLP parameters

        Returns:
            X: (n_samples, n_features) input features (root nodes only)
            y: (n_samples,) continuous target values
            p_hat: (n_samples,) model predictions (= y for synthetic data)
            exact_shap: (n_samples, n_features) exact Shapley values
            linear_shap: (n_samples, n_features) LinearSHAP approximation
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float32)

        topo_order = list(nx.topological_sort(dag))
        root_nodes = [n for n in topo_order if dag.in_degree(n) == 0]
        root_to_X_col = {node: i for i, node in enumerate(root_nodes)}
        n_samples, n_features = X.shape

        # p_hat = y (for synthetic data, we have the true predictions)
        p_hat = y.copy()

        # Compute exact_shap using X = root features only.
        #    SHAP methods now operate on root-node IDs and read X[:, root_to_X_col[root]].
        #    S (intervention set) is a subset of root node IDs.
        all_linear = all(name == "linear" for name in node_activations.values())

        # Pass root_to_X_col so SHAP methods can map root IDs to X columns.
        if self.shap_method == "closed_form" or (self.shap_method == "auto" and all_linear):
            # Tier 1: Linear DAGs — closed-form
            exact_shap = self._closed_form_shap(
                dag, X, sink_node, root_to_X_col=root_to_X_col
            ).astype(np.float32)
            tier = "closed_form"
        elif self.shap_method == "coalition_enumeration" or (
            self.shap_method == "auto" and not all_linear and n_features <= 10
        ):
            # Small nonlinear DAGs — enumerate all 2^d coalitions.
            # Capped at d=10 because the GPU forward pass is O(2^d * V)
            # bytes, which becomes prohibitive for d=15 (32K
            # coalitions × 200 mc × 2000 samples × 42 nodes ≈ 142 GB
            # of compute, ~140s wall-time on RTX 5070 Ti). With the
            # synthetic-roots augmentation the d distribution has
            # shifted upward, so d=10 is now the practical ceiling.
            exact_shap = self._enumerate_coalitions_shap(
                dag,
                X,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
            ).astype(np.float32)
            tier = "coalition_enumeration"
        elif self.shap_method == "irreducible_sets" or (
            self.shap_method == "auto" and not all_linear and 10 < n_features <= 15
        ):
            # Nonlinear DAGs — irreducible sets algorithm (Witter et al.).
            # Reachable in 'auto' mode for medium feature counts (10 < d <= 15).
            # The AllClasses BFS visits O(2^d) closed sets in the worst case
            # (e.g. when many roots are direct predecessors of the sink —
            # every subset of those roots is a closed set), so d > 15 would be
            # catastrophic (2^16 = 65K, 2^20 = 1M). d <= 10 uses the exact
            # coalition enumeration above.
            exact_shap = self._compute_shap_from_irreducible_sets(
                dag,
                X,
                sink_node,
                node_activations,
                feature_types,
                mlp_params,
                root_to_X_col=root_to_X_col,
            ).astype(np.float32)
            tier = "irreducible_sets"
        else:
            # Everything else is out of scope for ExplainerPFN: either an
            # unknown ``shap_method`` or a dataset beyond the exact-label
            # feature budget (d > 15, which upstream routed to Monte Carlo).
            raise ValueError(
                f"Unsupported SHAP configuration (shap_method={self.shap_method!r}, "
                f"n_features={n_features}, all_linear={all_linear}). ExplainerPFN "
                "training labels must use exact Shapley formulas with d <= 15; "
                "permutation-MC labels at d > 15 are DiffusionExplainerPFN's "
                "scaling mechanism."
            )

        # Diagnostic: which SHAP tier did this dataset land in?
        # Useful for understanding where the wall-time is going and
        # whether the GPU dispatch is firing. Logged at DEBUG level
        # (off by default). Enable in your own scripts with:
        #     logging.getLogger("explainerpfn.train.do_shapley").setLevel(logging.DEBUG)
        # or globally with:
        #     logging.basicConfig(level=logging.DEBUG)
        if self.use_torch_shap and self.shap_device.type == "cuda" and torch.cuda.is_available():
            device = "GPU"
        else:
            device = "CPU"
        logger.debug(
            "[SHAP] n_features=%d, all_linear=%s, tier=%s, device=%s",
            n_features,
            all_linear,
            tier,
            device,
        )

        # 4. X is already root-only; linear_shap uses the same X.
        #    With the target excluded, the linear model must approximate the
        #    true relationship from input features alone — giving meaningful
        #    LinearSHAP values that the diffuser can improve upon.
        linear_shap = self._compute_linear_shap(X, p_hat)

        return X, y, p_hat, exact_shap, linear_shap
