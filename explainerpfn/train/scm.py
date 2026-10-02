"""Structural causal model (SCM) container for DAGGenerator.

Adapted from DiffusionExplainerPFN @ commit d5f8258 (branch
``refactor/subpackage-restructure``), module
``diffusionexplainerpfn/data/scm.py``. Unmodified.

A sampled SCM bundles everything needed to (a) reproduce the sampled data's
root-feature matrix and target, and (b) later recompute do-Shapley values with
:class:`~diffusionexplainerpfn.attributions.do_shapley.DoShapley` — without the
generator being alive. It carries:

- the ``dag`` (networkx DiGraph with edge weights),
- the per-node structural equations (``node_activations`` + ``mlp_params``),
- the root ``feature_types`` and the ``sink_node``,
- snapshots of the per-dataset root-marginal state (Beta specs and, for
  external-X datasets, the empirical observed roots),
- the RNG object and the normal->Beta marginal map used by the value function.

The RNG is stored by reference (not copied) so that computing SHAP immediately
after generation consumes exactly the same seed stream as the combined
``DAGGenerator.generate_dataset`` call (bit-identical). Deferred computation
after intervening RNG use yields statistically equivalent, but not identical,
Monte Carlo draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import networkx as nx
import numpy as np


def _sample_feature_value(rng, n_samples, feature_type):
    """Draw a root feature value based on its declared type.

    Moved verbatim from ``DAGGenerator._generate_feature_value`` so the
    generator method and :meth:`StructuralCausalModel.sample_feature_value`
    share a single implementation (and therefore an identical RNG stream).
    """
    if feature_type == "numerical":
        return rng.standard_normal(n_samples)
    elif feature_type == "binary":
        return rng.binomial(1, 0.5, n_samples)
    elif feature_type == "categorical":
        n_categories = rng.integers(2, 5)
        return rng.integers(0, n_categories, n_samples)
    elif feature_type == "ordinal":
        return rng.integers(0, 3, n_samples)
    else:
        return rng.standard_normal(n_samples)


def extract_roots_and_target(dag, df, sink_node):
    """Extract root-feature matrix ``X`` (float64) and target ``y`` (float32).

    Verbatim mirror of the extraction block in
    :meth:`DoShapley.compute <diffusionexplainerpfn.attributions.do_shapley.DoShapley.compute>`
    so that ``generate_dataset(include_shap=False)`` followed by
    ``compute_shap`` reproduces the default path bit-for-bit. Only root-node
    columns become features; the sink node is the target.
    """
    topo_order = list(nx.topological_sort(dag))
    root_nodes = [n for n in topo_order if dag.in_degree(n) == 0]
    root_to_X_col = {node: i for i, node in enumerate(root_nodes)}

    X = np.zeros((df.shape[0], len(root_nodes)), dtype=np.float64)
    for root in root_nodes:
        col_name = f"ind_{root}" if f"ind_{root}" in df.columns else f"dep_{root}"
        X[:, root_to_X_col[root]] = df[col_name].values.astype(np.float64)

    sink_col = f"dep_{sink_node}" if f"dep_{sink_node}" in df.columns else f"ind_{sink_node}"
    if sink_col in df.columns:
        y = df[sink_col].values.astype(np.float32)
    else:
        y = df.values[:, -1].astype(np.float32)

    return X, y


@dataclass
class StructuralCausalModel:
    """A sampled structural causal model plus its attribution-time state.

    Args:
        dag: NetworkX DiGraph with ``weight`` edge attributes.
        sink_node: Target node ID.
        node_activations: ``{node: activation_name}``.
        feature_types: ``{node: feature_type}`` for root nodes.
        mlp_params: ``{node: (n_parents, W1, b1, W2, b2)}`` for MLP nodes.
        root_marginals: Snapshot of ``{root_node: beta_spec}`` for roots with
            a Beta marginal (empty otherwise).
        observed_roots: Snapshot of the standardized external-X matrix used
            for bootstrap resampling, or ``None`` for purely synthetic data.
        root_marginal_map: ``(z, spec) -> ndarray`` normal->Beta marginal map.
        rng: ``numpy.random.Generator`` shared with the source generator.
    """

    dag: Any
    sink_node: int
    node_activations: Dict[int, str]
    feature_types: Dict[int, str]
    mlp_params: Dict[int, Any] = field(default_factory=dict)
    root_marginals: Dict[int, Any] = field(default_factory=dict)
    observed_roots: Optional[np.ndarray] = None
    root_marginal_map: Optional[Callable] = None
    rng: Any = None

    @property
    def topo_order(self):
        """Nodes in topological order."""
        return list(nx.topological_sort(self.dag))

    @property
    def root_nodes(self):
        """Root (in-degree 0) nodes, in topological order."""
        return [n for n in self.topo_order if self.dag.in_degree(n) == 0]

    @property
    def n_features(self):
        """Number of root features (columns of ``X``)."""
        return len(self.root_nodes)

    def root_to_X_col(self):
        """Map each root node ID to its column index in ``X``."""
        return {node: i for i, node in enumerate(self.root_nodes)}

    def sample_feature_value(self, n_samples, feature_type):
        """Sample root values from this SCM's RNG (bound sampler)."""
        if self.rng is None:
            raise RuntimeError("StructuralCausalModel has no rng; cannot sample feature values")
        return _sample_feature_value(self.rng, n_samples, feature_type)

    def extract_roots_and_target(self, df):
        """Extract ``(X, y)`` from an all-node DataFrame generated by this SCM."""
        return extract_roots_and_target(self.dag, df, self.sink_node)
