"""GPU (torch + torch.compile) implementation of the SHAP value function.

Adapted from DiffusionExplainerPFN @ commit d5f8258 (branch
``refactor/subpackage-restructure``), module
``diffusionexplainerpfn/attributions/dag_value_torch.py``. Unmodified except
for dropping one dead local binding (``device = X.device``) to keep the
project lint-clean.

This module provides a torch-native re-implementation of the DAG value
function ``v(S)`` for the Witter/irreducible-sets SHAP algorithm. The pure-numpy
CPU path is used for small problems or when CUDA is unavailable; the torch path
is used when the problem is large enough that GPU launch + matmul overhead is
amortised.

Design
------

The function ``evaluate_v_batched_torch`` evaluates a *batch* of v(S)
calls in a single GPU forward pass. The batch axis is the *coalition*
axis: a single call returns v(S_r) for every coalition index r in
``[0, R)``. The implementation walks the DAG in topological order
once, and for every node v applies the per-coalition per-node
forward pass with a leading coalition axis. Chunking handles the
case where ``R * n_mc * n_samples * n_parents`` would exceed GPU
memory.

Numerical conventions
---------------------

* All tensors are float32 (matches the model's training dtype).
* ``mc_X`` is sampled ONCE per call (variance-reduction trick):
  each coalition shares the same (n_mc, n_samples, n_roots) noise
  tensor for the non-intervened roots.
* The torch path matches the numpy path's bias-free convention: no
  noise is added during the v() forward pass (the data-generation
  step already added noise when creating X).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math

import numpy as np
import torch

# Activation-name -> integer code. Defined once so we can pack the
# per-node activation into a single int tensor of shape (V,).
_ACT_LINEAR = 0
_ACT_RELU = 1
_ACT_TANH = 2
_ACT_SIN = 3
_ACT_SOFT_INTERACTION = 4
_ACT_GAUSSIAN = 5
_ACT_SIGMOID_C = 6
_ACT_MLP = 7


def _parse_activation_name(name: str) -> Tuple[int, float]:
    """Return (activation_code, scalar_param) for a given activation name.

    For scalar activations, ``scalar_param`` is unused. For
    ``sigmoid_c{c}`` it is the steepness c. For ``mlp_*`` it is a
    hash of the activation string used only to differentiate MLP
    nodes (the actual W1/b1/W2/b2 live in the metadata).
    """
    if name == "linear":
        return _ACT_LINEAR, 0.0
    if name == "relu":
        return _ACT_RELU, 0.0
    if name == "tanh":
        return _ACT_TANH, 0.0
    if name == "sin":
        return _ACT_SIN, 0.0
    if name == "soft_interaction":
        return _ACT_SOFT_INTERACTION, 0.0
    if name == "gaussian":
        return _ACT_GAUSSIAN, 0.0
    if name.startswith("sigmoid_c"):
        c = float(name.split("c", 1)[1])
        return _ACT_SIGMOID_C, c
    if name.startswith("mlp_"):
        return _ACT_MLP, 0.0
    return _ACT_LINEAR, 0.0


@dataclass
class DagValueMetadata:
    """GPU-friendly, pure-tensor representation of a DAG for v(S).

    Built once per SHAP computation via ``DagValueMetadata.from_dag``
    and reused across all coalition evaluations.
    """

    topo_order: List
    sink_idx: int
    root_indices: List[int]

    # parent_idx[v, p] = index in topo_order of the p-th parent of v.
    # Padded to (V, max_parents) with -1. -1 entries are ignored.
    parent_idx: torch.Tensor  # int64, (V, P)
    # parent_weight[v, p] = edge weight for the p-th parent. -1 ignored.
    parent_weight: torch.Tensor  # float32, (V, P)
    # activation_code[v] = integer code (see _ACT_* constants).
    activation_code: torch.Tensor  # int64, (V,)
    # activation_param[v] = scalar param (e.g. sigmoid steepness c).
    activation_param: torch.Tensor  # float32, (V,)
    # n_parents[v] = number of direct parents of v (0 for roots).
    n_parents: torch.Tensor  # int64, (V,)
    # root_to_X_col: which column in X corresponds to each root.
    # -1 for non-roots.
    root_to_X_col: torch.Tensor  # int64, (V,) with -1 for non-roots.
    # For MLP nodes: lookup table mapping (node_idx) -> (W1, b1, W2, b2).
    mlp_params: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]

    @classmethod
    def from_dag(
        cls,
        dag,
        sink_node,
        node_activations: Dict,
        mlp_params: Dict,
        root_to_X_col: Dict,
    ) -> "DagValueMetadata":
        """Build a DagValueMetadata from a NetworkX DiGraph."""
        import networkx as nx

        topo_order = list(nx.topological_sort(dag))
        V = len(topo_order)
        node_to_idx = {n: i for i, n in enumerate(topo_order)}

        root_indices = [i for i, n in enumerate(topo_order) if dag.in_degree(n) == 0]
        sink_idx = node_to_idx[sink_node]

        parents_per_node: List[List[Tuple[int, float]]] = []
        max_parents = 0
        for n in topo_order:
            ps = list(dag.predecessors(n))
            ps_w = [(node_to_idx[p], float(dag[p][n]["weight"])) for p in ps]
            parents_per_node.append(ps_w)
            if len(ps_w) > max_parents:
                max_parents = len(ps_w)
        P = max(max_parents, 1)

        parent_idx = np.full((V, P), -1, dtype=np.int64)
        parent_weight = np.zeros((V, P), dtype=np.float32)
        n_parents_arr = np.zeros(V, dtype=np.int64)
        for v, ps in enumerate(parents_per_node):
            n_parents_arr[v] = len(ps)
            for k, (p, w) in enumerate(ps):
                parent_idx[v, k] = p
                parent_weight[v, k] = w

        activation_code = np.zeros(V, dtype=np.int64)
        activation_param = np.zeros(V, dtype=np.float32)
        mlp_params_torch: Dict[
            int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for v, n in enumerate(topo_order):
            name = node_activations.get(n, "linear")
            code, param = _parse_activation_name(name)
            activation_code[v] = code
            activation_param[v] = param
            if code == _ACT_MLP:
                _n_parents, W1, b1, W2, b2 = mlp_params[n]
                mlp_params_torch[v] = (
                    torch.from_numpy(np.asarray(W1, dtype=np.float32)),
                    torch.from_numpy(np.asarray(b1, dtype=np.float32)),
                    torch.from_numpy(np.asarray(W2, dtype=np.float32)),
                    torch.from_numpy(np.asarray(b2, dtype=np.float32)),
                )

        root_to_X_col_arr = np.full(V, -1, dtype=np.int64)
        for root_node, col in root_to_X_col.items():
            root_to_X_col_arr[node_to_idx[root_node]] = int(col)

        return cls(
            topo_order=topo_order,
            sink_idx=sink_idx,
            root_indices=root_indices,
            parent_idx=torch.from_numpy(parent_idx),
            parent_weight=torch.from_numpy(parent_weight),
            activation_code=torch.from_numpy(activation_code),
            activation_param=torch.from_numpy(activation_param),
            mlp_params=mlp_params_torch,
            n_parents=torch.from_numpy(n_parents_arr),
            root_to_X_col=torch.from_numpy(root_to_X_col_arr),
        )

    def to(self, device: str) -> "DagValueMetadata":
        new_mlp = {}
        for k, (W1, b1, W2, b2) in self.mlp_params.items():
            new_mlp[k] = (W1.to(device), b1.to(device), W2.to(device), b2.to(device))
        return DagValueMetadata(
            topo_order=self.topo_order,
            sink_idx=self.sink_idx,
            root_indices=self.root_indices,
            parent_idx=self.parent_idx.to(device),
            parent_weight=self.parent_weight.to(device),
            activation_code=self.activation_code.to(device),
            activation_param=self.activation_param.to(device),
            mlp_params=new_mlp,
            n_parents=self.n_parents.to(device),
            root_to_X_col=self.root_to_X_col.to(device),
        )


def _sample_root_values_torch(
    feature_types_root: List[str],
    n_mc: int,
    n_samples: int,
    n_roots: int,
    device: str,
    generator: Optional[torch.Generator] = None,
    rng: Optional["np.random.Generator"] = None,
    observed_roots: Optional[np.ndarray] = None,
    root_marginals: Optional[List] = None,
) -> torch.Tensor:
    """Sample (n_mc, n_samples, n_roots) root-feature values on the device.

    Matches the numpy ``_generate_feature_value`` semantics for the
    four standard feature types (numerical, binary, categorical,
    ordinal). For categorical features, the numpy path samples
    ``n_categories`` per call from ``randint(2, 5)``; we mirror that
    here by drawing one ``n_categories`` per root feature (from the
    provided ``rng``) and using it for all ``(n_mc, n_samples)`` draws
    of that feature. If ``rng`` is None we fall back to a fixed
    ``n_cat=4`` (the previous behavior) so the SHAP estimator stays
    unbiased but may differ slightly from the numpy path.

    When ``observed_roots`` is provided (an ``(n_obs, n_roots)`` numpy
    array of empirically observed root feature values), numerical
    roots are bootstrap-resampled from the observed column instead of
    ``torch.randn``. This ensures correct interventional expectations
    for non-Gaussian real data injected via ``X_external``.

    When ``root_marginals[j]`` is a spec dict ``{"a", "b", "lu"}`` (a
    Beta base marginal for numerical root ``j``), the Gaussian draw is
    mapped through ``Phi`` -> Beta inverse CDF (a device-resident LUT)
    so the labels integrate against the same marginal as the data. The
    draw itself is still ``torch.randn`` with the same generator, so the
    RNG stream matches the legacy Gaussian path exactly.
    """
    out = torch.empty((n_mc, n_samples, n_roots), device=device, dtype=torch.float32)
    for j, ft in enumerate(feature_types_root):
        spec = root_marginals[j] if root_marginals is not None else None
        if observed_roots is not None and j < observed_roots.shape[1] and ft == "numerical":
            # Empirical bootstrap: sample with replacement from observed column
            obs_col = observed_roots[:, j]
            idx = (
                rng.integers(0, len(obs_col), size=n_mc * n_samples)
                if rng is not None
                else np.random.randint(0, len(obs_col), size=n_mc * n_samples)
            )
            sampled = obs_col[idx].astype(np.float32).reshape(n_mc, n_samples)
            out[:, :, j] = torch.from_numpy(sampled).to(device)
        elif ft == "binary":
            out[:, :, j] = torch.bernoulli(
                torch.full((n_mc, n_samples), 0.5, device=device), generator=generator
            )
        elif ft == "categorical":
            if rng is not None:
                n_cat = int(rng.integers(2, 5))
            else:
                n_cat = 4
            out[:, :, j] = torch.randint(
                0, n_cat, (n_mc, n_samples), device=device, generator=generator
            ).float()
        elif ft == "ordinal":
            out[:, :, j] = torch.randint(
                0, 3, (n_mc, n_samples), device=device, generator=generator
            ).float()
        else:  # numerical
            z = torch.randn(
                n_mc, n_samples, device=device, generator=generator, dtype=torch.float32
            )
            if spec is not None:
                z = _map_normal_to_beta_torch(z, spec, device)
            out[:, :, j] = z
    return out


def _map_normal_to_beta_torch(
    z: torch.Tensor, spec: dict, device: str
) -> torch.Tensor:
    """Map standard-normal draws to a standardized Beta marginal on device.

    ``u = Phi(z)`` (erf-based) then linear interpolation of the Beta
    inverse-CDF LUT, then standardization by the analytic Beta moments.
    The LUT is cached on the spec dict so repeated calls (many
    coalitions / datasets) do not rebuild it.
    """
    grid_u = spec.get("_torch_lut_u")
    grid_x = spec.get("_torch_lut_x")
    if grid_u is None or grid_u.device != torch.device(device):
        u_np, x_np = spec["lut"][0], spec["lut"][1]
        grid_u = torch.from_numpy(u_np.astype(np.float32)).to(device)
        grid_x = torch.from_numpy(x_np.astype(np.float32)).to(device)
        spec["_torch_lut_u"] = grid_u
        spec["_torch_lut_x"] = grid_x
    u = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    u = u.clamp(1e-6, 1.0 - 1e-6)
    # Uniform grid -> direct index interpolation (no searchsorted needed).
    pos = u * (grid_x.numel() - 1)
    i0 = pos.floor().long().clamp(0, grid_x.numel() - 2)
    frac = (pos - i0.float()).to(grid_x.dtype)
    x = grid_x[i0] + frac * (grid_x[i0 + 1] - grid_x[i0])
    mu = spec["lut"][2]
    sigma = spec["lut"][3]
    return (x - mu) / sigma


def evaluate_v_batched_torch(
    meta: DagValueMetadata,
    X: torch.Tensor,
    coalition_mask: torch.Tensor,
    mc_X: torch.Tensor,
    precomputed: Optional[Dict] = None,
) -> torch.Tensor:
    """Evaluate v(S_r) for r in [0, R) in a single GPU forward pass.

    Args:
        meta: Pre-built DagValueMetadata for the DAG.
        X: (n_samples, n_roots) root feature values, on the same device as meta.
        coalition_mask: (R, n_roots) boolean mask; True means the root
            is intervened (set to its observed value), False means it
            is sampled from the natural distribution (mc_X).
        mc_X: (n_mc, n_samples, n_roots) sampled natural-distribution
            values for non-intervened roots. Sampled ONCE per call
            (variance-reduction trick) and broadcast across all R
            coalitions.
        precomputed: Optional pre-fetched per-node metadata to avoid
            repeated CPU<->GPU syncs in the inner loop. When ``None``,
            the function does the precomputation once (suitable for
            a single chunked call). When provided, the values are
            reused across chunks (use ``evaluate_v_batched_chunked_torch``
            to build it once and pass it in).

    Returns:
        v: (R, n_samples) — v(S_r) for each coalition r and sample.
    """
    R, n_roots = coalition_mask.shape
    n_mc, n_samples, _ = mc_X.shape
    mlp_params_meta = meta.mlp_params

    if precomputed is None:
        precomputed = _precompute_meta_for_eval(meta)

    n_par_list = precomputed["n_par_list"]
    root_col_list = precomputed["root_col_list"]
    code_list = precomputed["code_list"]
    param_list = precomputed["param_list"]
    parents_per_node = precomputed["parents_per_node"]  # List[List[int]]
    parent_weight_device = precomputed["parent_weight_device"]
    sink_v = precomputed["sink_v"]

    # Pre-pack per-node parent+weight as a list of (parents_list,
    # weights_tensor) tuples. The list-of-tuples layout is faster
    # than repeated dict lookups in the inner loop.
    V = len(n_par_list)
    node_pw = []  # [(parents, weights), ...] for non-roots; None for roots
    for v in range(V):
        n_p = n_par_list[v]
        if n_p == 0:
            node_pw.append(None)
        else:
            parents = parents_per_node[v]
            weights = parent_weight_device[v, :n_p]
            node_pw.append((parents, weights))

    # Use a list (not dict) for node_vals — list indexing is ~2x
    # faster than dict lookup in CPython, and V is fixed.
    node_vals: List[Optional[torch.Tensor]] = [None] * V

    for v in range(V):
        n_p = n_par_list[v]
        if n_p == 0:
            x_col = root_col_list[v]
            if x_col < 0:
                continue
            x_obs = X[:, x_col]
            x_mc = mc_X[:, :, x_col]
            interv = coalition_mask[:, x_col].view(R, 1, 1)
            obs_b = x_obs.view(1, 1, n_samples).expand(R, n_mc, n_samples)
            mc_b = x_mc.view(1, n_mc, n_samples).expand(R, n_mc, n_samples)
            node_vals[v] = torch.where(interv, obs_b, mc_b)
        else:
            parents, weights = node_pw[v]
            weighted_sum = node_vals[parents[0]] * weights[0]
            for p_idx in range(1, len(parents)):
                p = parents[p_idx]
                w = weights[p_idx]
                weighted_sum = weighted_sum + node_vals[p] * w
            code = code_list[v]
            param = param_list[v]
            if code == _ACT_LINEAR:
                result = weighted_sum
            elif code == _ACT_RELU:
                result = torch.relu(weighted_sum)
            elif code == _ACT_TANH:
                result = torch.tanh(weighted_sum)
            elif code == _ACT_SIN:
                result = torch.sin(weighted_sum)
            elif code == _ACT_SOFT_INTERACTION:
                result = weighted_sum * (1 + torch.tanh(weighted_sum))
            elif code == _ACT_GAUSSIAN:
                result = torch.exp(-(weighted_sum**2))
            elif code == _ACT_SIGMOID_C:
                result = torch.sigmoid(param * weighted_sum)
            elif code == _ACT_MLP:
                # parent_stack (P, R, n_mc, n_samples) -> permute to
                # (R, n_mc, n_samples, P) for the linear @ W1.
                W1, b1, W2, b2 = mlp_params_meta[v]
                parent_stack = torch.stack([node_vals[p] for p in parents], dim=-1)
                hidden = torch.tanh(parent_stack @ W1 + b1)
                result = hidden @ W2 + b2
            else:
                result = weighted_sum
            node_vals[v] = result

    return node_vals[sink_v].mean(dim=1)


def _precompute_meta_for_eval(meta: DagValueMetadata) -> Dict:
    """Pre-fetch every per-node scalar / list from ``meta`` ONCE,
    so the inner ``evaluate_v_batched_torch`` loop can run with
    zero ``.item()`` / ``.tolist()`` syncs per node. Returns a
    dict with:
        * n_par_list: List[int] of n_parents per node
        * root_col_list: List[int] of root X-col per node (-1 for non-roots)
        * code_list: List[int] of activation codes
        * param_list: List[float] of activation params
        * parents_per_node: List[List[int]] of parent topo-order indices
        * parent_weight_device: torch.Tensor on device (V, P)
        * sink_v: int
    """
    V = len(meta.topo_order)
    n_par_list = meta.n_parents.cpu().tolist()
    root_col_list = meta.root_to_X_col.cpu().tolist()
    code_list = meta.activation_code.cpu().tolist()
    param_list = meta.activation_param.cpu().tolist()
    parent_idx_cpu = meta.parent_idx.cpu()

    parents_per_node: List[List[int]] = [[] for _ in range(V)]
    for v in range(V):
        n_p = n_par_list[v]
        if n_p > 0:
            parents_per_node[v] = parent_idx_cpu[v, :n_p].tolist()

    return {
        "n_par_list": n_par_list,
        "root_col_list": root_col_list,
        "code_list": code_list,
        "param_list": param_list,
        "parents_per_node": parents_per_node,
        "parent_weight_device": meta.parent_weight,
        "sink_v": int(meta.sink_idx),
    }


def evaluate_v_batched_chunked_torch(
    meta: DagValueMetadata,
    X: torch.Tensor,
    coalition_mask: torch.Tensor,
    mc_X: torch.Tensor,
    max_chunk_r: int = 128,
) -> torch.Tensor:
    """Same as ``evaluate_v_batched_torch`` but processes the R axis in
    chunks of size ``max_chunk_r`` to bound peak memory.

    Crucially, the per-node scalar precomputation is done ONCE
    (before the chunk loop) and reused across all chunks, so the
    inner loop is sync-free.
    """
    R = coalition_mask.shape[0]
    if R <= max_chunk_r:
        return evaluate_v_batched_torch(meta, X, coalition_mask, mc_X)
    precomputed = _precompute_meta_for_eval(meta)
    out = torch.empty((R, X.shape[0]), device=X.device, dtype=X.dtype)
    for start in range(0, R, max_chunk_r):
        end = min(start + max_chunk_r, R)
        out[start:end] = evaluate_v_batched_torch(
            meta,
            X,
            coalition_mask[start:end],
            mc_X,
            precomputed=precomputed,
        )
    return out
