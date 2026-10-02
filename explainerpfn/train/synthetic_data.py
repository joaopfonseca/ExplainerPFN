"""Online synthetic training-data generation with exact do-Shapley labels.

This module replaces ExplainerPFN's legacy surrogate-SHAP generator. It wraps
:class:`~explainerpfn.train.dag_generators.DAGGenerator` (a copy of the
DiffusionExplainerPFN generator) with ExplainerPFN-specific defaults and
exposes:

* :class:`SyntheticDataGenerator` — draws datasets from random structural
  causal models and labels them with **exact do-Shapley values** (closed-form
  for linear SCMs, full coalition enumeration for d <= 10, the irreducible-sets
  algorithm for 10 < d <= 15; Witter et al., 2026). The feature range is
  capped at d = 15 inclusive so every label lands in an exact tier — the
  permutation-Monte-Carlo SHAP estimator is deliberately excluded here and is
  instead DiffusionExplainerPFN's mechanism for scaling to d > 15.
* :class:`TrainingBatchIterator` — a background producer that generates and
  preprocesses datasets online (no pickle files), yielding the same batch
  dictionaries the legacy ``notebooks/6-train-explainerpfn.ipynb`` ``read_data``
  used to return.

Label semantics
---------------
The exact do-Shapley labels are computed against the **noise-free** structural
expectation: the value function ``v(S) = E[Y | do(X_S = x_S)]`` does not
re-inject the data-generation noise (doing so would violate the efficiency
axiom). Consequently the per-row efficiency residual
``|sum_j phi_j - (p_hat - E[p_hat])|`` is on the order of ``noise_std``
(default 0.1). This is the correct object, not label noise — the legacy
surrogate-SHAP labels had a tiny residual against the *noisy* ``y_pred``
precisely because they explained a fitted model rather than the SCM.

Seeding
-------
Each dataset index maps to an independent seed derived from
``numpy.random.SeedSequence([base_seed, index, attempt])``. Datasets are
therefore reproducible independent of generation order and resumable. A fresh
:class:`DAGGenerator` is built per dataset so the underlying single-RNG stream
cannot leak across datasets.

Reference: Witter et al., 2026. "Exactly Computing do-Shapley Values."
arXiv:2602.07203
"""

import queue
import threading

import numpy as np
import pandas as pd

from explainerpfn.train.dag_generators import DAGGenerator

__all__ = ["SyntheticDataGenerator", "TrainingBatchIterator"]


# Default configuration tuned for ExplainerPFN's proof-of-concept regime.
# These deliberately differ from DiffusionExplainerPFN's v29 scaling config:
# notably ``sink_sigmoid_steepness_range`` is the DEPFN default (0.5, 3.0),
# not v29's (3, 10) which exists to defeat LinearSHAP — a DEPFN-specific
# premise that ExplainerPFN does not share.
_DEFAULT_KWARGS = dict(
    n_features_range=(3, 15),  # inclusive
    n_samples_range=(200, 1500),
    n_dags_range=(1, 4),
    nodes_per_dag_range=(3, 8),
    edge_prob_range=(0.2, 0.4),
    feature_type_probs=[0.6, 0.1, 0.2, 0.1],
    observation_dependence=None,
    shap_method="auto",
    mc_samples=100,
    sink_sigmoid_steepness_range=(0.5, 3.0),
    sink_linear_prob=0.0,
    linear_dag_prob=0.05,
    mlp_hidden_units_range=(8, 16),
    mlp_prob=0.15,
    mlp_sink_prob=0.25,
    noise_std=0.1,
    feature_correlation=0.3,
    beta_marginals=False,
)


class SyntheticDataGenerator:
    """Draw SCM datasets with exact do-Shapley training labels.

    Args:
        n_features_range: Inclusive ``(low, high)`` range for the number of
            root features. Capped at 15 by design (see module docstring).
            Internally translated to the wrapped generator's exclusive-high
            convention so ``high`` is actually reachable.
        n_samples_range: ``(low, high)`` sample-count range.
        max_attempts: How many fresh seeds to try for a single dataset index
            before giving up. Needed because the topology generators can
            produce *more* root nodes than requested (the wrapped generator
            only ever augments roots), which would break the d <= 15
            exact-label invariant.
        random_state: Base seed. ``None`` (default) draws fresh entropy.
        **generator_kwargs: Any remaining keyword is forwarded verbatim to
            :class:`DAGGenerator` (e.g. ``sink_sigmoid_steepness_range``,
            ``mlp_prob``, ``noise_std``, ``feature_correlation``).
    """

    def __init__(
        self,
        n_features_range=(3, 15),
        n_samples_range=(200, 1500),
        max_attempts=100,
        random_state=None,
        verbose=False,
        **generator_kwargs,
    ):
        low, high = n_features_range
        if low < 1 or high < low:
            raise ValueError(f"invalid n_features_range {n_features_range!r}")
        self.n_features_range = (int(low), int(high))
        self._n_features_low = int(low)
        self._n_features_high = int(high)
        self.n_samples_range = tuple(n_samples_range)
        self.max_attempts = int(max_attempts)
        self.random_state = random_state
        self.verbose = verbose

        # Start from the tuned defaults, then let explicit kwargs override.
        config = dict(_DEFAULT_KWARGS)
        config["n_features_range"] = (int(low), int(high))
        config["n_samples_range"] = tuple(n_samples_range)
        config.update(generator_kwargs)
        self._generator_kwargs = config

        self.params_ = []
        self.X_ = []
        self.y_ = []
        self.y_pred_ = []
        self.dags_ = []
        self.explanations_ = []
        self.scms_ = []

    # ------------------------------------------------------------------
    # Seeding / construction
    # ------------------------------------------------------------------

    def _seed_for(self, index, attempt):
        """Deterministic, order-independent seed for (index, attempt)."""
        if self.random_state is None:
            base = int(np.random.SeedSequence().entropy) % (2**63)
        else:
            base = int(self.random_state) % (2**63)
        ss = np.random.SeedSequence([base, int(index), int(attempt)])
        return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63))

    def _make_generator(self, seed):
        """Fresh :class:`DAGGenerator` with this wrapper's config."""
        kwargs = dict(self._generator_kwargs)
        low, high = self.n_features_range
        # The wrapped generator samples n_features in [low, high) and only
        # augments roots upward, so pass high + 1 to make `high` reachable.
        if low == high:
            kwargs["n_features_range"] = (low, low)
        else:
            kwargs["n_features_range"] = (low, high + 1)
        return DAGGenerator(random_state=seed, **kwargs)

    # ------------------------------------------------------------------
    # Single-dataset generation
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_tier(scm, n_features):
        all_linear = all(a == "linear" for a in scm.node_activations.values())
        if all_linear:
            return "closed_form"
        if n_features <= 10:
            return "coalition_enumeration"
        return "irreducible_sets"

    def _pack(self, index, seed, result, scm):
        root_nodes = scm.root_nodes
        columns = [f"ind_{n}" for n in root_nodes]
        n_features = result["X"].shape[1]
        n_samples = result["X"].shape[0]
        X = pd.DataFrame(result["X"], columns=columns)
        shap = pd.DataFrame(result["exact_shap"], columns=columns)
        params = {
            "n_train_samples": int(n_samples),
            "n_features": int(n_features),
            "n_samples": int(n_samples),
            "tier": self._classify_tier(scm, n_features),
            "seed": int(seed),
            "index": int(index),
            "n_nodes": len(scm.dag.nodes),
        }
        return {
            "params": params,
            "X": X,
            "y": pd.Series(result["y"]),
            "y_pred": np.asarray(result["p_hat"]),
            "shap": shap,
            "dag": scm.dag,
            "scm": scm,
        }

    def generate_one(self, index):
        """Generate one dataset for ``index`` (exact labels, d in range).

        Retries with fresh seeds until the realized root count lies in the
        inclusive range, guaranteeing the exact-label invariant.
        """
        for attempt in range(self.max_attempts):
            seed = self._seed_for(index, attempt)
            gen = self._make_generator(seed)
            # Two-step path: bit-identical labels to the combined call, and
            # it hands back the StructuralCausalModel for provenance.
            bundle = gen.generate_dataset(include_shap=False)
            n_features = bundle["X"].shape[1]
            if self._n_features_low <= n_features <= self._n_features_high:
                result = gen.compute_shap(bundle)
                return self._pack(index, seed, result, bundle["scm"])
        raise RuntimeError(
            f"Could not sample a dataset with n_features in "
            f"[{self._n_features_low}, {self._n_features_high}] for index "
            f"{index} within {self.max_attempts} attempts."
        )

    def generate(self, n_datasets=1):
        """Generate ``n_datasets`` datasets and append them to the ``*_`` lists."""
        indices = range(len(self.params_), len(self.params_) + n_datasets)
        for index in indices:
            dataset = self.generate_one(index)
            self.params_.append(dataset["params"])
            self.X_.append(dataset["X"])
            self.y_.append(dataset["y"])
            self.y_pred_.append(dataset["y_pred"])
            self.dags_.append(dataset["dag"])
            self.explanations_.append(dataset["shap"])
            self.scms_.append(dataset["scm"])
        return self

    def describe(self):
        """DataFrame of per-dataset parameters (legacy-compatible columns)."""
        return pd.DataFrame(self.params_)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"n_datasets={len(self.params_)}, "
            f"n_features_range={self.n_features_range}, "
            f"n_samples_range={self.n_samples_range})"
        )


class TrainingBatchIterator:
    """Online producer of preprocessed training batches.

    Replaces the pickle-file ``read_data`` pipeline: a background thread
    generates datasets with :class:`SyntheticDataGenerator`, runs
    ``ExplainerPFN(fit_mode="low_memory").fit(X, y)`` to build the per-feature
    preprocessing executors, drops the fitted model (keeping only executor
    configs), and queues the resulting batch dicts. Callers consume them with
    :meth:`next_batch`.

    Each yielded batch is a dict with the same keys the legacy pipeline used::

        {
            "X": np.ndarray,               # (n_samples, n_features)
            "y": np.ndarray,               # (n_samples,)
            "shap": np.ndarray,            # (n_samples, n_features), exact
            "executor_configs": list,      # per-feature InferenceEngine configs
            "feature_exp_std": float,      # global std of the SHAP labels
        }

    Args:
        generator: A :class:`SyntheticDataGenerator`. If ``None``, a default
            one is constructed.
        prefetch: Number of batches buffered ahead of the consumer (>= 1).
        max_cells: Reject datasets whose ``n_samples * n_features`` reaches
            this bound. Kept for parity with the legacy 70e3 filter; it is a
            no-op at the default ranges but guards future config widening.
        num_samples: If set, deterministically subsample each dataset to at
            most this many rows. ``None`` (default) keeps all rows.
        max_datasets: Optional cap on the number of datasets produced. ``None``
            streams indefinitely.
        device: Device used for the throwaway preprocessing fit.
        random_state: Seed for the optional per-batch subsampling.
    """

    def __init__(
        self,
        generator=None,
        prefetch=1,
        max_cells=70_000,
        num_samples=None,
        max_datasets=None,
        device="cpu",
        random_state=None,
        verbose=False,
    ):
        self.generator = generator if generator is not None else SyntheticDataGenerator()
        self.prefetch = max(1, int(prefetch))
        self.max_cells = max_cells
        self.num_samples = num_samples
        self.max_datasets = max_datasets
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self._queue = queue.Queue(maxsize=self.prefetch)
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._started = False
        self._produced = 0

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, dataset):
        """Fit a throwaway ExplainerPFN and return the training batch dict."""
        from explainerpfn.base import ExplainerPFN
        import torch

        X = np.asarray(dataset["X"].values, dtype=np.float64)
        y = np.asarray(dataset["y"].values, dtype=np.float64)
        shap = np.asarray(dataset["shap"].values, dtype=np.float64)

        if self.num_samples is not None and len(X) > self.num_samples:
            rng = np.random.default_rng(dataset["params"]["seed"])
            idx = rng.choice(len(X), size=self.num_samples, replace=False)
            idx.sort()
            X, y, shap = X[idx], y[idx], shap[idx]

        xai = ExplainerPFN(
            fit_mode="low_memory", random_state=self.random_state, device="cpu"
        )
        xai.fit(X, y)
        # Drop the fitted model; keep only the preprocessing executor configs.
        del xai.model_
        for executor in xai.executor_:
            del executor.model
            executor.use_torch_inference_mode(False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "X": X,
            "y": y,
            "shap": shap,
            "executor_configs": xai.executor_,
            "feature_exp_std": float(shap.std()),
        }

    # ------------------------------------------------------------------
    # Producer thread
    # ------------------------------------------------------------------

    def _worker(self):
        index = 0
        try:
            while not self._stop.is_set():
                if self.max_datasets is not None and index >= self.max_datasets:
                    break
                dataset = self.generator.generate_one(index)
                n_cells = dataset["params"]["n_features"] * dataset["params"]["n_samples"]
                index += 1
                if self.max_cells is not None and n_cells >= self.max_cells:
                    continue
                batch = self._preprocess(dataset)
                self._produced += 1
                # Blocking put with stop-awareness.
                while not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except Exception as exc:  # surfaced on the consumer side
            self._error = exc
        finally:
            # Sentinel so a waiting consumer wakes up.
            try:
                self._queue.put(None, timeout=0.5)
            except queue.Full:
                pass

    def _ensure_started(self):
        if not self._started:
            self._thread = threading.Thread(
                target=self._worker, name="TrainingBatchIterator", daemon=True
            )
            self._thread.start()
            self._started = True

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def next_batch(self):
        """Return the next preprocessed batch, blocking until it is ready."""
        self._ensure_started()
        while True:
            if self._error is not None:
                raise self._error
            try:
                batch = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._error is not None:
                    raise self._error
                if not self._thread.is_alive():
                    raise StopIteration("TrainingBatchIterator exhausted")
                continue
            if batch is None:
                if self._error is not None:
                    raise self._error
                raise StopIteration("TrainingBatchIterator exhausted")
            return batch

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()

    def close(self):
        """Stop the producer thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
