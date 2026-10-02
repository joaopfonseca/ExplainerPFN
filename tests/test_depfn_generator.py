"""Tests for the online exact-do-Shapley training-data generator.

Covers the wrapper (:class:`SyntheticDataGenerator`) around the copied
DiffusionExplainerPFN generator and the online
:class:`TrainingBatchIterator`. The generator's own invariants are verified in
DiffusionExplainerPFN; here we focus on the ExplainerPFN contract: d in
[3, 15] inclusive, exact-label tiers only, roots-only X, correct ``ind_``
labelling, and the batch dict consumed by notebook 6.
"""

import numpy as np
import pytest

from explainerpfn.train.synthetic_data import (
    SyntheticDataGenerator,
    TrainingBatchIterator,
)


def _small_gen(**kwargs):
    """A fast, deterministic generator with a narrow configuration."""
    defaults = dict(
        n_features_range=(4, 4),
        n_samples_range=(50, 50),
        random_state=0,
        mc_samples=8,
    )
    defaults.update(kwargs)
    return SyntheticDataGenerator(**defaults)


def _efficiency_error(exact_shap, p_hat, baseline=None):
    if baseline is None:
        baseline = np.mean(p_hat)
    return float(np.abs(exact_shap.sum(axis=1) - (p_hat - baseline)).mean())


# ── Seeding / determinism ─────────────────────────────────────────────


def test_seeded_determinism():
    a = _small_gen().generate_one(0)
    b = _small_gen().generate_one(0)
    np.testing.assert_array_equal(a["X"].values, b["X"].values)
    np.testing.assert_array_equal(a["y"].values, b["y"].values)
    np.testing.assert_array_equal(a["shap"].values, b["shap"].values)


def test_seed_is_order_independent():
    """Dataset i must not depend on how many datasets were drawn before it."""
    gen = _small_gen()
    direct = gen.generate_one(0)["X"].values

    gen.generate_one(0)
    gen.generate_one(1)
    again = gen.generate_one(0)["X"].values
    np.testing.assert_array_equal(direct, again)


def test_different_indices_differ():
    gen = _small_gen()
    x0 = gen.generate_one(0)["X"].values
    x1 = gen.generate_one(1)["X"].values
    assert x0.shape != x1.shape or not np.allclose(x0, x1)


# ── Feature-range invariant (inclusive, d = 15) ───────────────────────


def test_feature_range_inclusive_high():
    """d = 15 must be reachable, and the exact-label tier must be closed-form."""
    gen = _small_gen(
        n_features_range=(15, 15),
        linear_dag_prob=1.0,  # force the cheap, exact closed-form tier
    )
    dataset = gen.generate_one(0)
    assert dataset["X"].shape[1] == 15
    assert dataset["params"]["tier"] == "closed_form"


def test_default_range_respected_over_many_datasets():
    gen = SyntheticDataGenerator(random_state=123, mc_samples=20, n_samples_range=(100, 100))
    tiers = set()
    for i in range(10):
        dataset = gen.generate_one(i)
        d = dataset["X"].shape[1]
        assert 3 <= d <= 15, f"index {i}: realized d={d} outside [3, 15]"
        tiers.add(dataset["params"]["tier"])
    # d = 15 may or may not occur in 10 draws; tiers must stay exact.
    assert tiers <= {"closed_form", "coalition_enumeration", "irreducible_sets"}


def test_no_root_count_above_high():
    """Rejection sampling must cap realized roots at n_features_range high."""
    gen = _small_gen(n_features_range=(3, 6), n_dags_range=(1, 4))
    for i in range(30):
        d = gen.generate_one(i)["X"].shape[1]
        assert 3 <= d <= 6


# ── Sample-count range (inclusive, high = 5000) ───────────────────────


def test_default_sample_range():
    """The wrapper must default to 250-5000 rows (inclusive high)."""
    assert SyntheticDataGenerator().n_samples_range == (250, 5000)


def test_n_samples_range_inclusive_high():
    """Both ends of a non-degenerate sample range must be reachable.

    The wrapped generator samples rows in [low, high); the ``+1`` translation
    in ``_make_generator`` must make ``high`` reachable while leaving a
    degenerate (low == high) range exact.
    """
    gen = _small_gen(
        n_samples_range=(4999, 5000),
        linear_dag_prob=1.0,  # closed-form tier: cheap even at 5000 rows
    )
    seen = {gen.generate_one(i)["X"].shape[0] for i in range(8)}
    assert seen == {4999, 5000}, f"expected both bounds, got {seen}"

    # Degenerate range stays exact (no off-by-one from the +1 translation).
    exact = _small_gen(n_samples_range=(250, 250), linear_dag_prob=1.0)
    assert exact.generate_one(0)["X"].shape[0] == 250


def test_iterator_max_cells_disabled_by_default():
    """The dataset-size filter must be off unless explicitly enabled."""
    it = TrainingBatchIterator(generator=_small_gen())
    try:
        assert it.max_cells is None
    finally:
        it.close()


# ── Exact-label semantics ─────────────────────────────────────────────


def test_roots_only_and_ind_labels():
    gen = _small_gen(linear_dag_prob=1.0)
    dataset = gen.generate_one(0)
    columns = list(dataset["X"].columns)
    assert all(c.startswith("ind_") for c in columns)
    assert columns == list(dataset["shap"].columns)
    assert len(columns) == len(dataset["scm"].root_nodes)
    # The sink node is not a feature column.
    assert f"ind_{dataset['scm'].sink_node}" not in columns


def test_efficiency_residual_is_noise_scale():
    """Exact labels leave a residual ~ noise_std against the noisy ``y_pred``.

    The value function is noise-free by construction (the efficiency axiom
    holds exactly there), while ``y_pred`` carries the data-generation noise.
    So ``|sum_j phi_j - (y_pred - E[y_pred])| ~ noise_std`` — the documented,
    correct semantics (see the module docstring), not a labelling error.
    """
    gen = _small_gen(linear_dag_prob=1.0, noise_std=0.1)
    dataset = gen.generate_one(0)
    err = _efficiency_error(dataset["shap"].values, dataset["y_pred"])
    assert 0.02 < err < 0.3, f"linear efficiency residual off noise scale: {err}"


def test_efficiency_residual_nonlinear_matches_noise_scale():
    """Nonlinear exact labels leave a residual on the order of noise_std."""
    gen = _small_gen(linear_dag_prob=0.0, noise_std=0.1, n_samples_range=(300, 300))
    errors = []
    for i in range(5):
        dataset = gen.generate_one(i)
        errors.append(_efficiency_error(dataset["shap"].values, dataset["y_pred"]))
    assert np.mean(errors) < 0.5


def test_no_nan_in_labels():
    gen = _small_gen(linear_dag_prob=0.0)
    for i in range(3):
        dataset = gen.generate_one(i)
        for key in ("X", "y", "shap"):
            assert np.all(np.isfinite(dataset[key].values)), f"index {i}: {key} non-finite"


# ── Tier routing ──────────────────────────────────────────────────────


def test_tier_coalition_enumeration_for_small_nonlinear():
    gen = _small_gen(n_features_range=(4, 4), linear_dag_prob=0.0)
    assert gen.generate_one(0)["params"]["tier"] == "coalition_enumeration"


def test_tier_irreducible_sets_for_medium_nonlinear():
    gen = _small_gen(
        n_features_range=(12, 12),
        linear_dag_prob=0.0,
        n_samples_range=(100, 100),
        mc_samples=10,
    )
    assert gen.generate_one(0)["params"]["tier"] == "irreducible_sets"


def test_tier_closed_form_for_forced_linear():
    gen = _small_gen(n_features_range=(8, 8), linear_dag_prob=1.0)
    assert gen.generate_one(0)["params"]["tier"] == "closed_form"


def test_monte_carlo_method_rejected():
    gen = _small_gen(shap_method="monte_carlo")
    with pytest.raises(ValueError, match="monte_carlo"):
        gen.generate_one(0)


def test_do_shapley_constructor_rejects_monte_carlo():
    from explainerpfn.train.do_shapley import DoShapley

    with pytest.raises(ValueError, match="monte_carlo"):
        DoShapley(
            rng=np.random.default_rng(0),
            generate_feature_value=lambda n, ft: np.zeros(n),
            root_marginal_map=lambda z, spec: z,
            shap_method="monte_carlo",
        )


# ── Wrapper contract ──────────────────────────────────────────────────


def test_generate_accumulates_and_describe():
    gen = _small_gen(linear_dag_prob=1.0)
    gen.generate(3)
    assert len(gen.X_) == len(gen.y_) == len(gen.explanations_) == 3
    desc = gen.describe()
    assert {"n_train_samples", "n_features", "tier"} <= set(desc.columns)
    assert len(desc) == 3


# ── TrainingBatchIterator ─────────────────────────────────────────────


def test_iterator_batch_contract():
    gen = _small_gen(linear_dag_prob=1.0, n_samples_range=(128, 128))
    it = TrainingBatchIterator(generator=gen, max_datasets=1, prefetch=1)
    try:
        batch = it.next_batch()
    finally:
        it.close()

    assert set(batch) == {
        "X",
        "y",
        "shap",
        "executor_configs",
        "feature_exp_std",
        "dataset_index",
        "dataset_seed",
    }
    n_samples, n_features = batch["X"].shape
    assert batch["y"].shape == (n_samples,)
    assert batch["shap"].shape == (n_samples, n_features)
    assert len(batch["executor_configs"]) == n_features
    assert np.isfinite(batch["feature_exp_std"]) and batch["feature_exp_std"] > 0
    assert np.all(np.isfinite(batch["X"]))
    assert np.all(np.isfinite(batch["shap"]))
    assert batch["dataset_index"] == 0


def test_iterator_start_index_resumes_stream():
    """``start_index=k`` must reproduce the k-th dataset of a fresh stream.

    This is the invariant the training CLI relies on: after ``--resume`` it
    rebuilds the iterator with ``start_index=<consumed>`` and expects the same
    data it would have generated contiguously.
    """
    gen_full = _small_gen(linear_dag_prob=1.0, n_samples_range=(60, 60))
    full = TrainingBatchIterator(generator=gen_full, max_datasets=3, prefetch=1)
    try:
        datasets = [full.next_batch() for _ in range(3)]
    finally:
        full.close()

    # A stream that starts at dataset 2 must equal the third full batch.
    gen_resumed = _small_gen(linear_dag_prob=1.0, n_samples_range=(60, 60))
    resumed = TrainingBatchIterator(
        generator=gen_resumed, max_datasets=1, start_index=2, prefetch=1
    )
    try:
        batch = resumed.next_batch()
    finally:
        resumed.close()

    assert batch["dataset_index"] == 2
    np.testing.assert_array_equal(batch["X"], datasets[2]["X"])
    np.testing.assert_array_equal(batch["shap"], datasets[2]["shap"])


def test_iterator_streams_multiple_batches_threaded():
    gen = _small_gen(linear_dag_prob=1.0, n_samples_range=(100, 100))
    it = TrainingBatchIterator(generator=gen, max_datasets=2, prefetch=1)
    try:
        shapes = [it.next_batch()["X"].shape for _ in range(2)]
        assert len(shapes) == 2
    finally:
        it.close()


def test_iterator_num_samples_subsampling():
    gen = _small_gen(linear_dag_prob=1.0, n_samples_range=(400, 400))
    it = TrainingBatchIterator(
        generator=gen, max_datasets=1, num_samples=100, prefetch=1
    )
    try:
        batch = it.next_batch()
    finally:
        it.close()
    assert batch["X"].shape[0] == 100


def test_iterator_exhausts_and_raises_stopiteration():
    gen = _small_gen(linear_dag_prob=1.0)
    it = TrainingBatchIterator(generator=gen, max_datasets=1, prefetch=1)
    try:
        it.next_batch()
        with pytest.raises(StopIteration):
            for _ in range(5):
                it.next_batch()
    finally:
        it.close()
