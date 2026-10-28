"""Tests for the single-pass architecture (one forward -> all attributions)."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tabpfn.model.bar_distribution import FullSupportBarDistribution

import explainerpfn.base as base_mod
from explainerpfn.base import ExplainerPFN
from explainerpfn.model.attribution_head import AttributionHead
from explainerpfn.train.synthetic_data import _permute_columns


@dataclass
class _FakeConfig:
    emsize: int = 8


class _FakeModel(torch.nn.Module):
    def __init__(self, emsize: int = 8):
        super().__init__()
        self.proj = torch.nn.Linear(emsize, emsize)

    def empty_trainset_representation_cache(self):
        return None


def _scalar_head(emb_dim=8, max_features=15) -> AttributionHead:
    return AttributionHead(
        emb_dim=emb_dim,
        n_buckets=4,
        hidden_dim=8,
        nhead=2,
        n_layers=1,
        max_features=max_features,
        output="scalar",
    )


# ── Head output modes ─────────────────────────────────────────────────


def test_head_scalar_output_shape():
    head = _scalar_head(emb_dim=8)
    n, d = 5, 4
    out = head(
        torch.randn(n, d, 8), torch.randn(n, d), torch.randn(n)
    )
    assert out.shape == (n, d)


def test_head_scalar_config_roundtrip():
    head = _scalar_head()
    cfg = head.config()
    assert cfg["output"] == "scalar"
    clone = AttributionHead.from_config(cfg)
    assert clone.config() == cfg
    clone.load_state_dict(head.state_dict())
    emb, feats, score = torch.randn(3, 2, 8), torch.randn(3, 2), torch.randn(3)
    assert torch.allclose(head(emb, feats, score), clone(emb, feats, score))


def test_head_scalar_gradient_flow():
    head = _scalar_head()
    out = head(torch.randn(4, 3, 8), torch.randn(4, 3), torch.randn(4))
    out.sum().backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_head_distribution_output_unchanged():
    head = AttributionHead(
        emb_dim=8, n_buckets=16, hidden_dim=8, nhead=2, n_layers=1
    )
    out = head(torch.randn(2, 3, 8), torch.randn(2, 3), torch.randn(2))
    assert out.shape == (2, 3, 16)


# ── Order augmentation (permutation with aligned labels) ──────────────


def test_permute_columns_keeps_alignment():
    X = np.arange(4 * 5, dtype=float).reshape(4, 5)
    shap = X * 10.0
    Xp, sp, perm = _permute_columns(X, shap, seed=123)
    assert Xp.shape == X.shape
    # Slot j of the permuted arrays is original column perm[j] in both.
    assert np.array_equal(Xp, X[:, perm])
    assert np.array_equal(sp, shap[:, perm])
    assert sorted(perm.tolist()) == list(range(X.shape[1]))


def test_permute_columns_is_deterministic():
    X = np.random.default_rng(0).normal(size=(10, 6))
    shap = np.random.default_rng(1).normal(size=(10, 6))
    _, _, p1 = _permute_columns(X, shap, seed=42)
    _, _, p2 = _permute_columns(X, shap, seed=42)
    assert np.array_equal(p1, p2)


# ── Checkpoint task-mode plumbing ─────────────────────────────────────


def test_load_attribution_head_restores_task_mode(tmp_path):
    head = _scalar_head()
    path = tmp_path / "v2.ckp"
    torch.save(
        {
            "state_dict": {},
            "config": {},
            "head_state_dict": head.state_dict(),
            "head_config": head.config(),
            "task_mode": "single-pass",
        },
        path,
    )

    xai = ExplainerPFN()
    xai.config_ = _FakeConfig(8)
    xai.bardist_ = FullSupportBarDistribution(torch.linspace(-3.0, 3.0, 513))
    xai.device_ = torch.device("cpu")

    assert xai.load_attribution_head(path) is True
    assert xai.task_mode == "single-pass"
    assert xai.head_.output == "scalar"


def test_save_foundation_model_records_task_mode(tmp_path):
    xai = ExplainerPFN(task_mode="single-pass")
    xai.model_ = _FakeModel(8)
    xai.config_ = _FakeConfig(8)
    xai.head_ = _scalar_head()
    xai.head_config_ = xai.head_.config()

    path = tmp_path / "v2.ckp"
    xai.save_foundation_model(path)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["task_mode"] == "single-pass"
    assert ckpt["head_config"]["output"] == "scalar"


# ── predict path ──────────────────────────────────────────────────────


def _prepared_single_pass(monkeypatch, emb_dim=8, n_features=3):
    xai = ExplainerPFN(task_mode="single-pass")
    xai.head_ = _scalar_head(emb_dim=emb_dim, max_features=15)
    xai.device_ = torch.device("cpu")
    xai.use_autocast_ = False
    xai.feature_exp_std_ = 2.0
    xai.n_features_in_ = n_features
    xai.n_estimators = 1
    monkeypatch.setattr(base_mod, "check_is_fitted", lambda *a, **k: None)
    return xai


def test_predict_single_pass_uses_one_forward(monkeypatch):
    xai = _prepared_single_pass(monkeypatch)
    calls = {"n": 0}

    def fake_embeddings(X, y):
        calls["n"] += 1
        return torch.zeros(len(X), 8)

    monkeypatch.setattr(xai, "_single_pass_embeddings", fake_embeddings)

    X = np.random.default_rng(0).normal(size=(6, 3))
    y = np.random.default_rng(1).normal(size=6)
    out = xai.predict(X, y)

    assert out.shape == (6, 3)
    assert np.all(np.isfinite(out))
    assert calls["n"] == 1  # exactly one backbone forward


def test_predict_per_feature_rejected_for_single_pass(monkeypatch):
    xai = _prepared_single_pass(monkeypatch)
    X = np.zeros((4, 3))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="single-pass"):
        xai.predict(X, y, mode="per-feature")


def test_predict_quantiles_rejected_for_scalar(monkeypatch):
    xai = _prepared_single_pass(monkeypatch)
    monkeypatch.setattr(
        xai, "_single_pass_embeddings", lambda X, y: torch.zeros(len(X), 8)
    )
    with pytest.raises(ValueError, match="quantiles"):
        xai.predict(np.zeros((4, 3)), np.zeros(4), output_type="quantiles")


# ── LinearSHAP prior channel (v3) ─────────────────────────────────────


def _prior_head(emb_dim=8) -> AttributionHead:
    return AttributionHead(
        emb_dim=emb_dim,
        n_buckets=4,
        hidden_dim=8,
        nhead=2,
        n_layers=1,
        max_features=15,
        output="scalar",
        use_prior=True,
    )


def test_head_prior_channel_shape_and_config_roundtrip():
    head = _prior_head()
    n, d = 5, 4
    emb, feat, score, prior = (
        torch.randn(n, d, 8),
        torch.randn(n, d),
        torch.randn(n),
        torch.randn(n, d),
    )
    out = head(emb, feat, score, prior)
    assert out.shape == (n, d)

    cfg = head.config()
    assert cfg["use_prior"] is True
    clone = AttributionHead.from_config(cfg)
    assert clone.config() == cfg
    clone.load_state_dict(head.state_dict())
    assert torch.allclose(
        head(emb, feat, score, prior), clone(emb, feat, score, prior)
    )


def test_head_prior_validation():
    emb, feat, score = (
        torch.randn(3, 2, 8),
        torch.randn(3, 2),
        torch.randn(3),
    )
    with pytest.raises(ValueError, match="no prior"):
        _prior_head()(emb, feat, score)
    with pytest.raises(ValueError, match="use_prior=False"):
        _scalar_head()(emb, feat, score, torch.randn(3, 2))


def test_old_head_config_without_prior_still_loads():
    # A v1/v2 checkpoint has no ``use_prior`` key in its head config.
    cfg = {
        "emb_dim": 8,
        "n_buckets": 4,
        "hidden_dim": 8,
        "n_layers": 1,
        "nhead": 2,
        "dropout": 0.0,
        "max_features": 15,
        "output": "scalar",
    }
    head = AttributionHead.from_config(cfg)
    assert head.use_prior is False
    assert head.config()["use_prior"] is False


def test_linear_prior_matches_ols_and_is_permutation_aligned():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 4] + 1.0

    xai = ExplainerPFN(task_mode="single-pass", use_linear_prior=True)
    xai._fit_linear_prior(X, y)
    np.testing.assert_allclose(
        xai.linear_prior_coef_[[0, 1, 4]], [3.0, -2.0, 0.5], atol=1e-6
    )
    np.testing.assert_allclose(
        xai.linear_prior_coef_[[2, 3]], [0.0, 0.0], atol=1e-6
    )

    shap = X * 10.0  # labels only used for permutation alignment
    Xp, _, perm = _permute_columns(X, shap, seed=7)
    prior = xai._linear_prior(X)
    # Computing the prior on the permuted X equals permuting the prior.
    xai._fit_linear_prior(Xp, y)
    np.testing.assert_allclose(
        xai._linear_prior(Xp), prior[:, perm], atol=1e-8
    )


def test_predict_single_pass_passes_prior(monkeypatch):
    xai = _prepared_single_pass(monkeypatch)
    xai.head_ = _prior_head()
    xai.use_linear_prior = True
    xai.linear_prior_coef_ = np.array([1.0, -2.0, 0.5])
    xai.linear_prior_mean_ = np.array([0.5, 0.0, -1.0])
    monkeypatch.setattr(
        xai, "_single_pass_embeddings", lambda X, y: torch.zeros(len(X), 8)
    )

    captured = {}
    original = xai.head_.forward

    def wrapped(emb, feat, score, prior=None, confidence=None):
        captured["prior"] = None if prior is None else prior.detach().clone()
        return original(emb, feat, score, prior, confidence)

    monkeypatch.setattr(xai.head_, "forward", wrapped)

    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y = np.array([0.1, 0.2])
    out = xai.predict(X, y)

    expected = ((X - xai.linear_prior_mean_) * xai.linear_prior_coef_) / (
        xai.feature_exp_std_
    )
    assert captured["prior"] is not None
    np.testing.assert_allclose(
        captured["prior"].numpy(), expected.astype(np.float32), atol=1e-6
    )
    assert out.shape == (2, 3)


# ── v4: item states, confidence, residual ─────────────────────────────


def _v4_head(emb_dim=8) -> AttributionHead:
    return AttributionHead(
        emb_dim=emb_dim,
        n_buckets=4,
        hidden_dim=8,
        nhead=2,
        n_layers=1,
        max_features=15,
        output="scalar",
        use_prior=True,
        use_confidence=True,
        residual=True,
    )


def test_head_residual_starts_exactly_at_prior():
    head = _v4_head()
    n, d = 4, 3
    emb = torch.randn(n, d, 8)
    feat = torch.randn(n, d)
    score = torch.randn(n)
    prior = torch.randn(n, d)
    conf = torch.rand(n)
    out = head(emb, feat, score, prior, conf)
    # Zero-initialised output layer => the correction is exactly zero at init.
    assert torch.allclose(out, prior)


def test_head_residual_and_confidence_config_roundtrip():
    head = _v4_head()
    cfg = head.config()
    assert cfg["residual"] is True
    assert cfg["use_confidence"] is True
    clone = AttributionHead.from_config(cfg)
    assert clone.config() == cfg
    clone.load_state_dict(head.state_dict())
    emb, feat, score = torch.randn(3, 2, 8), torch.randn(3, 2), torch.randn(3)
    prior, conf = torch.randn(3, 2), torch.rand(3)
    assert torch.allclose(
        head(emb, feat, score, prior, conf),
        clone(emb, feat, score, prior, conf),
    )


def test_head_residual_requires_prior_and_scalar():
    with pytest.raises(ValueError, match="requires use_prior"):
        AttributionHead(output="scalar", residual=True)
    with pytest.raises(ValueError, match="scalar"):
        AttributionHead(output="distribution", residual=True, use_prior=True)


def test_head_confidence_validation():
    emb, feat, score = torch.randn(3, 2, 8), torch.randn(3, 2), torch.randn(3)
    prior = torch.randn(3, 2)
    with pytest.raises(ValueError, match="no confidence"):
        _v4_head()(emb, feat, score, prior)
    with pytest.raises(ValueError, match="use_confidence=False"):
        _scalar_head()(emb, feat, score, None, torch.zeros(3))


def test_fit_item_mapping_handles_permutation_recoding_and_extra_columns():
    from explainerpfn.utils import fit_item_mapping

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    perm = np.array([2, 0, 3, 1])
    recoded = X[:, perm] ** 3  # monotone (power) recode of permuted columns
    fingerprints = rng.normal(size=(60, 2))
    X_pre = np.column_stack(
        [
            fingerprints[:, 0],
            recoded[:, 0],  # raw 2
            fingerprints[:, 1],
            recoded[:, 1],  # raw 0
            recoded[:, 2],  # raw 1
            recoded[:, 3],  # raw 3
        ]
    )
    mapping = fit_item_mapping(X, X_pre)
    # X_pre = [fp0, X2^3, fp1, X0^3, X3^3, X1^3]
    assert mapping.tolist() == [3, 5, 1, 4]


def test_fit_item_mapping_raises_on_unmatched_columns():
    from explainerpfn.utils import fit_item_mapping

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3))
    X_pre = rng.normal(size=(40, 3))  # unrelated columns
    with pytest.raises(ValueError, match="could not match"):
        fit_item_mapping(X, X_pre)


class _FakeExecutor:
    def __init__(self, items):
        self._items = items

    def iter_outputs(self, X, y, *, device, autocast, only_return_standard_out=False):
        yield {"test_item_embeddings": self._items}, None


def test_single_pass_item_states_reorders_to_raw_columns():
    xai = ExplainerPFN(
        task_mode="single-pass", use_item_states=True, features_per_group=1
    )
    xai.model_ = _FakeModel(8)
    xai.model_.features_per_group = 1
    xai.device_ = torch.device("cpu")
    xai.use_autocast_ = False
    xai.item_mapping_ = np.array([2, 0])  # raw0 <- pre2, raw1 <- pre0

    n, f, e = 5, 3, 8
    items = torch.arange(n * f * e, dtype=torch.float32).reshape(n, 1, f, e)
    xai.executor_ = [_FakeExecutor(items)]

    out = xai._single_pass_item_states(np.zeros((n, 2)), np.zeros(n))
    assert out.shape == (n, 2, e)
    assert torch.equal(out[:, 0], items[:, 0, 2])
    assert torch.equal(out[:, 1], items[:, 0, 0])


def test_predict_single_pass_passes_item_states_prior_and_confidence(
    monkeypatch,
):
    xai = _prepared_single_pass(monkeypatch)
    xai.use_item_states = True
    xai.use_linear_prior = True
    xai.head_ = _v4_head()
    xai.linear_prior_coef_ = np.array([1.0, -2.0, 0.5])
    xai.linear_prior_mean_ = np.array([0.0, 0.0, 0.0])
    xai.linear_prior_r2_ = 0.42
    xai.feature_exp_std_ = 2.0

    captured = {}

    def fake_items(X, y, mapping=None):
        t = torch.zeros(len(X), 3, 8)
        captured["shape"] = tuple(t.shape)
        return t

    monkeypatch.setattr(xai, "_single_pass_item_states", fake_items)
    original = xai.head_.forward

    def wrapped(emb, feat, score, prior=None, confidence=None):
        captured["emb"] = tuple(emb.shape)
        captured["conf"] = None if confidence is None else confidence.clone()
        return original(emb, feat, score, prior, confidence)

    monkeypatch.setattr(xai.head_, "forward", wrapped)
    out = xai.predict(np.zeros((4, 3)), np.zeros(4))

    assert out.shape == (4, 3)
    assert captured["shape"] == (4, 3, 8)
    assert captured["emb"] == (4, 3, 8)  # not expanded from a row embedding
    assert float(captured["conf"][0]) == pytest.approx(0.42)


def test_save_foundation_model_records_v4_layout(tmp_path):
    xai = ExplainerPFN(
        task_mode="single-pass",
        features_per_group=1,
        use_item_states=True,
    )
    xai.model_ = _FakeModel(8)
    xai.config_ = _FakeConfig(8)
    xai.head_ = _v4_head()
    xai.head_config_ = xai.head_.config()

    path = tmp_path / "v4.ckp"
    xai.save_foundation_model(path)
    ckpt = torch.load(path, weights_only=False)
    assert ckpt["features_per_group"] == 1
    assert ckpt["use_item_states"] is True
    assert ckpt["head_config"]["residual"] is True

    restored = ExplainerPFN()
    restored.config_ = _FakeConfig(8)
    restored.bardist_ = FullSupportBarDistribution(torch.linspace(-3.0, 3.0, 513))
    restored.device_ = torch.device("cpu")
    assert restored.load_attribution_head(path) is True
    assert restored.features_per_group == 1
    assert restored.use_item_states is True


def test_predict_residual_correction_clamp(monkeypatch):
    xai = _prepared_single_pass(monkeypatch)
    xai.use_item_states = True
    xai.use_linear_prior = True
    xai.correction_clamp = 0.5
    xai.head_ = _v4_head()
    xai.linear_prior_coef_ = np.array([1.0, 1.0, 1.0])
    xai.linear_prior_mean_ = np.zeros(3)
    xai.linear_prior_r2_ = 0.5
    xai.feature_exp_std_ = 1.0
    monkeypatch.setattr(
        xai,
        "_single_pass_item_states",
        lambda X, y, mapping=None: torch.zeros(len(X), 3, 8),
    )

    X = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    prior = X.copy()  # (X - 0) * 1 / std(1)

    def fake_forward(emb, feat, score, prior=None, confidence=None):
        return prior + torch.full_like(prior, 100.0)

    monkeypatch.setattr(xai.head_, "forward", fake_forward)
    out = xai.predict(X, np.zeros(2))

    corr = out - prior
    np.testing.assert_allclose(
        np.linalg.norm(corr, axis=1),
        0.5 * np.linalg.norm(prior, axis=1),
        rtol=1e-4,
    )

