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
