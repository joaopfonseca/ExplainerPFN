"""Unit tests for the joint multi-attribution head."""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from tabpfn.model.bar_distribution import FullSupportBarDistribution

import explainerpfn.base as base_mod
from explainerpfn.base import ExplainerPFN
from explainerpfn.model.attribution_head import (
    AttributionHead,
    bucket_means,
    downsample_borders,
    efficiency_penalty,
)


def test_downsample_borders_preserves_endpoints_and_count():
    borders = np.linspace(-5.0, 7.0, 5001)
    out = downsample_borders(borders, 256)
    assert out.shape == (257,)
    assert out[0] == pytest.approx(-5.0)
    assert out[-1] == pytest.approx(7.0)
    # Monotonic non-decreasing.
    assert torch.all(out[1:] >= out[:-1])


def test_downsample_borders_passthrough_when_small():
    borders = np.array([0.0, 1.0, 2.0, 3.0])
    out = downsample_borders(borders, 256)
    assert out.shape == (4,)
    assert torch.equal(out, torch.tensor(borders, dtype=torch.float32))


def test_head_output_shape_and_batch():
    head = AttributionHead(emb_dim=8, n_buckets=16, hidden_dim=8, nhead=2, n_layers=1)
    n, d = 5, 4
    heads = {
        "embeddings": torch.randn(n, d, 8),
        "features": torch.randn(n, d),
        "score": torch.randn(n),
    }
    logits = head(**heads)
    assert logits.shape == (n, d, 16)


def test_head_accepts_score_column():
    head = AttributionHead(emb_dim=8, n_buckets=4, hidden_dim=8, nhead=2, n_layers=1)
    logits = head(
        torch.randn(3, 2, 8), torch.randn(3, 2), torch.randn(3, 1)
    )
    assert logits.shape == (3, 2, 4)


def test_head_gradient_flow():
    head = AttributionHead(emb_dim=8, n_buckets=8, hidden_dim=8, nhead=2, n_layers=1)
    logits = head(torch.randn(4, 3, 8), torch.randn(4, 3), torch.randn(4))
    logits.sum().backward()
    grads = [p.grad for p in head.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_head_rejects_too_many_features():
    head = AttributionHead(emb_dim=8, n_buckets=4, hidden_dim=8, nhead=2, max_features=3)
    with pytest.raises(ValueError, match="exceeds max_features"):
        head(torch.randn(2, 4, 8), torch.randn(2, 4), torch.randn(2))


def test_head_rejects_wrong_emb_dim():
    head = AttributionHead(emb_dim=8, n_buckets=4, hidden_dim=8, nhead=2)
    with pytest.raises(ValueError, match="emb_dim mismatch"):
        head(torch.randn(2, 3, 5), torch.randn(2, 3), torch.randn(2))


def test_head_config_roundtrip():
    head = AttributionHead(
        emb_dim=8, n_buckets=12, hidden_dim=8, nhead=2, n_layers=3, max_features=7
    )
    clone = AttributionHead.from_config(head.config())
    assert clone.config() == head.config()
    # Same parameters -> same output.
    emb, feats, score = torch.randn(2, 3, 8), torch.randn(2, 3), torch.randn(2)
    clone.load_state_dict(head.state_dict())
    assert torch.allclose(head(emb, feats, score), clone(emb, feats, score))


def test_bucket_means_are_convex_combination():
    centers = torch.tensor([-2.0, 0.0, 3.0])
    logits = torch.randn(4, 2, 3) * 5
    means = bucket_means(logits, centers)
    assert means.shape == (4, 2)
    assert torch.all(means >= centers.min()) and torch.all(means <= centers.max())


def test_efficiency_penalty_zero_and_positive():
    target = torch.tensor([1.0, -2.0])
    matching = torch.tensor([[0.5, 0.5], [-1.0, -1.0]])
    assert efficiency_penalty(matching, target).item() == pytest.approx(0.0)
    off = torch.tensor([[0.5, 0.7], [-1.0, -1.0]])
    assert efficiency_penalty(off, target).item() > 0


def test_head_optimization_reduces_efficiency_error():
    """A few gradient steps on a sum-consistency objective must reduce the error."""
    torch.manual_seed(0)
    n, d, k = 16, 3, 8
    head = AttributionHead(emb_dim=6, n_buckets=k, hidden_dim=6, nhead=2, n_layers=1)
    centers = torch.linspace(-3.0, 3.0, k)
    embeddings = torch.randn(n, d, 6)
    features = torch.randn(n, d)
    score = torch.randn(n)
    # Consistency target: the per-feature means must sum to this.
    target_sum = torch.randn(n)

    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    def err():
        with torch.no_grad():
            logits = head(embeddings, features, score)
            means = bucket_means(logits, centers)
            return efficiency_penalty(means, target_sum).item()

    initial = err()
    for _ in range(60):
        optimizer.zero_grad()
        logits = head(embeddings, features, score)
        means = bucket_means(logits, centers)
        loss = efficiency_penalty(means, target_sum)
        loss.backward()
        optimizer.step()
    assert err() < initial * 0.5


# ── Loading / saving the head through checkpoints ──────────────────────


@dataclass
class _FakeConfig:
    emsize: int = 8


class _FakeModel(torch.nn.Module):
    def __init__(self, emsize: int = 8):
        super().__init__()
        self.proj = torch.nn.Linear(emsize, emsize)

    def empty_trainset_representation_cache(self):
        return None


def _fake_bardist() -> FullSupportBarDistribution:
    return FullSupportBarDistribution(torch.linspace(-3.0, 3.0, 513))


def _head_state_and_config(
    emb_dim: int = 8, n_buckets: int = 8, hidden_dim: int = 16, nhead: int = 2
) -> tuple[dict, dict]:
    head = AttributionHead(
        emb_dim=emb_dim,
        n_buckets=n_buckets,
        hidden_dim=hidden_dim,
        nhead=nhead,
        n_layers=1,
    )
    return head.state_dict(), head.config()


def test_save_foundation_model_embeds_head(tmp_path):
    xai = ExplainerPFN()
    xai.model_ = _FakeModel(8)
    xai.config_ = _FakeConfig(8)
    xai.head_ = AttributionHead(
        emb_dim=8, n_buckets=8, hidden_dim=16, nhead=2, n_layers=1
    )
    xai.head_config_ = xai.head_.config()

    path = tmp_path / "foundation.ckp"
    xai.save_foundation_model(path)

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["config"] == {"emsize": 8}
    assert ckpt["head_config"] == xai.head_.config()
    for key, value in xai.head_.state_dict().items():
        assert torch.equal(ckpt["head_state_dict"][key], value)


def test_save_foundation_model_without_head_has_no_head_keys(tmp_path):
    xai = ExplainerPFN()
    xai.model_ = _FakeModel(8)
    xai.config_ = _FakeConfig(8)

    path = tmp_path / "plain.ckp"
    xai.save_foundation_model(path)

    ckpt = torch.load(path, weights_only=False)
    assert "head_state_dict" not in ckpt
    assert "head_config" not in ckpt


@pytest.mark.parametrize("wrapper", ["foundation", "trainer"])
def test_load_attribution_head_from_checkpoint(tmp_path, wrapper):
    state, cfg = _head_state_and_config(hidden_dim=16)
    payload: dict = {"config": {}}
    if wrapper == "foundation":
        payload["state_dict"] = {}
    else:
        payload["model_state_dict"] = {}
    payload["head_state_dict"] = state
    payload["head_config"] = cfg
    path = tmp_path / f"{wrapper}.pt"
    torch.save(payload, path)

    # head_kwargs must be overridden by the checkpoint's saved config.
    xai = ExplainerPFN(head_kwargs={"hidden_dim": 999})
    xai.config_ = _FakeConfig(8)
    xai.bardist_ = _fake_bardist()
    xai.device_ = torch.device("cpu")

    assert xai.load_attribution_head(path) is True
    assert xai.head_.config() == cfg
    assert xai.head_.hidden_dim == 16


def test_load_attribution_head_returns_false_without_head(tmp_path):
    path = tmp_path / "nohead.pt"
    torch.save({"state_dict": {}, "config": {}}, path)

    xai = ExplainerPFN()
    assert xai.load_attribution_head(path) is False
    assert xai.head_ is None


def test_initialize_model_variables_autoloads_head(tmp_path, monkeypatch):
    state, cfg = _head_state_and_config(hidden_dim=16)
    ckpt_path = tmp_path / "joint.ckp"
    torch.save(
        {
            "state_dict": {},
            "config": {},
            "head_state_dict": state,
            "head_config": cfg,
        },
        ckpt_path,
    )
    _patch_model_loading(monkeypatch)

    xai = ExplainerPFN(model_path=str(ckpt_path))
    xai._initialize_model_variables()

    assert xai.head_ is not None
    assert xai.head_.config() == cfg


def test_initialize_model_variables_autoload_fallbacks(tmp_path, monkeypatch):
    _patch_model_loading(monkeypatch)
    plain = tmp_path / "plain.ckp"
    torch.save({"state_dict": {}, "config": {}}, plain)

    # Checkpoint without a head + joint_head=True -> freshly initialized head.
    head_kwargs = {"emb_dim": 8, "n_buckets": 4, "hidden_dim": 8, "nhead": 2}
    xai = ExplainerPFN(
        model_path=str(plain), joint_head=True, head_kwargs=head_kwargs
    )
    xai._initialize_model_variables()
    assert xai.head_ is not None

    # Checkpoint without a head + joint_head=False -> no head.
    xai2 = ExplainerPFN(model_path=str(plain))
    xai2._initialize_model_variables()
    assert xai2.head_ is None

    # Non-file model_path directive -> no head, no crash.
    xai3 = ExplainerPFN(model_path="auto")
    xai3._initialize_model_variables()
    assert xai3.head_ is None


def _patch_model_loading(monkeypatch):
    monkeypatch.setattr(
        base_mod,
        "initialize_explainerpfn_model",
        lambda **kwargs: (_FakeModel(8), _FakeConfig(8), _fake_bardist()),
    )
    monkeypatch.setattr(base_mod, "update_encoder_params", lambda **kwargs: None)
    monkeypatch.setattr(
        base_mod, "determine_precision", lambda *args, **kwargs: (False, None, 8)
    )


def test_predict_mode_resolution(monkeypatch):
    xai = ExplainerPFN()
    monkeypatch.setattr(base_mod, "check_is_fitted", lambda *args, **kwargs: None)
    monkeypatch.setattr(xai, "executor_", [], raising=False)
    monkeypatch.setattr(xai, "model_", torch.nn.Module(), raising=False)

    calls: list[str] = []
    monkeypatch.setattr(
        ExplainerPFN,
        "_predict_joint",
        lambda self, X, y, **kw: calls.append("joint")
        or np.zeros((len(X), X.shape[1])),
    )
    monkeypatch.setattr(
        ExplainerPFN,
        "predict_feature",
        lambda self, X, y, **kw: calls.append("per-feature") or np.zeros(len(X)),
    )

    X = np.zeros((2, 3))
    y = np.zeros(2)

    xai.head_ = object()
    xai.predict(X, y)
    assert calls == ["joint"]

    calls.clear()
    xai.head_ = None
    xai.predict(X, y)
    assert calls == ["per-feature"] * X.shape[1]

    calls.clear()
    xai.head_ = object()
    xai.predict(X, y, mode="per-feature")
    assert calls == ["per-feature"] * X.shape[1]

