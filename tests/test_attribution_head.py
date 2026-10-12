"""Unit tests for the joint multi-attribution head."""

import numpy as np
import pytest
import torch

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
