"""Joint multi-attribution head for ExplainerPFN.

The per-feature pipeline runs the backbone once per feature (each on a task
whose target is that feature's value) and reads one attribution distribution
per pass. This head keeps that backbone structure but *mixes* the per-feature
query representations jointly, so all d attribution distributions are produced
together and can be trained for cross-feature consistency (in particular the
efficiency identity ``sum_j phi_j = f(x) - base_value``).

Input, per query row: one token per feature slot ``j``,

    token_j = proj(cat(h_j, x_j, score)) + slot_embedding_j

where ``h_j`` is the backbone ``test_embeddings`` for feature ``j``'s task,
``x_j`` is the query row's value for feature ``j`` and ``score`` is the model
score (p_hat). The tokens of a row are mixed by a small transformer encoder
over the feature slots; each slot is then linearly projected to a distribution
over attribution buckets.

Output: ``logits`` of shape ``(n_samples, n_features, n_buckets)``, read on the
(possibly downsampled) TabPFN bar-distribution borders on the normalized
attribution scale (``phi / feature_exp_std`` — same scale as the per-feature
training target).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


def bucket_means(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Expected value per bucket distribution.

    Args:
        logits: ``(..., n_buckets)`` logits.
        centers: ``(n_buckets,)`` bucket center values.

    Returns:
        ``(...)`` expected values.
    """
    probs = torch.softmax(logits, dim=-1)
    return (probs * centers).sum(dim=-1)


def efficiency_penalty(
    mean_per_feature: torch.Tensor, target_sum: torch.Tensor
) -> torch.Tensor:
    """MSE of the efficiency residual ``sum_j phi_j - (f(x) - base_value)``.

    Args:
        mean_per_feature: ``(n_samples, n_features)`` predicted attribution
            means (normalized scale).
        target_sum: ``(n_samples,)`` target total (``f(x) - base_value`` on the
            same scale).

    Returns:
        Scalar penalty.
    """
    residual = mean_per_feature.sum(dim=-1) - target_sum.reshape(-1)
    return (residual**2).mean()


def downsample_borders(borders, n_buckets: int) -> torch.Tensor:
    """Uniformly subsample a border grid, preserving both endpoints.

    Args:
        borders: 1-D array/Tensor of ``n + 1`` border values (bucket edges).
        n_buckets: Desired number of buckets; yields ``n_buckets + 1`` borders.

    Returns:
        Float32 tensor of ``n_buckets + 1`` borders. If the input already has
        at most ``n_buckets + 1`` borders it is returned (as a tensor) as-is.
    """
    borders = np.asarray(
        borders.detach().cpu() if torch.is_tensor(borders) else borders,
        dtype=np.float64,
    ).ravel()
    if borders.size <= n_buckets + 1:
        return torch.as_tensor(borders, dtype=torch.float32)
    idx = np.linspace(0, borders.size - 1, n_buckets + 1).round().astype(int)
    # Guard against (very unlikely) duplicate indices after rounding.
    idx = np.unique(idx)
    if idx.size < n_buckets + 1:
        idx = np.linspace(0, borders.size - 1, n_buckets + 1).astype(int)
    return torch.as_tensor(borders[idx], dtype=torch.float32)


class AttributionHead(nn.Module):
    """Small slot-mixer over per-feature backbone representations.

    Args:
        emb_dim: Backbone embedding dimension (``h_j``).
        n_buckets: Number of attribution buckets to predict per feature.
        hidden_dim: Internal width of the mixer. Defaults to ``emb_dim``.
        n_layers: Number of transformer-encoder layers over feature slots.
        nhead: Attention heads (must divide ``hidden_dim``).
        dropout: Dropout inside the encoder. Defaults to 0.0 (matches the
            deterministic training protocol).
        max_features: Largest number of feature slots (d) the slot embedding
            supports. Datasets with more features than this raise.
    """

    def __init__(
        self,
        *,
        emb_dim: int = 192,
        n_buckets: int = 256,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        nhead: int = 6,
        dropout: float = 0.0,
        max_features: int = 15,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or emb_dim)
        if hidden_dim % nhead != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by nhead={nhead}"
            )
        self.emb_dim = int(emb_dim)
        self.n_buckets = int(n_buckets)
        self.hidden_dim = hidden_dim
        self.n_layers = int(n_layers)
        self.nhead = int(nhead)
        self.dropout = float(dropout)
        self.max_features = int(max_features)

        # Per-slot token: [h_j (emb_dim), x_j (1), score (1)] -> hidden.
        self.token_proj = nn.Linear(emb_dim + 2, hidden_dim)
        self.slot_embedding = nn.Parameter(
            torch.zeros(max_features, hidden_dim)
        )
        nn.init.normal_(self.slot_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mixer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

        self.out_proj = nn.Linear(hidden_dim, self.n_buckets)
        nn.init.zeros_(self.out_proj.bias)

    # ------------------------------------------------------------------
    # Config / construction
    # ------------------------------------------------------------------

    def config(self) -> dict[str, Any]:
        """Serializable constructor kwargs (stored in checkpoints)."""
        return {
            "emb_dim": self.emb_dim,
            "n_buckets": self.n_buckets,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "nhead": self.nhead,
            "dropout": self.dropout,
            "max_features": self.max_features,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "AttributionHead":
        return cls(**config)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings: torch.Tensor,
        features: torch.Tensor,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Predict all features' attribution-bucket logits jointly.

        Args:
            embeddings: ``(n_samples, n_features, emb_dim)`` per-feature
                backbone representations (``test_embeddings``).
            features: ``(n_samples, n_features)`` raw query feature values.
            score: ``(n_samples,)`` or ``(n_samples, 1)`` model score.

        Returns:
            ``(n_samples, n_features, n_buckets)`` logits.
        """
        if embeddings.ndim != 3:
            raise ValueError(
                f"embeddings must be (n, d, emb_dim); got {tuple(embeddings.shape)}"
            )
        n_samples, n_features, emb_dim = embeddings.shape
        if emb_dim != self.emb_dim:
            raise ValueError(
                f"emb_dim mismatch: head expects {self.emb_dim}, got {emb_dim}"
            )
        if n_features > self.max_features:
            raise ValueError(
                f"n_features={n_features} exceeds max_features="
                f"{self.max_features}"
            )

        score = score.reshape(n_samples, 1).expand(n_samples, n_features).unsqueeze(-1)
        features = features.reshape(n_samples, n_features, 1)

        tokens = torch.cat([embeddings, features, score], dim=-1)
        tokens = self.token_proj(tokens)
        tokens = tokens + self.slot_embedding[:n_features].unsqueeze(0)

        mixed = self.mixer(tokens)  # (n, d, hidden)
        logits = self.out_proj(mixed)  # (n, d, n_buckets)
        return logits
