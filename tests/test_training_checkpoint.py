"""Tests for the training checkpoint helpers and CLI resume semantics.

These cover the parts of ``scripts/train_explainerpfn.py`` and
``explainerpfn.utils`` that make cluster runs resumable: atomic saves,
latest-checkpoint discovery, and a full save→load roundtrip that restores the
model, optimizer, epoch, and online-data index.
"""

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from explainerpfn.utils import _retry_io, find_latest_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    """Import scripts/train_explainerpfn.py as a module."""
    path = _REPO_ROOT / "scripts" / "train_explainerpfn.py"
    spec = importlib.util.spec_from_file_location("train_explainerpfn", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_explainerpfn"] = module
    spec.loader.exec_module(module)
    return module


train_mod = _load_train_module()


# ── find_latest_checkpoint ─────────────────────────────────────────────


def test_find_latest_checkpoint_empty_and_missing(tmp_path):
    assert find_latest_checkpoint(str(tmp_path)) is None
    assert find_latest_checkpoint(str(tmp_path / "does_not_exist")) is None
    assert find_latest_checkpoint(None) is None


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    for step in (100, 20, 500, 50):
        (tmp_path / f"checkpoint_{step}.pt").write_bytes(b"x")
    # final_model.pt must be ignored even though it sorts later lexically.
    (tmp_path / "final_model.pt").write_bytes(b"x")
    (tmp_path / "checkpoint_junk.pt").write_bytes(b"x")

    latest = find_latest_checkpoint(str(tmp_path))
    assert os.path.basename(latest) == "checkpoint_500.pt"


# ── _retry_io ──────────────────────────────────────────────────────────


def test_retry_io_recovers_from_transient_oserror(tmp_path):
    calls = {"n": 0}
    target = tmp_path / "out.txt"

    @_retry_io(max_retries=3, base_delay=0.0, backoff=1.0)
    def flaky(path):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("transient EIO")
        with open(path, "w") as fh:
            fh.write("done")
        return path

    assert flaky(str(target)) == str(target)
    assert calls["n"] == 3
    assert target.read_text() == "done"


def test_retry_io_reraises_after_max(tmp_path):
    calls = {"n": 0}

    @_retry_io(max_retries=2, base_delay=0.0, backoff=1.0)
    def always_fails(path):
        calls["n"] += 1
        raise OSError("persistent")

    with pytest.raises(OSError):
        always_fails(str(tmp_path / "x"))
    assert calls["n"] == 3  # 1 initial + 2 retries


def test_retry_io_does_not_retry_non_oserror(tmp_path):
    calls = {"n": 0}

    @_retry_io(max_retries=5, base_delay=0.0)
    def value_error():
        calls["n"] += 1
        raise ValueError("not an I/O error")

    with pytest.raises(ValueError):
        value_error()
    assert calls["n"] == 1


# ── Atomic save + full checkpoint roundtrip ────────────────────────────


def test_atomic_torch_save_no_tmp_left(tmp_path):
    path = tmp_path / "checkpoint_7.pt"
    train_mod._atomic_torch_save(str(path), {"a": torch.tensor([1.0])})
    assert path.exists()
    assert not (tmp_path / "checkpoint_7.pt.tmp").exists()
    loaded = torch.load(str(path), weights_only=False)
    assert torch.equal(loaded["a"], torch.tensor([1.0]))


class _FakeModel(torch.nn.Linear):
    """``torch.nn.Linear`` plus the cache method ``save_checkpoint`` calls."""

    def empty_trainset_representation_cache(self):
        return None


class _FakeXai:
    """Minimal stand-in exposing the ``ExplainerPFN`` attribute used by saves."""

    def __init__(self, model):
        self.model_ = model


def test_checkpoint_roundtrip_restores_state(tmp_path):
    """save_checkpoint → load_checkpoint restores model/optim/epoch/data index."""
    torch.manual_seed(0)
    model = _FakeModel(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Take a step so the optimizer carries non-trivial state.
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    scheduler.step()

    xai = _FakeXai(model)

    losses = [1.0, 0.5, 0.25]
    rng = np.random.default_rng(123)
    rng.standard_normal(5)  # advance so state is non-initial
    args = type("Args", (), {"num_epochs": 10, "seed": 42})()

    logs = []
    path = tmp_path / "checkpoint_9.pt"
    train_mod.save_checkpoint(
        str(path), xai, optimizer, scheduler,
        epoch=9, best_avg_loss=0.25, losses=losses,
        data_start_index=17, rng=rng, args=args, logger=logs.append,
    )
    assert path.exists()

    # Fresh objects: load and compare.
    torch.manual_seed(999)
    model2 = torch.nn.Linear(3, 1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    for group in optimizer2.param_groups:
        group["initial_lr"] = 1e-3
    state = train_mod.load_checkpoint(
        str(path), type("X", (), {"model_": model2})(), optimizer2,
        torch.device("cpu"), logs.append,
    )

    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)
    assert state["epoch"] == 9
    assert state["data_start_index"] == 17
    assert state["losses"] == losses
    assert state["best_avg_loss"] == pytest.approx(0.25)
    # Optimizer state (e.g. Adam step count) must survive.
    assert len(optimizer2.state) == len(optimizer.state)


def test_checkpoint_omits_missing_optional_fields(tmp_path):
    """A minimal checkpoint (model only) loads with sensible defaults."""
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "checkpoint_3.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(path))

    model2 = torch.nn.Linear(2, 1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    state = train_mod.load_checkpoint(
        str(path), type("X", (), {"model_": model2})(), optimizer2,
        torch.device("cpu"), lambda *_: None,
    )
    assert state["epoch"] == 0
    assert state["data_start_index"] == 0
    assert state["best_avg_loss"] == float("inf")
