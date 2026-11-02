"""Format detection for model loading: foundation .ckp vs trainer .pt."""

import pytest
import torch

from explainerpfn import model_loading
from explainerpfn.model_loading import (
    _load_foundation_payload,
    _reconstitute_foundation_checkpoint,
    _resolve_foundation_reference,
    load_model,
)


def _reference_payload():
    """Foundation-format checkpoint, including a criterion (bardist) key."""
    return {
        "state_dict": {
            "encoder.weight": torch.ones(3, 4),
            "head.bias": torch.zeros(4),
            "criterion.borders": torch.linspace(0.0, 1.0, 5),
        },
        "config": {"emsize": 64},
    }


def _trainer_payload(epoch=83000, model_path="/nonexistent/stock.ckp"):
    """Trainer-format checkpoint as written by scripts/train_explainerpfn.py."""
    return {
        "model_state_dict": {
            "encoder.weight": torch.zeros(3, 4),
            "head.bias": torch.ones(4),
        },
        "optimizer_state_dict": {},
        "epoch": epoch,
        "config": {"model_path": model_path, "lr": 1e-5},
    }


def test_trainer_weights_replace_reference_model_tensors():
    out = _reconstitute_foundation_checkpoint(
        _trainer_payload(),
        _reference_payload(),
        source_path="trainer.pt",
        reference_path="ref.ckp",
    )
    # Shared keys: weights must come from the trainer checkpoint.
    for key, tensor in _trainer_payload()["model_state_dict"].items():
        assert torch.equal(out["state_dict"][key], tensor)
    # Criterion keys: must come from the reference (never trained).
    for key, tensor in _reference_payload()["state_dict"].items():
        if key.startswith("criterion."):
            assert torch.equal(out["state_dict"][key], tensor)
    # Full key set of the reference is preserved, config comes from reference.
    assert set(out["state_dict"]) == set(_reference_payload()["state_dict"])
    assert out["config"] == {"emsize": 64}


def test_reconstitution_does_not_mutate_reference():
    reference = _reference_payload()
    before = {k: v.clone() for k, v in reference["state_dict"].items()}
    _reconstitute_foundation_checkpoint(
        _trainer_payload(),
        reference,
        source_path="trainer.pt",
        reference_path="ref.ckp",
    )
    for key, tensor in reference["state_dict"].items():
        assert torch.equal(tensor, before[key])


def test_trainer_key_unknown_to_reference_rejected():
    trainer = _trainer_payload()
    trainer["model_state_dict"]["strange.weight"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="strange.weight"):
        _reconstitute_foundation_checkpoint(
            trainer,
            _reference_payload(),
            source_path="trainer.pt",
            reference_path="ref.ckp",
        )


def test_trainer_shape_mismatch_rejected():
    trainer = _trainer_payload()
    trainer["model_state_dict"]["encoder.weight"] = torch.zeros(3, 5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        _reconstitute_foundation_checkpoint(
            trainer,
            _reference_payload(),
            source_path="trainer.pt",
            reference_path="ref.ckp",
        )


def test_load_model_rejects_unrecognized_format(tmp_path):
    path = tmp_path / "mystery.ckp"
    torch.save({"optimizer_state_dict": {}}, path)
    with pytest.raises(ValueError, match="neither supported format"):
        load_model(path=path)


def test_reference_resolution_prefers_recorded_model_path(tmp_path):
    recorded = tmp_path / "stock.ckp"
    torch.save(_reference_payload(), recorded)
    resolved = _resolve_foundation_reference(
        _trainer_payload(model_path=str(recorded)),
        which="regressor",
        version="v2",
        download=False,
    )
    assert resolved == recorded


def test_reference_resolution_errors_without_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_loading,
        "resolve_model_path",
        lambda *_args, **_kwargs: (
            tmp_path / "missing.ckp",
            tmp_path,
            "model.ckp",
            "regressor",
        ),
    )
    with pytest.raises(ValueError, match="downloading is disabled"):
        _resolve_foundation_reference(
            _trainer_payload(),
            which="regressor",
            version="v2",
            download=False,
        )


def test_chained_trainer_checkpoints_resolve_recursively(tmp_path):
    """v5 <- v4 <- stock must resolve the architecture back to the stock model."""
    stock = tmp_path / "stock.ckp"
    torch.save(_reference_payload(), stock)

    v4 = tmp_path / "v4.pt"
    torch.save(_trainer_payload(model_path=str(stock)), v4)

    v5 = tmp_path / "v5.pt"
    torch.save(_trainer_payload(model_path=str(v4)), v5)

    out = _load_foundation_payload(
        v5, which="regressor", version="v2", download=False
    )
    assert out["config"] == {"emsize": 64}
    assert set(out["state_dict"]) == set(_reference_payload()["state_dict"])


def test_foundation_reference_cycle_is_rejected(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    torch.save(_trainer_payload(model_path=str(b)), a)
    torch.save(_trainer_payload(model_path=str(a)), b)

    with pytest.raises(ValueError, match="Cycle in the foundation reference"):
        _load_foundation_payload(a, which="regressor", version="v2", download=False)
