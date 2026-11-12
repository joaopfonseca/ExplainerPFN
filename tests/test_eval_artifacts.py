"""Tests for the helpers that let the eval scripts consume ground-truth artifacts."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import _common as common  # noqa: E402
from explainerpfn.datasets import REGISTRY, ExpandedDatasets  # noqa: E402
from explainerpfn.datasets.processing import normalize_name  # noqa: E402


def _write(directory, dataset, payload):
    with open(Path(directory) / f"{normalize_name(dataset)}.pkl", "wb") as f:
        pickle.dump(payload, f)


def test_artifact_dir_is_lazy_and_keyed_by_display_name(tmp_path):
    names = [REGISTRY[0]["dataset"], REGISTRY[1]["dataset"]]
    for name in names:
        _write(tmp_path, name, {"dataset": name, "shap": np.zeros((2, 2))})
    results = common.load_results(tmp_path)
    expected = sorted(ExpandedDatasets.display_name(n) for n in names)
    assert sorted(results) == expected and len(results) == 2
    assert results[expected[0]]["shap"].shape == (2, 2)


def test_artifact_dir_lists_only_finished_registry_datasets(tmp_path):
    _write(tmp_path, REGISTRY[0]["dataset"], {})
    (tmp_path / "not_in_registry.pkl").write_bytes(b"x")
    assert len(common.load_results(tmp_path)) == 1


def test_shard_items_partition():
    items = list("abcdefg")
    parts = [common.shard_items(items, f"{i}/3") for i in range(3)]
    assert sorted(x for p in parts for x in p) == items
    assert common.shard_items(items, None) == items
    with pytest.raises(ValueError):
        common.shard_items(items, "3/3")


def test_safe_spearman():
    a = np.arange(10.0)
    assert common.safe_spearman(a, a[::-1]) == pytest.approx(-1.0)
    assert np.isnan(common.safe_spearman(a, np.ones(10)))


def test_write_pickle_atomic(tmp_path):
    path = tmp_path / "sub" / "x.pkl"
    common.write_pickle_atomic(path, {"a": 1})
    assert pickle.load(open(path, "rb")) == {"a": 1}
    assert not list(path.parent.glob("*.tmp"))
