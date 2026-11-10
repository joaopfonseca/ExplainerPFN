"""Offline tests for Stage A of the expanded benchmark (scripts/results/compute_ground_truth.py)."""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from explainerpfn.datasets import ExpandedDatasets
from explainerpfn.datasets import processing as reg

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script():
    sys.path[:0] = [str(_SCRIPTS), str(_SCRIPTS / "results")]
    spec = importlib.util.spec_from_file_location(
        "compute_ground_truth", _SCRIPTS / "results" / "compute_ground_truth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gt = _load_script()


def _frame(n=300, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    logit = X.iloc[:, 0] - 0.5 * X.iloc[:, 1]
    return X.assign(target=(logit + rng.normal(scale=0.5, size=n) > 0).astype(int))


@pytest.fixture
def datasets(tmp_path):
    rows = []
    for i, name in enumerate(["alpha", "beta", "gamma"]):
        df = _frame(seed=i)
        reg.write_cache(df, tmp_path / reg.cache_filename(name, 1000 + i))
        rows.append({"dataset": name, "openml_did": 1000 + i, "openml_version": 1,
                     "source": "uci", "n": len(df), "d": 3, "bucket": "[3,7)",
                     "n_classes": 2, "missing": 0, "sha256": reg.sha256_of(df)})
    return ExpandedDatasets(data_home=tmp_path, entries=rows,
                            download_if_missing=False).download()


def test_parse_shard():
    assert gt.parse_shard(None) == (0, 1)
    assert gt.parse_shard("2/5") == (2, 5)
    for bad in ("5/5", "-1/3"):
        with pytest.raises(ValueError):
            gt.parse_shard(bad)


def test_shards_partition_the_registry(datasets):
    parts = [gt.select_rows(datasets, (i, 2)) for i in range(2)]
    names = [r.dataset for part in parts for r in part]
    assert sorted(names) == ["alpha", "beta", "gamma"] and len(names) == 3


def test_choose_shap_method():
    table = {"random_forest": {"exact_max_d": 10, "fallback": "permutation"}}
    assert gt.choose_shap_method("random_forest", 10, "auto", table) == "exact"
    assert gt.choose_shap_method("random_forest", 11, "auto", table) == "permutation"
    assert gt.choose_shap_method("logreg", 14, "auto", table) == "exact"  # no rule -> exact
    assert gt.choose_shap_method("mlp", 14, "permutation") == "permutation"
    with pytest.raises(ValueError):
        gt.choose_shap_method("mlp", 3, "tree")


@pytest.mark.parametrize("family", ["logreg", "knn"])
def test_run_pair_schema_and_exact_additivity(datasets, family):
    art = gt.run_pair(datasets, "ALPHA", family, "exact")
    for key in ("X_train", "X_test", "y_train", "y_test", "base_model", "y_train_scores",
                "y_test_scores", "shap", "shap_time", "shap_method", "n_train", "n_test",
                "test_auc", "shap_std", "fit_time"):
        assert key in art
    assert art["shap"].shape == art["X_test"].shape
    assert (art["n_train"], art["n_test"]) == (210, 90)
    assert set(np.unique(art["y_test"])) <= {0, 1}
    # Exact SHAP is additive: base value (mean score on the background) + sum = score.
    model = art["base_model"]
    background = model.named_steps["scaler"].transform(art["X_train"])
    import shap
    base = model.steps[-1][1].predict_proba(
        shap.maskers.Independent(background, max_samples=gt.BACKGROUND_SAMPLES).data)[:, 1].mean()
    np.testing.assert_allclose(art["shap"].sum(axis=1) + base, art["y_test_scores"], atol=1e-8)


def test_run_pair_is_deterministic(datasets):
    a = gt.run_pair(datasets, "BETA", "logreg", "exact")
    b = gt.run_pair(datasets, "BETA", "logreg", "exact")
    np.testing.assert_array_equal(a["shap"], b["shap"])
    np.testing.assert_array_equal(a["y_test_scores"], b["y_test_scores"])


def test_permutation_method_approximates_exact(datasets):
    exact = gt.run_pair(datasets, "GAMMA", "logreg", "exact")["shap"]
    approx = gt.run_pair(datasets, "GAMMA", "logreg", "permutation", max_evals=400)
    assert approx["shap_method"] == "permutation" and approx["shap_max_evals"] == 400
    assert np.corrcoef(exact.ravel(), approx["shap"].ravel())[0, 1] > 0.99


def test_write_artifact_roundtrip_and_resume_path(tmp_path, datasets):
    art = gt.run_pair(datasets, "ALPHA", "logreg", "exact")
    path = gt.artifact_path(tmp_path, "logreg", "alpha")
    assert path == tmp_path / "logreg" / "alpha.pkl"
    gt.write_artifact(path, art)
    assert not list(path.parent.glob("*.tmp"))
    with open(path, "rb") as f:
        back = pickle.load(f)
    np.testing.assert_array_equal(back["shap"], art["shap"])


def test_run_log_header_written_once(tmp_path):
    log = tmp_path / "runs.csv"
    gt.append_log({"dataset": "A", "family": "logreg", "status": "ok"}, log)
    gt.append_log({"dataset": "B", "family": "knn", "status": "ok"}, log)
    back = pd.read_csv(log)
    assert back.dataset.tolist() == ["A", "B"] and list(back.columns) == gt.LOG_FIELDS
