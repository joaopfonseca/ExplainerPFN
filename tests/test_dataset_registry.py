"""Offline tests for the expanded-benchmark datasets.

``explainerpfn.datasets`` (loader, processing helpers) is imported directly; the curation
code lives in ``scripts/datasets/build_registry.py`` and is loaded by path.
"""
import importlib.util
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from explainerpfn.datasets import ExpandedDatasets  # noqa: E402
from explainerpfn.datasets import processing as reg  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_curation():
    sys.path[:0] = [str(_SCRIPTS), str(_SCRIPTS / "datasets")]
    spec = importlib.util.spec_from_file_location(
        "build_registry", _SCRIPTS / "datasets" / "build_registry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cur = _load_curation()
UCI_CANDIDATES = cur.UCI_CANDIDATES


@pytest.mark.parametrize(
    "d,bucket",
    [(3, "[3,7)"), (6, "[3,7)"), (7, "[7,11)"), (10, "[7,11)"),
     (11, "[11,15)"), (14, "[11,15)"), (2, None), (15, None)],
)
def test_bucket_boundaries(d, bucket):
    assert reg.assign_bucket(d) == bucket


def _frame(seed=0, n=300, d=4):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    return X.assign(target=(rng.random(n) > 0.5).astype(int))


def test_candidate_manifest_has_38_unique():
    assert len(UCI_CANDIDATES) == 38 == len(set(UCI_CANDIDATES))


def test_binarize_binary_maps_min_to_zero():
    y = pd.Series([2, 1, 2, 2, 1])
    assert reg.binarize_target(y).tolist() == [1, 0, 1, 1, 0]


def test_binarize_multiclass_majority_vs_rest():
    y = pd.Series(["a", "b", "b", "c", "b", "a"])
    assert reg.binarize_target(y).tolist() == [0, 1, 1, 0, 1, 0]


def test_clean_features_drops_cat_id_const():
    X = pd.DataFrame({
        "id": np.arange(10), "const": 1.0, "cat": pd.Categorical(list("ababababab")),
        "x": np.random.default_rng(0).normal(size=10),
    })
    out, dropped = reg.clean_features(X)
    assert list(out.columns) == ["x"]
    assert sorted(dropped) == ["cat(cat)", "const(const)", "id(id)"]


def test_missing_fraction_and_median_imputation():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0], "b": [1.0, 2.0, 3.0, 4.0]})
    assert reg.missing_fraction(X) == pytest.approx(1 / 8)
    out = reg.impute_median(X)
    assert not out.isna().to_numpy().any()
    assert out.loc[1, "a"] == 3.0  # median of [1, 3, 5]


def test_finalize_applies_criteria():
    ok = reg.finalize(reg.Entry("a", "uci"), _frame(d=4), 2, [])
    assert ok.status == "included" and ok.bucket == "[3,7)" and ok.sha256
    small = reg.finalize(reg.Entry("b", "uci"), _frame(n=50), 2, [])
    assert small.status == "skipped" and "n=50" in small.notes
    wide = reg.finalize(reg.Entry("c", "uci"), _frame(d=20), 2, [])
    assert wide.status == "skipped" and "d=20" in wide.notes
    nan = _frame()
    nan.iloc[0, 0] = np.nan  # un-imputed NaN must never be frozen
    assert reg.finalize(reg.Entry("d", "uci"), nan, 2, []).status == "skipped"
    imputed = reg.finalize(reg.Entry("e", "uci"), _frame(), 2, [], n_imputed=12)
    assert imputed.status == "included" and imputed.missing == 12


def test_content_key_is_order_invariant_and_detects_duplicates():
    df = _frame()
    shuffled = df.sample(frac=1, random_state=1)[df.columns[::-1]]
    assert cur.content_key(df) == cur.content_key(shuffled)
    assert cur.content_key(df) != cur.content_key(_frame(seed=1))


def test_sha256_deterministic():
    assert reg.sha256_of(_frame()) == reg.sha256_of(_frame())


def test_cache_round_trip_and_byte_identity(tmp_path):
    df = _frame()
    a, b = tmp_path / "a.csv.gz", tmp_path / "b.csv.gz"
    reg.write_cache(df, a)
    reg.write_cache(df, b)
    assert a.read_bytes() == b.read_bytes()
    back = reg.read_cache(a)
    assert back.shape == df.shape
    assert reg.sha256_of(back) == reg.sha256_of(df)
    assert cur.content_key(back) == cur.content_key(df)


def test_plan_topup_is_deterministic_and_bounded():
    deficits = {"[3,7)": 3, "[7,11)": 0, "[11,15)": 5}
    pools = {"[3,7)": [1] * 10, "[7,11)": [1] * 10, "[11,15)": [1] * 2}
    a = cur.plan_topup(deficits, 8, pools)
    assert a == cur.plan_topup(deficits, 8, pools)
    assert a == {"[3,7)": 3, "[7,11)": 0, "[11,15)": 2}  # capped by deficit/pool


def _fake_registry(tmp_path, frames):
    """Registry rows + cache directory for ``frames`` = {name: (frame, bucket, source)}."""
    rows = []
    for i, (name, (df, bucket, source)) in enumerate(frames.items()):
        reg.write_cache(df, tmp_path / reg.cache_filename(name, 1000 + i))
        rows.append({"dataset": name, "openml_did": 1000 + i, "openml_version": 1,
                     "source": source, "n": len(df), "d": df.shape[1] - 1, "bucket": bucket,
                     "n_classes": 2, "missing": 0, "sha256": reg.sha256_of(df)})
    return rows


def test_expanded_datasets_loads_filters_and_indexes(tmp_path):
    frames = {"alpha_set": (_frame(0, d=4), "[3,7)", "uci"),
              "Beta": (_frame(1, d=8), "[7,11)", "openml-suite")}
    path = _fake_registry(tmp_path, frames)
    ds = ExpandedDatasets(data_home=tmp_path, entries=path,
                          download_if_missing=False).download()
    assert len(ds) == 2 and list(ds.keys()) == ["ALPHA SET", "BETA"]
    assert [name for name, _ in ds] == ["ALPHA SET", "BETA"]  # iterable like mlresearch
    assert ds["BETA"].shape == (300, 9) and ds[0][0] == "ALPHA SET"
    assert ds.summary()["n"].tolist() == [300, 300]

    only = ExpandedDatasets(buckets=["[7,11)"], data_home=tmp_path, entries=path)
    assert only.registry.dataset.tolist() == ["Beta"]
    named = ExpandedDatasets(names=["alpha set"], data_home=tmp_path, entries=path)
    assert named.registry.dataset.tolist() == ["alpha_set"]
    with pytest.raises(KeyError):
        ExpandedDatasets(names=["nope"], data_home=tmp_path, entries=path).registry
    with pytest.raises(ValueError):
        ExpandedDatasets(buckets=["[1,2)"], data_home=tmp_path, entries=path).registry


def test_expanded_datasets_detects_modified_or_missing_cache(tmp_path):
    path = _fake_registry(tmp_path, {"alpha": (_frame(0), "[3,7)", "uci")})
    cached = tmp_path / reg.cache_filename("alpha", 1000)
    reg.write_cache(_frame(5), cached)  # tampered content, sha256 no longer matches
    with pytest.raises(ValueError, match="sha256"):
        ExpandedDatasets(data_home=tmp_path, entries=path,
                         download_if_missing=False).download()
    cached.unlink()
    with pytest.raises(FileNotFoundError):
        ExpandedDatasets(data_home=tmp_path, entries=path,
                         download_if_missing=False).download()


def test_train_test_split_follows_benchmark_protocol(tmp_path):
    df = _frame(0).assign(target=lambda x: x.target + 1)  # raw labels {1, 2}
    path = _fake_registry(tmp_path, {"alpha": (df, "[3,7)", "uci")})
    ds = ExpandedDatasets(data_home=tmp_path, entries=path,
                          download_if_missing=False).download()
    X_tr, X_te, y_tr, y_te = ds.train_test_split("alpha")
    assert X_tr.shape == (210, 4) and X_te.shape == (90, 4)
    assert set(y_tr) | set(y_te) == {0, 1}


def test_default_data_home_respects_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(reg.DATA_HOME_ENV, str(tmp_path))
    assert reg.default_data_home() == tmp_path


def test_frozen_registry_is_consistent():
    registry = ExpandedDatasets().registry
    assert len(registry) == registry.dataset.nunique() == 46
    assert set(registry.bucket) == {"[3,7)", "[7,11)", "[11,15)"}
    assert (registry.bucket == registry.d.astype(int).map(reg.assign_bucket)).all()
    assert registry.sha256.str.len().eq(64).all()
    assert registry.n.astype(int).between(reg.MIN_N, reg.MAX_N).all()
    names = [ExpandedDatasets.display_name(d) for d in registry.dataset]
    assert len(set(names)) == len(names)


_LEGACY_CSV = Path.home() / "mlresearch_data" / "binary_banknote_authentication.csv"
_PICKLE = Path(__file__).resolve().parents[1] / (
    "notebooks/paper_results/pickles/results_across_datasets_mlp.pkl")


@pytest.mark.skipif(not (_LEGACY_CSV.exists() and _PICKLE.exists()),
                    reason="legacy cache or results pickle not available")
def test_legacy_banknote_matches_existing_pickle():
    import pickle

    old = pickle.load(open(_PICKLE, "rb"))["BANKNOTE AUTHENTICATION"]
    ds = ExpandedDatasets(names=["banknote authentication"]).download()
    split = ds.train_test_split("banknote authentication")
    for key, got in zip(("X_train", "X_test", "y_train", "y_test"), split):
        assert np.array_equal(got, old[key]), key
