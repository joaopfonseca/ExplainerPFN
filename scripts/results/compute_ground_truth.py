"""Stage A of the expanded benchmark: base models and SHAP ground truth.

For every (dataset, base-model family) pair of the frozen registry
(``explainerpfn.datasets.ExpandedDatasets``) this script

    1. splits the data with the benchmark rule (70/30, test set capped at 2,000 rows,
       ``random_state=42``),
    2. fits the base model (a scaler followed by the classifier),
    3. computes the SHAP ground truth of the test rows,

and stores one pickle per pair under ``<PICKLES_DIR>/ground_truth/<family>/``. The pickle
has the schema of the existing ``results_across_datasets_<family>.pkl`` entries (without
the ExplainerPFN / few-shot keys), so ``scripts/eval/benchmark_*.py`` can consume it.
This is the only expensive, checkpoint-independent step; evaluation reuses the artifacts.

SHAP method (``--shap-method``): ``exact`` (``shap.explainers.Exact``, the paper's ground
truth), ``permutation`` (approximate, for pairs where exact SHAP is too slow), or ``auto`` (look up
``--method-table``, a frozen JSON ``{family: {"exact_max_d": int, "fallback": method}}``;
without a table every pair uses ``exact``). The method used is stored with each artifact.

    python scripts/results/compute_ground_truth.py --families logreg --datasets heart
    python scripts/results/compute_ground_truth.py --shard 3/46 --resume
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import shap
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import LOGS_DIR, PICKLES_DIR, add_repo_to_path  # noqa: E402

add_repo_to_path()

from explainerpfn.datasets import ExpandedDatasets  # noqa: E402
from explainerpfn.datasets.processing import normalize_name  # noqa: E402

RANDOM_STATE = 42
BACKGROUND_SAMPLES = 100  # shap's default for the Independent masker (as in notebook 8)
GROUND_TRUTH_DIR = PICKLES_DIR / "ground_truth"
RUN_LOG = LOGS_DIR / "benchmarks" / "ground_truth_runs.csv"
LOG_FIELDS = ["dataset", "family", "bucket", "n", "d", "n_train", "n_test", "shap_method",
              "test_auc", "shap_std", "fit_time", "shap_time", "status"]
SHAP_METHODS = ("exact", "permutation")


def _pipeline(name: str, estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), (name, estimator)])


# RF and MLP are the notebook-8 configurations; the others follow base_model_families.py.
FAMILIES = {
    "random_forest": _pipeline("rfc", RandomForestClassifier(
        n_estimators=500, random_state=RANDOM_STATE)),
    "xgboost": _pipeline("xgb", XGBClassifier(
        n_estimators=500, random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1)),
    "mlp": _pipeline("mlp", MLPClassifier(
        hidden_layer_sizes=(12, 12), max_iter=5000, random_state=RANDOM_STATE)),
    "logreg": _pipeline("logreg", LogisticRegression(
        max_iter=5000, random_state=RANDOM_STATE)),
    "knn": _pipeline("knn", KNeighborsClassifier(n_neighbors=15)),
}


def choose_shap_method(family: str, d: int, requested: str = "exact",
                       table: dict | None = None) -> str:
    """Resolve the SHAP method of one pair (``auto`` uses the frozen table)."""
    if requested != "auto":
        method = requested
    else:
        rule = (table or {}).get(family)
        if rule is None or d <= rule["exact_max_d"]:
            method = "exact"
        else:
            method = rule["fallback"]
    if method not in SHAP_METHODS:
        raise ValueError(f"unknown SHAP method {method!r}; use {SHAP_METHODS} or 'auto'")
    return method


def shap_values(model: Pipeline, family: str, X_train, X_test, method: str,
                max_evals: int = 1000) -> np.ndarray:
    """SHAP values of the positive class for ``X_test`` (shape ``(n_test, d)``).

    All methods share the Independent-masker background that ``Exact`` uses by default
    (100 rows sampled from the scaled training data), so they target the same quantity.
    """
    scaler, clf = model.named_steps["scaler"], model.steps[-1][1]
    Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)
    masker = shap.maskers.Independent(Xs_train, max_samples=BACKGROUND_SAMPLES)
    if method == "exact":
        explainer = shap.explainers.Exact(clf.predict_proba, Xs_train)
    elif method == "permutation":
        explainer = shap.explainers.Permutation(clf.predict_proba, masker, seed=RANDOM_STATE)
    else:
        raise ValueError(f"unknown SHAP method {method!r}")
    kwargs = {"max_evals": max_evals} if method == "permutation" else {}
    return explainer(Xs_test, **kwargs).values[:, :, -1]


def run_pair(datasets: ExpandedDatasets, name: str, family: str, method: str,
             max_evals: int = 1000) -> dict:
    """Split, fit and explain one pair; returns the artifact dictionary."""
    X_train, X_test, y_train, y_test = datasets.train_test_split(name)
    model = clone(FAMILIES[family])
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    scores_train = model.predict_proba(X_train)[:, 1]
    scores_test = model.predict_proba(X_test)[:, 1]
    t0 = time.time()
    shap_test = shap_values(model, family, X_train, X_test, method, max_evals)
    shap_time = time.time() - t0
    return {
        "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "base_model": model, "y_train_scores": scores_train, "y_test_scores": scores_test,
        "shap": shap_test, "shap_time": shap_time,
        # provenance and quality fields
        "dataset": name, "family": family, "shap_method": method,
        "shap_max_evals": max_evals if method == "permutation" else None,
        "fit_time": fit_time, "n_train": len(X_train), "n_test": len(X_test),
        "test_auc": float(roc_auc_score(y_test, scores_test)),
        "shap_std": float(shap_test.std()), "random_state": RANDOM_STATE,
        "versions": {"shap": shap.__version__, "sklearn": sklearn.__version__,
                     "numpy": np.__version__},
    }


def artifact_path(out_dir: Path, family: str, dataset: str) -> Path:
    return Path(out_dir) / family / f"{normalize_name(dataset)}.pkl"


def write_artifact(path: Path, artifact: dict) -> None:
    """Atomic write, so an interrupted job never leaves a truncated pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(artifact, f)
    tmp.replace(path)


def append_log(row: dict, log_path: Path = RUN_LOG) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def parse_shard(value: str | None) -> tuple[int, int]:
    """``"i/N"`` -> (i, N) with 0 <= i < N; ``None`` -> (0, 1)."""
    if value is None:
        return 0, 1
    i, n = (int(x) for x in value.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"shard {value!r} must satisfy 0 <= i < N")
    return i, n


def select_rows(datasets: ExpandedDatasets, shard: tuple[int, int]):
    """Registry rows of this shard (every N-th row of the sorted registry)."""
    rows = list(datasets.registry.itertuples())
    return [r for k, r in enumerate(rows) if k % shard[1] == shard[0]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--datasets", nargs="+", default="all", help="names (case-insensitive)")
    ap.add_argument("--buckets", nargs="+", default=None)
    ap.add_argument("--shard", default=None, help="i/N: this task's share of the datasets")
    ap.add_argument("--resume", action="store_true", help="skip pairs with an artifact")
    ap.add_argument("--shap-method", default="exact", choices=[*SHAP_METHODS, "auto"])
    ap.add_argument("--method-table", default=None, help="frozen JSON for --shap-method auto")
    ap.add_argument("--max-evals", type=int, default=1000, help="permutation SHAP budget")
    ap.add_argument("--out-dir", default=str(GROUND_TRUTH_DIR))
    ap.add_argument("--data-home", default=None)
    ap.add_argument("--log", default=str(RUN_LOG))
    args = ap.parse_args(argv)

    table = json.loads(Path(args.method_table).read_text()) if args.method_table else None
    datasets = ExpandedDatasets(names=args.datasets, buckets=args.buckets,
                                data_home=args.data_home).download()
    rows = select_rows(datasets, parse_shard(args.shard))
    pairs = [(row, family) for row in rows for family in args.families]
    print(f"{len(rows)} datasets x {len(args.families)} families = {len(pairs)} pairs")
    failures = 0
    bar = tqdm(pairs, desc="ground truth", unit="pair")
    for row, family in bar:
        name = datasets.display_name(row.dataset)
        bar.set_postfix_str(f"{name[:28]} / {family}")
        path = artifact_path(args.out_dir, family, row.dataset)
        if args.resume and path.exists():
            tqdm.write(f"skip   {name} / {family} (exists)")
            continue
        method = choose_shap_method(family, int(row.d), args.shap_method, table)
        log = {"dataset": name, "family": family, "bucket": row.bucket, "n": row.n,
               "d": row.d, "shap_method": method}
        try:
            artifact = run_pair(datasets, name, family, method, args.max_evals)
        except Exception as exc:  # keep the array task alive; the log records it
            failures += 1
            tqdm.write(f"FAILED {name} / {family}: {type(exc).__name__}: {exc}")
            append_log({**log, "status": f"error:{type(exc).__name__}"}, Path(args.log))
            continue
        write_artifact(path, artifact)
        append_log({**log, **artifact, "status": "ok"}, Path(args.log))
        tqdm.write(f"ok     {name} / {family} [{method}] n_test={artifact['n_test']} "
                   f"auc={artifact['test_auc']:.3f} fit={artifact['fit_time']:.1f}s "
                   f"shap={artifact['shap_time']:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
