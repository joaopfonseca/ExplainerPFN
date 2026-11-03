"""Few-shot (semi-supervised) surrogate benchmarks under the EPFN protocol.

Reproduces notebook 8's few-shot explainers -- a TabPFN / MLP / RF regressor
trained on ``n`` exact-SHAP examples to predict per-feature Shapley values --
and scores them with exactly the metrics and data handling of
``benchmark_checkpoints.py`` (Pearson corr + top-k Jaccard vs the stored
SHAP reference, same test split).

Two protocols are evaluated so the comparison is unambiguous:

* ``aligned``  -- the ``n`` examples are drawn from ``concat(X_train, X_test)``
  and the surrogate is conditioned on the base model's **scores** in addition
  to the features, i.e. the same information ExplainerPFN receives.
* ``original`` -- the ``n`` examples are drawn from ``X_train`` and the
  surrogate is conditioned on the **true labels** (as in notebook 8).

Usage::

    python scripts/eval/benchmark_surrogates.py \
        --results-pkl notebooks/paper_results/pickles/results_across_datasets_mlp.pkl \
        --output logs/bench_surrogates.csv
"""

import argparse
import csv
import pickle
import sys
from itertools import product
from pathlib import Path

import numpy as np
import shap
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    DEFAULT_RESULTS,
    LOGS_DIR,
    TABPFN_MODEL,
    safe_corr,
    top_k_jaccard_similarity,
)


def _make_predictors(device, model_path):
    return {
        "tabpfn": lambda: TabPFNRegressor(
            model_path=model_path, random_state=42, device=device
        ),
        "mlp": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(12, 12), max_iter=5000, random_state=42
                    ),
                ),
            ]
        ),
        "rfr": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "rfr",
                    RandomForestRegressor(n_estimators=500, random_state=42),
                ),
            ]
        ),
    }


def _few_shot(Xy_sample, targets_sample, Xy_test, predictor, n_features):
    preds = np.zeros((Xy_test.shape[0], n_features), dtype=float)
    model = clone(predictor)
    for i in range(n_features):
        model.fit(Xy_sample, targets_sample[:, i])
        preds[:, i] = model.predict(Xy_test)
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-pkl", default=str(DEFAULT_RESULTS["mlp"])
    )
    parser.add_argument("--output", default=str(LOGS_DIR / "bench_surrogates.csv"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tabpfn-model", default=str(TABPFN_MODEL))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-samples", nargs="+", type=int, default=[2, 4, 6, 8, 10])
    parser.add_argument("--jaccard-perc", type=float, default=1.0 / 3.0)
    parser.add_argument("--protocols", nargs="+", default=["aligned", "original"])
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.results_pkl, "rb") as f:
        results = pickle.load(f)

    predictors = _make_predictors(args.device, args.tabpfn_model)
    names = args.datasets if args.datasets else sorted(results.keys())

    rows = []
    for ds in names:
        r = results[ds]
        X_train, X_test = r["X_train"], r["X_test"]
        y_train, y_test = r["y_train"], r["y_test"]
        s_train, s_test = r["y_train_scores"], r["y_test_scores"]
        shap_ref = np.asarray(r["shap"], float)
        base_model = r["base_model"]
        scaler = base_model.named_steps["scaler"]
        predict_proba = base_model.steps[-1][1].predict_proba

        # Exact-SHAP oracle for the base model (background = scaled X_train).
        explainer = shap.explainers.Exact(
            predict_proba, scaler.transform(X_train)
        )
        X_fit = np.concatenate([X_train, X_test], axis=0)
        score_fit = np.concatenate([s_train, s_test], axis=0)
        k = max(1, int(X_test.shape[1] * args.jaccard_perc))

        for protocol, n in product(args.protocols, args.n_samples):
            if protocol == "aligned":
                pool_X, pool_lab = X_fit, score_fit
                test_lab = s_test
            elif protocol == "original":
                pool_X, pool_lab = X_train, y_train
                test_lab = y_test
            else:
                raise ValueError(protocol)

            X_s, _, lab_s, _ = train_test_split(
                pool_X, pool_lab, train_size=n, random_state=args.random_state
            )
            shap_s = explainer(scaler.transform(X_s)).values[:, :, -1]
            Xy_sample = np.concatenate([X_s, lab_s.reshape(-1, 1)], axis=1)
            Xy_test = np.concatenate([X_test, test_lab.reshape(-1, 1)], axis=1)

            for pname, factory in predictors.items():
                preds = _few_shot(
                    Xy_sample, shap_s, Xy_test, factory(), X_test.shape[1]
                )
                corr = safe_corr(preds, shap_ref)
                jac = float(
                    np.mean(
                        top_k_jaccard_similarity(np.abs(preds), np.abs(shap_ref), k=k)
                    )
                )
                rows.append(
                    {
                        "dataset": ds,
                        "protocol": protocol,
                        "predictor": pname,
                        "n": n,
                        "corr": corr,
                        "jaccard": jac,
                    }
                )
                print(
                    f"{ds:24s} {protocol:8s} {pname:6s} n={n:2d} "
                    f"corr={corr:6.3f} jac={jac:6.3f}",
                    flush=True,
                )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["dataset", "protocol", "predictor", "n", "corr", "jaccard"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out}")

    print("\nMeans (protocol x predictor x n):")
    for protocol in args.protocols:
        for pname in predictors:
            for n in args.n_samples:
                sub = [
                    row
                    for row in rows
                    if row["protocol"] == protocol
                    and row["predictor"] == pname
                    and row["n"] == n
                ]
                corrs = [row["corr"] for row in sub if np.isfinite(row["corr"])]
                print(
                    f"  {protocol:8s} {pname:6s} n={n:2d} "
                    f"mean corr={np.mean(corrs):.3f}"
                )


if __name__ == "__main__":
    main()
