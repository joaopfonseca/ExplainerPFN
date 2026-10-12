"""Fair checkpoint benchmark across datasets, under the current protocol.

Evaluates one or more ExplainerPFN checkpoints against the SHAP references
stored in an experiment results pickle (e.g. results_across_datasets_mlp.pkl),
using exactly the metrics from notebooks/8-analysis-across-datasets.ipynb:

- fit on ``concat([X_train, X_test])`` with the model scores,
- ``predict(X_test, y_test_scores)``,
- ``apply_correction(y_test_scores, exp, kind=["statistical", "additive"])``,
- Pearson correlation and top-k Jaccard similarity vs the stored SHAP values.

Usage:
    python scripts/eval_checkpoint_benchmark.py \
        --results-pkl notebooks/results_across_datasets_mlp.pkl \
        --models stock=notebooks/tabpfn-v2-regressor.ckpt \
                 nov15=eval_models/explainerpfn_model_NOV_15.ckp \
        --output logs/checkpoint_benchmark.csv
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainerpfn.base import ExplainerPFN  # noqa: E402

_INFERENCE_PRECISION = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _safe_corr(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_k_jaccard_similarity(arr1, arr2, k=3):
    """Mean top-k Jaccard similarity between the absolute values of two arrays."""
    if arr1.ndim == 1:
        arr1 = arr1.reshape(1, -1)
    if arr2.ndim == 1:
        arr2 = arr2.reshape(1, -1)
    feats1 = np.argpartition(arr1, -k, axis=1)[:, -k:]
    feats2 = np.argpartition(arr2, -k, axis=1)[:, -k:]
    intersection = [
        np.intersect1d(r1, r2).shape[0] for r1, r2 in zip(feats1, feats2)
    ]
    union = [
        r1.shape[0] + r2.shape[0] - inter
        for r1, r2, inter in zip(feats1, feats2, intersection)
    ]
    return np.array(intersection) / np.array(union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-pkl",
        default="notebooks/results_across_datasets_mlp.pkl",
        help="Pickle mapping dataset name -> experiment results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint(s) to evaluate as NAME=PATH.",
    )
    parser.add_argument("--output", default="logs/checkpoint_benchmark.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--inference-precision",
        default="auto",
        help="TabPFN inference precision; use float32 on GPU to match CPU "
        "numerics (the paper protocol).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--jaccard-perc", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--mode",
        choices=["per-feature", "joint"],
        default="per-feature",
        help="Decoding path. 'joint' loads each checkpoint's AttributionHead "
        "and uses predict(mode='joint').",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names (default: all in the pickle).",
    )
    args = parser.parse_args()

    models = []
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"invalid --models entry {spec!r}; expected NAME=PATH")
        models.append((name, path))

    with open(args.results_pkl, "rb") as f:
        results = pickle.load(f)

    inference_precision = _INFERENCE_PRECISION.get(
        args.inference_precision, args.inference_precision
    )

    dataset_names = (
        args.datasets if args.datasets else sorted(results.keys())
    )

    rows = []
    print(f"{'dataset':24s} {'model':14s} {'corr':>7s} {'jaccard':>8s}", flush=True)
    for dataset_name in dataset_names:
        r = results[dataset_name]
        X_test, y_test = r["X_test"], r["y_test_scores"]
        X_fit = np.concatenate([r["X_train"], X_test], axis=0)
        y_fit = np.concatenate([r["y_train_scores"], y_test], axis=0)
        shap_ref = r["shap"]
        k = max(1, int(X_test.shape[1] * args.jaccard_perc))

        for label, model_path in models:
            corr = jac = float("nan")
            try:
                xai = ExplainerPFN(
                    model_path=model_path,
                    random_state=args.random_state,
                    device=args.device,
                    inference_precision=inference_precision,
                )
                xai.fit(X_fit, y_fit)
                # A joint head embedded in the checkpoint is auto-loaded by
                # ``fit``; ``--mode joint`` on a headless checkpoint raises a
                # clear error from ``predict``.
                exp = xai.predict(X_test, y_test, mode=args.mode)
                corrected = xai.apply_correction(
                    y_test, exp, kind=["statistical", "additive"]
                )
                corr = _safe_corr(corrected, shap_ref)
                jac = float(
                    np.mean(
                        top_k_jaccard_similarity(
                            np.abs(corrected), np.abs(shap_ref), k=k
                        )
                    )
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  !! {dataset_name} / {label} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )

            rows.append(
                {
                    "dataset": dataset_name,
                    "model": label,
                    "corr": corr,
                    "jaccard": jac,
                    "fit_rows": int(X_fit.shape[0]),
                    "n_features": int(X_test.shape[1]),
                }
            )
            print(
                f"{dataset_name:24s} {label:14s} {corr:7.3f} {jac:8.3f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model", "corr", "jaccard", "fit_rows", "n_features"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {output}")

    # Per-model summary
    print("\nPer-model means:")
    print(f"{'model':14s} {'mean corr':>10s} {'mean jaccard':>13s}")
    for label, _ in models:
        model_rows = [row for row in rows if row["model"] == label]
        corrs = [row["corr"] for row in model_rows if np.isfinite(row["corr"])]
        jacs = [row["jaccard"] for row in model_rows if np.isfinite(row["jaccard"])]
        mean_corr = float(np.mean(corrs)) if corrs else float("nan")
        mean_jac = float(np.mean(jacs)) if jacs else float("nan")
        print(f"{label:14s} {mean_corr:10.3f} {mean_jac:13.3f}")


if __name__ == "__main__":
    main()
