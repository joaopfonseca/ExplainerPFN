"""Fair checkpoint benchmark across datasets, under the current protocol.

Evaluates one or more ExplainerPFN checkpoints against the SHAP references
stored in an experiment results pickle (e.g. results_across_datasets_mlp.pkl),
using exactly the metrics from notebooks/8-analysis-across-datasets.ipynb:

- fit on ``concat([X_train, X_test])`` with the model scores,
- ``predict(X_test, y_test_scores)``,
- ``apply_correction(y_test_scores, exp, kind=["statistical", "additive"])``,
- Pearson correlation and top-k Jaccard similarity vs the stored SHAP values.

Usage::

    python scripts/eval/benchmark_checkpoints.py \
        --results-pkl notebooks/paper_results/pickles/results_across_datasets_mlp.pkl \
        --models v5=checkpoints/final_model.ckp \
        --output logs/checkpoint_benchmark.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    LOGS_DIR,
    add_repo_to_path,
    DEFAULT_RESULTS,
    load_results,
    safe_corr,
    safe_spearman,
    shard_items,
    top_k_jaccard_similarity,
    write_pickle_atomic,
)

add_repo_to_path()

from explainerpfn.base import ExplainerPFN  # noqa: E402

_INFERENCE_PRECISION = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _prediction_path(save_dir, label, dataset_name):
    """``DIR/<model>/<dataset>.pkl`` (None when predictions are not saved)."""
    if not save_dir:
        return None
    slug = "".join(c for c in dataset_name.lower() if c.isalnum())
    return Path(save_dir) / label / f"{slug}.pkl"


def _prediction_record(xai, dataset_name, label, exp, scores, elapsed):
    """Explanations under the keys of the results pickles (see base_model_families.py)."""
    return {
        "dataset": dataset_name,
        "model": label,
        "explainerpfn": exp / np.sqrt(exp.shape[1]),
        "explainerpfn_statistical+additive": xai.apply_correction(
            scores, exp, kind=["statistical", "additive"]
        ),
        "explainerpfn_additive": xai.apply_correction(
            scores, exp - exp.mean(axis=0), kind=["additive"]
        ),
        "explainerpfn_time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-pkl",
        default=str(DEFAULT_RESULTS["mlp"]),
        help="Pickle mapping dataset name -> experiment results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Checkpoint(s) to evaluate as NAME=PATH.",
    )
    parser.add_argument("--output", default=str(LOGS_DIR / "checkpoint_benchmark.csv"))
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
        choices=["auto", "per-feature", "joint"],
        default="auto",
        help="Decoding path. 'auto' uses the checkpoint's AttributionHead "
        "(joint) when present and per-feature otherwise.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names (default: all in the pickle).",
    )
    parser.add_argument(
        "--correction-clamp",
        type=float,
        default=None,
        help="Residual models only: bound each row's correction norm to "
        "clamp * ||prior|| (safety against over-correction).",
    )
    parser.add_argument(
        "--order-ensemble",
        type=int,
        default=1,
        help="Average predictions over this many random feature-column orders "
        "(fit and predict are re-run per order; outputs are un-permuted "
        "before averaging). 1 = disabled.",
    )
    parser.add_argument(
        "--corrections",
        nargs="*",
        default=["statistical", "additive"],
        help="Explanation corrections applied before scoring, matching the "
        "paper protocol. Use 'none' to score raw predictions.",
    )
    parser.add_argument(
        "--shard",
        default=None,
        help="i/N: evaluate every N-th dataset (for SLURM arrays).",
    )
    parser.add_argument(
        "--save-predictions",
        default=None,
        metavar="DIR",
        help="Also store the explanations of every (model, dataset) as "
        "DIR/<model>/<dataset>.pkl with the keys of the results pickles "
        "(explainerpfn, explainerpfn_statistical+additive, explainerpfn_additive, "
        "explainerpfn_time), so tables and bootstrap CIs can be built later.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="With --save-predictions: skip (model, dataset) pairs already stored.",
    )
    args = parser.parse_args()

    models = []
    for spec in args.models:
        name, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"invalid --models entry {spec!r}; expected NAME=PATH")
        models.append((name, path))

    # A results pickle, or a ground-truth artifact directory (one pickle per dataset).
    results = load_results(args.results_pkl)

    inference_precision = _INFERENCE_PRECISION.get(
        args.inference_precision, args.inference_precision
    )

    dataset_names = shard_items(
        args.datasets if args.datasets else sorted(results.keys()), args.shard
    )

    rows = []
    print(f"{'dataset':24s} {'model':14s} {'corr':>7s} {'jaccard':>8s}", flush=True)
    for dataset_name in dataset_names:
        if args.save_predictions and args.resume and all(
            _prediction_path(args.save_predictions, label, dataset_name).exists()
            for label, _ in models
        ):
            print(f"skip {dataset_name} (predictions exist)", flush=True)
            continue
        r = results[dataset_name]
        X_test, y_test = r["X_test"], r["y_test_scores"]
        X_fit = np.concatenate([r["X_train"], X_test], axis=0)
        y_fit = np.concatenate([r["y_train_scores"], y_test], axis=0)
        shap_ref = r["shap"]
        k = max(1, int(X_test.shape[1] * args.jaccard_perc))

        for label, model_path in models:
            corr = jac = spearman = elapsed = float("nan")
            pred_path = _prediction_path(args.save_predictions, label, dataset_name)
            if args.save_predictions and args.resume and pred_path.exists():
                continue
            try:
                t0 = time.time()
                n_feat = X_test.shape[1]
                preds = []
                for order_idx in range(max(1, args.order_ensemble)):
                    if args.order_ensemble <= 1:
                        perm = np.arange(n_feat)
                    else:
                        perm = np.random.default_rng(
                            args.random_state + order_idx
                        ).permutation(n_feat)
                    inv = np.argsort(perm)
                    xai = ExplainerPFN(
                        model_path=model_path,
                        random_state=args.random_state,
                        device=args.device,
                        inference_precision=inference_precision,
                        correction_clamp=args.correction_clamp,
                    )
                    xai.fit(X_fit[:, perm], y_fit)
                    # A joint head embedded in the checkpoint is auto-loaded
                    # by ``fit``; ``--mode joint`` on a headless checkpoint
                    # raises a clear error from ``predict``.
                    mode = (
                        args.mode
                        if args.mode != "auto"
                        else ("joint" if xai.head_ is not None else "per-feature")
                    )
                    preds.append(
                        xai.predict(X_test[:, perm], y_test, mode=mode)[:, inv]
                    )
                exp = preds[0] if len(preds) == 1 else np.mean(preds, axis=0)
                elapsed = time.time() - t0
                if args.save_predictions:
                    write_pickle_atomic(
                        pred_path,
                        _prediction_record(xai, dataset_name, label, exp, y_test, elapsed),
                    )
                if args.corrections == ["none"]:
                    corrected = exp
                else:
                    corrected = xai.apply_correction(
                        y_test, exp, kind=args.corrections
                    )
                corr = safe_corr(corrected, shap_ref)
                spearman = safe_spearman(corrected, shap_ref)
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
                    "spearman": spearman,
                    "fit_predict_time": elapsed,
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
            fieldnames=["dataset", "model", "corr", "jaccard", "fit_rows", "n_features",
                        "spearman", "fit_predict_time"],
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
