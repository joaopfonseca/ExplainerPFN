"""Validate the online exact-do-Shapley training data.

Generates datasets in memory with :class:`SyntheticDataGenerator` and reports
distributions over feature count, sample count, cells, SHAP tier, and the
efficiency residual. Optionally contrasts against a legacy pickle produced by
the old surrogate-SHAP pipeline (plain ``pickle`` — no legacy code required).

Usage:
    python scripts/validate_training_data.py --n-datasets 20
    python scripts/validate_training_data.py --n-datasets 20 \\
        --legacy-pkl ../synthetic_data_sample.pkl
"""

import argparse
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.append(".")

from explainerpfn.train.synthetic_data import SyntheticDataGenerator  # noqa: E402


def _efficiency_residual(shap, p_hat):
    return float(np.abs(shap.sum(axis=1) - (p_hat - p_hat.mean())).mean())


def summarize_online(n_datasets, seed, **kwargs):
    gen = SyntheticDataGenerator(random_state=seed, **kwargs)
    ds, ns, cells, tiers, residuals, scales = [], [], [], Counter(), [], []
    for i in range(n_datasets):
        d = gen.generate_one(i)
        n_features = d["X"].shape[1]
        n_samples = d["X"].shape[0]
        ds.append(n_features)
        ns.append(n_samples)
        cells.append(n_features * n_samples)
        tiers[d["params"]["tier"]] += 1
        residuals.append(_efficiency_residual(d["shap"].values, d["y_pred"]))
        scales.append(float(d["shap"].values.std()))

    print(f"=== online exact-do-Shapley generator ({n_datasets} datasets) ===")
    print(f"n_features : min={min(ds)} median={int(np.median(ds))} max={max(ds)}")
    print(f"n_samples  : min={min(ns)} median={int(np.median(ns))} max={max(ns)}")
    print(f"cells      : min={min(cells)} median={int(np.median(cells))} max={max(cells)}")
    print(f"tiers      : {dict(tiers)}")
    print(f"eff. resid : mean={np.mean(residuals):.4f} max={np.max(residuals):.4f}")
    print(f"shap std   : mean={np.mean(scales):.4f}")
    assert 3 <= min(ds) and max(ds) <= 15, "feature range invariant violated!"
    assert set(tiers) <= {"closed_form", "coalition_enumeration", "irreducible_sets"}, (
        f"unexpected (non-exact) tier present: {tiers}"
    )
    print("OK: all datasets in exact-label tiers, d in [3, 15]")


def summarize_legacy(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    explanations = data["explanations"]
    y_pred = data["y_pred"]
    n_features = [e.shape[1] for e in explanations]
    n_samples = [e.shape[0] for e in explanations]
    residuals, ind_counts = [], []
    for e, p in zip(explanations, y_pred):
        residuals.append(_efficiency_residual(np.asarray(e), np.asarray(p)))
        ind_counts.append(sum(str(c).startswith("ind_") for c in e.columns))

    print(f"\n=== legacy surrogate-SHAP pkl ({len(explanations)} datasets) ===")
    print(
        f"n_features : min={min(n_features)} median={int(np.median(n_features))} "
        f"max={max(n_features)}"
    )
    print(
        f"n_samples  : min={min(n_samples)} median={int(np.median(n_samples))} "
        f"max={max(n_samples)}"
    )
    print(
        f"eff. resid : mean={np.mean(residuals):.4f} max={np.max(residuals):.4f} "
        "(vs noisy y_pred)"
    )
    print(f"ind_ cols  : total={sum(ind_counts)} (0 => legacy ind/dep mislabelling bug was live)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-datasets", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--legacy-pkl", type=str, default=None)
    parser.add_argument(
        "--mc-samples", type=int, default=None, help="Override v(S) MC draws (speed)."
    )
    args = parser.parse_args()

    kwargs = {}
    if args.mc_samples is not None:
        kwargs["mc_samples"] = args.mc_samples
    summarize_online(args.n_datasets, args.seed, **kwargs)
    if args.legacy_pkl:
        summarize_legacy(args.legacy_pkl)


if __name__ == "__main__":
    main()
