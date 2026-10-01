"""Validation harness for the ExplainerPFN fixes.

Runs the end-to-end pipeline on a synthetic dataset for which the true Shapley
values are known analytically, and reports how well the produced explanations
recover them. This is meant to be run before and after retraining the model.

Usage:
    python scripts/eval_synthetic_shap.py [--model-path PATH] [--n-samples N]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainerpfn.base import ExplainerPFN  # noqa: E402


def _safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="notebooks/tabpfn-v2-regressor.ckpt",
    )
    parser.add_argument("--n-train", type=int, default=90)
    parser.add_argument("--n-test", type=int, default=30)
    parser.add_argument("--n-features", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    p = args.n_features
    X = rng.normal(size=(args.n_train + args.n_test, p))
    beta = np.zeros(p)
    beta[0] = 2.0
    beta[1] = 1.0
    y = X @ beta

    X_train, y_train = X[: args.n_train], y[: args.n_train]
    X_test, y_test = X[args.n_train :], y[args.n_train :]

    # Interventional Shapley values for a linear model with independent
    # Gaussian features: phi_j = beta_j * (x_j - E[x_j]).
    true_phi = (X_test - X_train.mean(axis=0)) * beta

    explainer = ExplainerPFN(model_path=args.model_path, device="cpu")
    explainer.fit(X_train, y_train)
    exp = explainer.predict(X_test, y_test)

    print(f"explanations shape: {np.asarray(exp).shape}")
    print("\nper-feature diagnostics (importance=beta):")
    for j in range(p):
        print(
            f"  feat {j} (beta={beta[j]:.2f}): "
            f"corr(exp, phi)={_safe_corr(exp[:, j], true_phi[:, j]): .3f}  "
            f"corr(exp, x_j)={_safe_corr(exp[:, j], X_test[:, j]): .3f}  "
            f"exp_std={exp[:, j].std():.3f}  "
            f"|exp|mean={np.abs(exp[:, j]).mean():.3f}"
        )

    # A good explainer assigns near-zero importance to irrelevant features.
    relevant = beta != 0
    if relevant.any() and (~relevant).any():
        rel_scale = np.abs(exp[:, relevant]).mean()
        irr_scale = np.abs(exp[:, ~relevant]).mean()
        print(
            f"\nirrelevant/relevant magnitude ratio: "
            f"{irr_scale / rel_scale:.3f} (lower is better; ideal ~0)"
        )

    corrected = explainer.apply_correction(
        y_test, np.asarray(exp).copy(), kind=["statistical", "additive"]
    )
    print("\nper-feature corr(corrected_exp, phi):")
    print(
        "  "
        + ", ".join(
            f"feat{j}={_safe_corr(corrected[:, j], true_phi[:, j]):.3f}" for j in range(p)
        )
    )


if __name__ == "__main__":
    main()
