"""Efficiency-residual analysis.

Quantifies Shapley *efficiency-axiom* adherence of ExplainerPFN outputs at each
post-processing stage, over the cached synthetic ablation datasets produced by
``notebooks/10-ablations.ipynb``.

The efficiency axiom requires, per instance i:
    sum_j phi_{ij} = y_hat_i - v ,  with v = mean(y_hat)  (the SHAP base value).

We report the *relative* efficiency residual
    | sum_j phi_{ij} - (y_hat_i - v) | / | y_hat_i - v |
per instance, aggregated (mean / median / 90th percentile) per dataset and pooled,
for four ExplainerPFN stages plus a SHAP sanity baseline:

    Raw                     : transformer output, no post-processing
                              (unscaled from the stored EPFN_raw = out / sqrt(n_cols))
    Statistical (affine)    : mean re-centering + variance rescaling only
    Additive                : instance-level efficiency step (Eq. 7) on column-centered raw
    Statistical+Additive    : full pipeline (affine then instance-level) -> residual ~ 0
    SHAP (sanity)           : ground-truth Shapley values -> residual ~ 0

Outputs go to ``notebooks/paper_results/``:
    csv/efficiency_residuals.csv                       per-dataset mean/median/p90 per stage
    tables/efficiency_residuals_summary.tex            pooled summary table
    figures/efficiency_residuals_by_features.pdf       median residual vs n_features
    figures/efficiency_residuals_by_observations.pdf   median residual vs n_samples

NOTE: the few-shot surrogate context (TabPFN_n2/n10) is intentionally omitted;
this analysis only covers the ExplainerPFN post-processing stages and the SHAP
sanity baseline.
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CACHES_DIR,
    CSV_DIR,
    FIGURES_DIR,
    PICKLES_DIR,
    TABLES_DIR,
    ensure_results_dirs,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DATA_PKL = CACHES_DIR / "ablations_synthetic_datasets.pkl"
RESULTS_PKL = PICKLES_DIR / "ablations_synthetic_datasets_results.pkl"

RANDOM_STATE = 42
EPS = 1e-9  # guard for |y_hat - v| ~ 0 in the relative residual denominator

# Stage name -> (key, source, unscale?, own_base?)
#   source:   "results" -> ablations_synthetic_datasets_results.pkl
#             "data"    -> ablations_synthetic_datasets.pkl
#   unscale:  EPFN_raw was stored as out / sqrt(n_cols); undo it to recover ŷ-units.
#   own_base: if True, measure residual against the method's *own* base value
#             v = mean(ŷ - Σφ) (used for SHAP, whose masker is X_train, so its base
#             value is mean(ŷ over X_train), not mean(ŷ) over the full set).
#             ExplainerPFN's base value IS mean(ŷ) (base_value_), so it uses the
#             common v = mean(ŷ).
STAGES = [
    ("Raw", "EPFN_raw", "results", True, False),
    ("Statistical (affine)", "EPFN_statistical", "results", False, False),
    ("Additive", "EPFN_additive", "results", False, False),
    ("Statistical+Additive", "EPFN_statistical+additive", "results", False, False),
    ("SHAP (sanity)", "shap", "data", False, True),
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.labelsize": 18,
    "font.size": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "serif",
})


def _arr(x):
    return np.asarray(x.values) if hasattr(x, "values") else np.asarray(x)


def relative_residual(phi, y_hat, v):
    """Per-instance relative efficiency residual |sum phi - (y-v)| / |y-v|.

    Returns (residuals, n_skipped) where instances with |y-v| < EPS are skipped.
    """
    phi = _arr(phi)
    y_hat = _arr(y_hat).ravel()
    resid = phi.sum(axis=1) - (y_hat - v)
    denom = np.abs(y_hat - v)
    valid = denom >= EPS
    rel = np.abs(resid[valid]) / denom[valid]
    return rel, int((~valid).sum())


def main():
    ensure_results_dirs()
    with open(DATA_PKL, "rb") as f:
        data = pickle.load(f)
    with open(RESULTS_PKL, "rb") as f:
        results = pickle.load(f)

    n_datasets = len(data["X"])
    print(f"Loaded {n_datasets} cached ablation datasets.")

    records = []
    pooled = {label: [] for label, *_ in STAGES}
    skipped = {label: 0 for label, *_ in STAGES}

    for i in range(n_datasets):
        n_feat = int(data["n_features"][i])
        n_samp = int(data["n_samples"][i])
        y_hat = _arr(data["y_pred"][i]).ravel()
        v = float(np.mean(y_hat))
        n_cols = _arr(results["EPFN_raw"][i]).shape[1]  # attribution columns

        rec = {"n_features": n_feat, "n_samples": n_samp, "n_cols": n_cols}
        for label, key, source, unscale, own_base in STAGES:
            phi = _arr(data[key][i]) if source == "data" else _arr(results[key][i])
            if unscale:  # EPFN_raw was stored as out / sqrt(n_cols); undo it
                phi = phi * np.sqrt(n_cols)
            v_stage = float(np.mean(y_hat - phi.sum(axis=1))) if own_base else v
            rel, n_skip = relative_residual(phi, y_hat, v_stage)
            rec[f"{label}_mean"] = float(rel.mean()) if rel.size else np.nan
            rec[f"{label}_median"] = float(np.median(rel)) if rel.size else np.nan
            rec[f"{label}_p90"] = float(np.percentile(rel, 90)) if rel.size else np.nan
            pooled[label].append(np.median(rel) if rel.size else np.nan)
            skipped[label] += n_skip
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    df.to_csv(CSV_DIR / "efficiency_residuals.csv", index=False)
    print(f"Wrote per-dataset CSV: {CSV_DIR / 'efficiency_residuals.csv'}")

    # ---- Sanity checks ----------------------------------------------------- #
    print("\nSanity check (pooled median of per-dataset medians, should be ~0 for "
          "Additive / Statistical+Additive / SHAP):")
    for label, *_ in STAGES:
        meds = [m for m in pooled[label] if not np.isnan(m)]
        print(f"  {label:<26} median-of-medians = {np.median(meds):.3e}  "
              f"(skipped instances: {skipped[label]})")

    # ---- Pooled summary table --------------------------------------------- #
    summary_rows = []
    for label, *_ in STAGES:
        meds = np.array([m for m in pooled[label] if not np.isnan(m)])
        summary_rows.append({
            "Stage": label,
            "Mean (of dataset medians)": meds.mean(),
            "Median (of dataset medians)": np.median(meds),
            "P90 (of dataset medians)": np.percentile(meds, 90),
        })
    summary = pd.DataFrame(summary_rows)
    summary_tex = summary.copy()
    for c in summary_tex.columns[1:]:
        summary_tex[c] = summary_tex[c].map(lambda x: f"{x:.3e}")

    caption = (
        "Relative efficiency-axiom residual of ExplainerPFN attributions at each "
        "post-processing stage, pooled over the 950 synthetic ablation datasets "
        "from Appendix~G. For each instance $i$ we report "
        "$|\\sum_j \\hat\\phi_{ij} - (\\hat y_i - v)| / |\\hat y_i - v|$ with "
        "$v = \\mathrm{mean}(\\hat y)$; the columns give the mean, median, and 90th "
        "percentile of the per-dataset medians (lower is better; 0 indicates exact "
        "efficiency). ``Raw'' is the transformer output (unscaled from the stored "
        "$\\mathrm{EPFN\\_raw} / \\sqrt{m}$); ``Statistical (affine)'' applies mean "
        "re-centering and variance rescaling only; ``Additive'' applies the "
        "instance-level efficiency step (Eq.~7); ``Statistical+Additive'' is the "
        "full pipeline; ``SHAP (sanity)'' is the ground-truth Shapley value "
        "($\\sim\\!10^{-10}$, confirming the metric). The instance-level step "
        "enforces efficiency exactly ($\\sim\\!10^{-16}$), while the affine-only "
        "stage leaves a residual of $\\sim\\!0.6$, confirming that Pearson-invariant "
        "post-processing cannot by itself satisfy the axiom."
    )
    tabular = summary_tex.to_latex(index=False, float_format="%.3e")
    table_env = (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:efficiency_residuals}\n"
        f"{tabular}"
        "\\end{table}\n"
    )
    with open(TABLES_DIR / "efficiency_residuals_summary.tex", "w") as f:
        f.write(table_env)
    print(f"\nPooled summary:\n{summary.to_string(index=False)}")
    print(f"Wrote LaTeX: {TABLES_DIR / 'efficiency_residuals_summary.tex'}")

    # ---- Figures: median relative residual vs Features / Observations ----- #
    # Log y-axis (with a floor) so the ~0 stages (Additive / Full / SHAP) are visible.
    PLOT_FLOOR = 1e-4
    metric_col = "median"  # per-dataset median relative residual
    labels = [s[0] for s in STAGES]

    for x_var, x_label, fname in [
        ("n_features", "Features", "efficiency_residuals_by_features.pdf"),
        ("n_samples", "Observations", "efficiency_residuals_by_observations.pdf"),
    ]:
        long = df.melt(id_vars=[x_var],
                       value_vars=[f"{l}_{metric_col}" for l in labels],
                       var_name="Stage", value_name="Relative residual")
        long["Stage"] = long["Stage"].str.replace(f"_{metric_col}", "", regex=False)
        long["Relative residual"] = long["Relative residual"].clip(lower=PLOT_FLOOR)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.lineplot(data=long, x=x_var, y="Relative residual", hue="Stage",
                     style="Stage", errorbar=None, ax=ax)
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Median relative efficiency residual")
        ax.legend(title=None, loc="best", fontsize=12)
        plt.tight_layout()
        out = FIGURES_DIR / fname
        plt.savefig(out)
        plt.close(fig)
        print(f"Wrote figure: {out}")


if __name__ == "__main__":
    main()