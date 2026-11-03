"""Dataset characterization versus zero-shot performance.

Computes dataset statistics (correlation structure, dimensionality, pairwise
interaction depth) and correlates them with ExplainerPFN's zero-shot Pearson
correlation with SHAP.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CSV_DIR,
    DEFAULT_RESULTS,
    FIGURES_DIR,
    PICKLES_DIR,
    TABLES_DIR,
    ensure_results_dirs,
    load_results,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.labelsize": (10 / 8) * 18,
    "font.size": (10 / 8) * 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "figure.subplot.left": 0.098,
    "figure.subplot.right": 0.938,
    "figure.subplot.bottom": 0.12,
    "figure.subplot.top": 0.944,
    "figure.subplot.wspace": 0.071,
    "font.family": "serif",
})

RANDOM_STATE = 42


def compute_interaction_depth(X, n_bins=5):
    """Estimate average pairwise interaction via normalized mutual information.

    Discretizes each feature into quantile bins, then computes NMI for every
    feature pair and averages. Returns 0 if fewer than 2 features.
    """
    from sklearn.metrics import normalized_mutual_info_score

    n_features = X.shape[1]
    if n_features < 2:
        return 0.0

    X_disc = np.empty_like(X)
    for j in range(n_features):
        try:
            X_disc[:, j] = pd.qcut(X[:, j], q=n_bins, labels=False,
                                   duplicates='drop').astype(float)
        except ValueError:
            X_disc[:, j] = np.zeros(X.shape[0])

    nmi_vals = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            nmi = normalized_mutual_info_score(
                X_disc[:, i].astype(int), X_disc[:, j].astype(int))
            nmi_vals.append(nmi)

    return float(np.mean(nmi_vals)) if nmi_vals else 0.0


def compute_dataset_stats(X):
    """Compute statistics for a dataset's feature matrix."""
    n_samples, n_features = X.shape

    corr_matrix = np.corrcoef(X.T)

    mask = ~np.eye(n_features, dtype=bool)
    mean_abs_corr = np.mean(np.abs(corr_matrix[mask]))
    max_abs_corr = np.max(np.abs(corr_matrix[mask]))

    abs_corr = np.abs(corr_matrix)
    try:
        cond_number = np.linalg.cond(abs_corr)
    except np.linalg.LinAlgError:
        cond_number = np.nan

    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    min_eigenvalue = np.min(eigenvalues)

    interaction_depth = compute_interaction_depth(X)

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "condition_number": cond_number,
        "min_eigenvalue": min_eigenvalue,
        "interaction_depth": interaction_depth,
    }


def compute_performance(shap_values, predictions):
    """Compute Pearson and Spearman correlation between SHAP and predictions."""
    shap_flat = shap_values.flatten()
    pred_flat = predictions.flatten()
    pearson_r, _ = pearsonr(shap_flat, pred_flat)
    spearman_r, _ = spearmanr(shap_flat, pred_flat)
    return {"pearson": pearson_r, "spearman": spearman_r}


def characterize_datasets(results, base_model_name):
    """Characterize all datasets in a results dict."""
    records = []

    for dataset_name, data in results.items():
        # Combine train and test for characterization
        X = np.vstack([data["X_train"], data["X_test"]])

        # Compute dataset statistics
        stats = compute_dataset_stats(X)

        # Compute performance for ExplainerPFN zero-shot (statistical+additive)
        shap = data["shap"]
        epfn = data.get("explainerpfn_statistical+additive", data.get("explainerpfn"))
        perf = compute_performance(shap, epfn)

        # Also compute raw EPFN performance
        epfn_raw = data["explainerpfn"]
        perf_raw = compute_performance(shap, epfn_raw)

        record = {
            "dataset": dataset_name,
            "base_model": base_model_name,
            **stats,
            "epfn_pearson": perf["pearson"],
            "epfn_spearman": perf["spearman"],
            "epfn_raw_pearson": perf_raw["pearson"],
            "epfn_raw_spearman": perf_raw["spearman"],
        }
        records.append(record)

    return pd.DataFrame(records)


def main():
    ensure_results_dirs()
    print("Loading results...")
    rf_results = load_results(DEFAULT_RESULTS["random_forest"])
    mlp_results = load_results(DEFAULT_RESULTS["mlp"])

    print("Characterizing RF datasets...")
    df_rf = characterize_datasets(rf_results, "Random Forest")

    print("Characterizing MLP datasets...")
    df_mlp = characterize_datasets(mlp_results, "MLP")

    # Combine
    df = pd.concat([df_rf, df_mlp], ignore_index=True)

    # Save
    output_csv = CSV_DIR / "dataset_characterization.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved characterization to {output_csv}")

    # Also save as pickle for later use
    output_pkl = PICKLES_DIR / "dataset_characterization.pkl"
    df.to_pickle(output_pkl)

    # Compute correlations between stats and performance
    print("\n=== Correlation between dataset stats and EPFN performance ===")
    stats_cols = ["mean_abs_corr", "max_abs_corr", "interaction_depth", "condition_number", "min_eigenvalue", "n_features"]

    for stat_col in stats_cols:
        valid = df[stat_col].notna() & ~np.isinf(df[stat_col].values)
        sub = df.loc[valid]
        if len(sub) < 3:
            print(f"{stat_col:20s}: insufficient data")
            continue
        r_pearson, _ = pearsonr(sub[stat_col], sub["epfn_pearson"])
        r_spearman, _ = spearmanr(sub[stat_col], sub["epfn_pearson"])
        print(f"{stat_col:20s}: Pearson={r_pearson:+.3f}, Spearman={r_spearman:+.3f}")

    # Generate plots: 1x4 row with key metrics, regression lines, legend on right
    print("\nGenerating plots...")

    from sklearn.linear_model import LinearRegression

    fig, axes = plt.subplots(1, 4, figsize=(21, 5))

    plot_configs = [
        ("mean_abs_corr", "Mean Abs. Corr.", "EPFN Pearson"),
        ("max_abs_corr", "Max Abs. Corr.", "EPFN Pearson"),
        ("n_features", "Number of Features", "EPFN Pearson"),
        ("interaction_depth", "Mean Pairwise MI", "EPFN Pearson"),
    ]

    for idx, (stat_col, xlabel, ylabel) in enumerate(plot_configs):
        ax = axes[idx]
        for base_model, color in zip(["Random Forest", "MLP"], ["#1f77b4", "#ff7f0e"]):
            sub = df[df["base_model"] == base_model]
            x_vals = sub[stat_col].values.reshape(-1, 1)
            y_vals = sub["epfn_pearson"].values.reshape(-1, 1)
            ax.scatter(x_vals, y_vals, color=color, alpha=0.7, s=80,
                       label=base_model if idx == 0 else None)

            ax.set_xlabel(xlabel)

        # Single trend line using all points (RF + MLP combined)
        all_x = df[stat_col].dropna().values.reshape(-1, 1)
        all_y = df.loc[df[stat_col].notna(), "epfn_pearson"].values.reshape(-1, 1)
        if len(all_x) >= 2:
            model = LinearRegression().fit(all_x, all_y)
            x_range = np.linspace(all_x.min(), all_x.max(), 100).reshape(-1, 1)
            y_pred = model.predict(x_range)
            ax.plot(x_range, y_pred, linestyle='--', color='red',
                    linewidth=2, label='Trend line' if idx == 0 else None)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Place a single legend to the right of the fourth subplot
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    handles, labels = zip(*unique)

    desired = ['Random Forest', 'MLP', 'Trend line']
    ordered = []
    for lbl in desired:
        idx = labels.index(lbl)
        ordered.append(handles[idx])
    fig.legend(ordered, desired, loc='center left', bbox_to_anchor=(0.91, 0.55),
               frameon=False, fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plot_path = FIGURES_DIR / "dataset_characterization.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {plot_path}")

    # Save a combined summary table for LaTeX
    summary = df.groupby("dataset").agg({
        "n_samples": "first",
        "n_features": "first",
        "mean_abs_corr": "mean",
        "max_abs_corr": "mean",
        "interaction_depth": "mean",
        "epfn_pearson": "mean",
        "epfn_spearman": "mean",
    })
    abbrev = {
        "BANKNOTE AUTHENTICATION": "BA",
        "ABALONE": "AB",
        "ANNEALING": "AN",
        "CONTRACEPTIVE": "CC",
        "ECHOCARDIOGRAM": "EC",
        "FLAGS": "FL",
        "GERMAN CREDIT": "GC",
        "HEART": "HE",
        "HEART DISEASE": "HD",
        "HEPATITIS": "HP",
        "THYROID": "TH",
    }
    display = summary.rename(
        columns={
            "n_features": "$m$",
            "mean_abs_corr": "Mean Abs. Corr.",
            "max_abs_corr": "Max Abs. Corr.",
            "interaction_depth": "Interaction Depth",
            "epfn_pearson": "Pearson",
            "epfn_spearman": "Spearman",
        }
    )[
        ["$m$", "Mean Abs. Corr.", "Max Abs. Corr.", "Interaction Depth", "Pearson", "Spearman"]
    ]
    display.index = display.index.map(lambda name: abbrev.get(name, name))
    display = display.loc[
        [short for short in abbrev.values() if short in display.index]
    ]
    display.index.name = "Dataset"
    summary_tex = TABLES_DIR / "dataset_characterization_summary.tex"
    latex = display.round(3).to_latex(
        index=True, index_names=False, float_format="%.3f"
    )
    latex = latex.replace(" & $m$", "Dataset & $m$", 1)
    summary_tex.write_text(latex)
    print(f"Saved LaTeX summary to {summary_tex}")


if __name__ == "__main__":
    main()
