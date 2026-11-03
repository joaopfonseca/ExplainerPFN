"""Synthetic-to-real distribution gap.

Compares statistics of the synthetic training datasets against the real UCI
evaluation datasets.
"""
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CSV_DIR,
    DEFAULT_RESULTS,
    FIGURES_DIR,
    TABLES_DIR,
    add_repo_to_path,
    ensure_results_dirs,
)

add_repo_to_path()

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
N_SYNTHETIC_SAMPLES = 500  # Number of synthetic datasets to sample
N_OBSERVATIONS_SYNTH = 500  # Number of observations per synthetic dataset


def compute_dataset_stats(X):
    """Compute statistics for a feature matrix."""
    n_samples, n_features = X.shape

    # Correlation matrix
    corr_matrix = np.corrcoef(X.T)
    mask = ~np.eye(n_features, dtype=bool)
    mean_abs_corr = np.mean(np.abs(corr_matrix[mask])) if n_features > 1 else 0
    max_abs_corr = np.max(np.abs(corr_matrix[mask])) if n_features > 1 else 0

    # Standard deviations
    feature_stds = np.std(X, axis=0)
    mean_std = np.mean(feature_stds)

    # Skewness (proxy for non-normality)
    feature_skews = pd.DataFrame(X).skew().values
    mean_skew = np.mean(np.abs(feature_skews))

    return {
        "n_features": n_features,
        "mean_abs_corr": mean_abs_corr,
        "max_abs_corr": max_abs_corr,
        "mean_std": mean_std,
        "mean_skew": mean_skew,
    }


def sample_synthetic_datasets(n_datasets, n_observations, random_state):
    """Sample synthetic datasets and compute their statistics."""
    from explainerpfn.train import DAGGenerator
    from explainerpfn.train.synthetic_data import _DEFAULT_KWARGS

    # Sample from the same generator configuration used for training, but skip
    # the (expensive) exact-Shapley labels: only the feature matrix is needed.
    config = dict(_DEFAULT_KWARGS)
    config["n_features_range"] = (2, 15)
    config["n_samples_range"] = (max(10, n_observations - 100), n_observations)

    records = []
    for i in tqdm(range(n_datasets), desc="Sampling synthetic datasets"):
        try:
            for attempt in range(100):
                generator = DAGGenerator(
                    random_state=int(random_state) + 1000 * i + attempt,
                    **config,
                )
                X = np.asarray(generator.generate_dataset(include_shap=False)["X"])
                if 2 <= X.shape[1] <= 15:
                    break
            else:
                raise RuntimeError("could not sample a dataset in the 2-15 range")
            stats = compute_dataset_stats(X)
            stats["dataset_type"] = "synthetic"
            stats["dataset_id"] = f"synth_{i}"
            records.append(stats)
        except Exception as e:
            print(f"  Error on synthetic dataset {i}: {e}")

    return pd.DataFrame(records)


def load_uci_stats():
    """Load UCI dataset statistics from existing pickles."""
    with open(DEFAULT_RESULTS["random_forest"], "rb") as f:
        rf_results = pickle.load(f)
    records = []
    for dataset_name in rf_results:
        X = np.vstack([rf_results[dataset_name]["X_train"], rf_results[dataset_name]["X_test"]])
        stats = compute_dataset_stats(X)
        stats["dataset_type"] = "uci"
        stats["dataset_id"] = dataset_name
        records.append(stats)

    return pd.DataFrame(records)


def main():
    ensure_results_dirs()
    print("Sampling synthetic datasets...")
    df_synth = sample_synthetic_datasets(N_SYNTHETIC_SAMPLES, N_OBSERVATIONS_SYNTH, RANDOM_STATE)

    print("Loading UCI dataset statistics...")
    df_uci = load_uci_stats()

    # Combine
    df = pd.concat([df_synth, df_uci], ignore_index=True)

    # Save
    output_csv = CSV_DIR / "synthetic_to_real_gap.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    # Generate comparison plots
    stats_to_compare = ["mean_abs_corr", "max_abs_corr", "mean_skew"]
    stat_labels_short = {
        "mean_abs_corr": "Mean Abs. Corr.",
        "max_abs_corr": "Max Abs. Corr.",
        "mean_skew": "Mean Abs. Skew.",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes = np.atleast_1d(axes)

    for ax, stat in zip(axes, stats_to_compare):
        synth_vals = df[df["dataset_type"] == "synthetic"][stat]
        uci_vals = df[df["dataset_type"] == "uci"][stat]

        if synth_vals.nunique() <= 2 or synth_vals.std() < 1e-6:
            print(f"  Plotting {stat}: synthetic has near-zero variance, using vertical line")
            sns.histplot(uci_vals, ax=ax, color="red", alpha=0.5, label="UCI", kde=True, stat="density")
            x_range = uci_vals.max() - uci_vals.min()
            synth_val = synth_vals.mean()
            if x_range > 10 * abs(synth_val) + 1e-6:
                ymax = ax.get_ylim()[1]
                ax.scatter([synth_val], [ymax * 0.85], color="blue", marker="*", s=400, zorder=5, label="Synthetic")
                ax.annotate(f'Synthetic\n({synth_val:.1f})', xy=(synth_val, ymax*0.85),
                            xytext=(synth_val + x_range*0.05, ymax*0.75),
                            fontsize=14, color="blue", ha="left",
                            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5))
            else:
                ax.axvline(synth_vals.mean(), color="blue", linestyle="--", linewidth=2, label="Synthetic")
        else:
            sns.histplot(synth_vals, ax=ax, color="blue", alpha=0.5, label="Synthetic", kde=True, stat="density")
            sns.histplot(uci_vals, ax=ax, color="red", alpha=0.5, label="UCI", kde=True, stat="density")

        ax.set_xlabel(stat_labels_short[stat])
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.3)

    # Single shared legend to the right of the third plot, no frame
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.885, 0.55),
               frameon=False, fontsize=18)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plot_path = FIGURES_DIR / "synthetic_to_real_gap_histograms.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # Summary statistics table
    summary = df.groupby("dataset_type")[stats_to_compare].agg(
        ["mean", "std", "median"]
    ).round(3)
    summary = summary.rename(
        columns={
            "mean_abs_corr": "Mean Abs. Corr.",
            "max_abs_corr": "Max Abs. Corr.",
            "mean_skew": "Mean Abs. Skew.",
        },
        level=0,
    )
    summary.index = summary.index.map({"synthetic": "Synthetic", "uci": "UCI"})
    summary.index.name = "Dataset"
    print("\n=== Summary Statistics ===")
    print(summary)

    summary_tex = TABLES_DIR / "synthetic_to_real_gap_summary.tex"
    summary.to_latex(summary_tex)
    print(f"Saved summary table to {summary_tex}")


if __name__ == "__main__":
    main()
