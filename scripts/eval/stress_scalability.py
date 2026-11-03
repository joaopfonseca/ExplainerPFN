"""Scalability stress tests.

Generates synthetic datasets with 2 to 20 features, runs zero-shot
ExplainerPFN, and correlates its attributions with SHAP values. The upper bound
comes from the attribution head's ``max_features``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CSV_DIR,
    FIGURES_DIR,
    PICKLES_DIR,
    add_repo_to_path,
    ensure_results_dirs,
    resolve_model_path,
)

add_repo_to_path()

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

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

MODEL_PATH = resolve_model_path()
RANDOM_STATE = 42
N_OBSERVATIONS = 1000  # Increased for stability
N_FEATURES_LIST = list(range(2, 21, 2))  # 2, 4, 6, ..., 20 (head limit)
N_REPLICATES = 10  # Increased replicates for reliable error bars


def generate_synthetic_dataset(n_features, n_observations, random_state):
    """Generate a synthetic dataset with exactly ``n_features`` columns."""
    from explainerpfn.train import DAGGenerator

    low_nodes = max(2, n_features // 2)
    generator = DAGGenerator(
        n_features_range=[n_features, n_features],
        n_samples_range=[n_observations, n_observations],
        n_dags_range=[1, 1],
        nodes_per_dag_range=[low_nodes, max(low_nodes + 1, n_features)],
        random_state=random_state,
    )
    bundle = generator.generate_dataset(include_shap=False)

    X = np.asarray(bundle["X"])
    # The SCM sink is continuous; turn it into a balanced binary target so the
    # protocol matches the paper's classifier + KernelSHAP setup.
    y_continuous = np.asarray(bundle["y"]).ravel()
    y = (y_continuous > np.median(y_continuous)).astype(int)
    if X.shape[1] > 20:
        raise ValueError(
            f"generator produced {X.shape[1]} features (head limit is 20)"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )

    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(12, 12), max_iter=5000, random_state=random_state))
    ])
    base_model.fit(X_train, y_train)
    y_train_scores = base_model.predict_proba(X_train)[:, 1]
    y_test_scores = base_model.predict_proba(X_test)[:, 1]

    # Compute exact SHAP
    def model_fn(X):
        return base_model.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(model_fn, base_model.named_steps["scaler"].transform(X_train))
    shap_values = explainer.shap_values(base_model.named_steps["scaler"].transform(X_test), nsamples=100)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_scores": y_train_scores,
        "y_test_scores": y_test_scores,
        "shap": shap_values,
        "base_model": base_model,
    }


def run_explainerpfn(X_train, X_test, y_train_scores, y_test_scores, model_path):
    """Run ExplainerPFN zero-shot inference."""
    from explainerpfn.base import ExplainerPFN

    xai = ExplainerPFN(model_path=model_path, random_state=RANDOM_STATE)
    xai.fit(X_train, y_train_scores)
    exp_values = xai.predict(X_test, y_test_scores)
    exp_values = xai.apply_correction(
        y_test_scores, exp_values - exp_values.mean(axis=0), kind=["statistical", "additive"]
    )
    return exp_values


def main():
    ensure_results_dirs()
    print("Starting scalability stress tests...")
    print(f"Feature dimensions to test: {N_FEATURES_LIST}")
    print(f"Replicates per level: {N_REPLICATES}")

    records = []

    for n_features in tqdm(N_FEATURES_LIST, desc="Dimensionality levels"):
        for rep in range(N_REPLICATES):
            random_state = RANDOM_STATE + n_features * 100 + rep

            try:
                data = generate_synthetic_dataset(
                    n_features=n_features,
                    n_observations=N_OBSERVATIONS,
                    random_state=random_state,
                )

                epfn_values = run_explainerpfn(
                    data["X_train"], data["X_test"],
                    data["y_train_scores"], data["y_test_scores"],
                    MODEL_PATH,
                )

                shap_flat = data["shap"].flatten()
                epfn_flat = epfn_values.flatten()

                pearson_r, _ = pearsonr(shap_flat, epfn_flat)
                spearman_r, _ = spearmanr(shap_flat, epfn_flat)

                records.append({
                    "n_features": n_features,
                    "replicate": rep,
                    "pearson": pearson_r,
                    "spearman": spearman_r,
                    "random_state": random_state,
                })

                print(f"  n={n_features}, rep={rep}: Pearson={pearson_r:.3f}, Spearman={spearman_r:.3f}")

            except Exception as e:
                print(f"  ERROR n={n_features}, rep={rep}: {e}")
                records.append({
                    "n_features": n_features,
                    "replicate": rep,
                    "pearson": np.nan,
                    "spearman": np.nan,
                    "random_state": random_state,
                    "error": str(e),
                })

    df = pd.DataFrame(records)

    # Save results
    output_csv = CSV_DIR / "scalability_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    output_pkl = PICKLES_DIR / "scalability_results.pkl"
    df.to_pickle(output_pkl)

    # Generate plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(axes, ["pearson", "spearman"], ["Pearson", "Spearman"]):
        df_summary = df.groupby("n_features")[metric].agg(["mean", "std"]).reset_index()
        ax.plot(df_summary["n_features"], df_summary["mean"], marker="o")
        ax.fill_between(
            df_summary["n_features"],
            df_summary["mean"] - df_summary["std"],
            df_summary["mean"] + df_summary["std"],
            alpha=0.3,
        )
        ax.set_xlabel("Number of Features")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "scalability_dimensionality.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
