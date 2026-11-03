"""Bootstrap confidence intervals for the attribution-quality metrics.

Loads the experiment result pickles, computes bootstrap 95% CIs for the
Pearson and Spearman correlations of every method, and saves the results for
``make_tables.py``.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CSV_DIR,
    DEFAULT_RESULTS,
    PICKLES_DIR,
    ensure_results_dirs,
)

RANDOM_STATE = 42
N_BOOTSTRAP = 50
CI_LEVEL = 0.95


def bootstrap_correlation_ci(shap_arr, pred_arr, n_bootstrap=50, random_state=42, metric="pearson", ci=0.95):
    """Compute bootstrap CI for correlation via instance-level resampling.

    Resamples instances (rows) with replacement, preserving the within-instance
    feature structure, then computes the correlation on the flattened vector.
    """
    rng = np.random.default_rng(random_state)
    n_instances = shap_arr.shape[0]
    corrs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_instances, size=n_instances)
        s = shap_arr[idx].flatten()
        p = pred_arr[idx].flatten()
        if metric == "pearson":
            corrs.append(np.corrcoef(s, p)[0, 1])
        elif metric == "spearman":
            corrs.append(spearmanr(s, p)[0])
    corrs = np.array(corrs)
    alpha = 1 - ci
    return {
        "mean": float(np.mean(corrs)),
        "lower": float(np.percentile(corrs, 100 * alpha / 2)),
        "upper": float(np.percentile(corrs, 100 * (1 - alpha / 2))),
        "std": float(np.std(corrs)),
    }


def compute_all_cis(results, base_model_name):
    """Compute bootstrap CIs for all methods in results dict."""
    records = []

    methods = {
        "EPFN": "explainerpfn_statistical+additive",
        "EPFN (raw)": "explainerpfn",
    }
    # Few-shot methods
    for predictor in ["mlp", "rfr", "tabpfn"]:
        for n in [2, 4, 6, 8, 10]:
            key = f"{predictor}_n{n}"
            label = f"{predictor.upper()}_n{n}"
            methods[label] = key

    for dataset_name, data in tqdm(results.items(), desc=f"Processing {base_model_name}"):
        shap = data["shap"]

        for method_label, method_key in methods.items():
            if method_key not in data:
                continue
            pred = data[method_key]

            pearson_ci = bootstrap_correlation_ci(
                shap, pred, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE, metric="pearson", ci=CI_LEVEL
            )
            spearman_ci = bootstrap_correlation_ci(
                shap, pred, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE, metric="spearman", ci=CI_LEVEL
            )

            records.append({
                "dataset": dataset_name,
                "base_model": base_model_name,
                "method": method_label,
                "pearson_mean": pearson_ci["mean"],
                "pearson_lower": pearson_ci["lower"],
                "pearson_upper": pearson_ci["upper"],
                "pearson_std": pearson_ci["std"],
                "spearman_mean": spearman_ci["mean"],
                "spearman_lower": spearman_ci["lower"],
                "spearman_upper": spearman_ci["upper"],
                "spearman_std": spearman_ci["std"],
            })

    return pd.DataFrame(records)


def main():
    ensure_results_dirs()
    print("Loading RF results...")
    with open(DEFAULT_RESULTS["random_forest"], "rb") as f:
        rf_results = pickle.load(f)

    print("Loading MLP results...")
    with open(DEFAULT_RESULTS["mlp"], "rb") as f:
        mlp_results = pickle.load(f)

    print("Computing CIs for RF...")
    df_rf = compute_all_cis(rf_results, "Random Forest")

    print("Computing CIs for MLP...")
    df_mlp = compute_all_cis(mlp_results, "MLP")

    df = pd.concat([df_rf, df_mlp], ignore_index=True)

    output_csv = CSV_DIR / "bootstrap_cis_all_methods.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    # Also save as pickle
    output_pkl = PICKLES_DIR / "bootstrap_cis_all_methods.pkl"
    df.to_pickle(output_pkl)
    print(f"Saved to {output_pkl}")


if __name__ == "__main__":
    main()
