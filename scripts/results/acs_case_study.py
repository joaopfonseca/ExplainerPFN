"""ACS Public Coverage case study waterfalls.

Fits the paper's ACS model, computes exact SHAP and ExplainerPFN attributions,
and plots one high-agreement and one low-agreement per-instance waterfall.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import (  # noqa: E402
    CACHES_DIR,
    FIGURES_DIR,
    add_repo_to_path,
    ensure_results_dirs,
    resolve_model_path,
)

add_repo_to_path()

from explainerpfn.base import ExplainerPFN  # noqa: E402

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

def load_acs_data():
    from folktables import ACSDataSource, ACSPublicCoverage
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["AL"], download=True)
    df, y, group = ACSPublicCoverage.df_to_pandas(acs_data)
    df = df[["AGEP", "SCHL", "PINCP", "SEX"]]
    return df, y.values.squeeze()

def train_model(X_train, y_train):
    """Fit the same tuned MLP used by the main ACS case study."""
    search = GridSearchCV(
        estimator=Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        param_grid={
            "mlp__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "mlp__activation": ["relu", "tanh"],
            "mlp__alpha": [0.0001, 0.001, 0.01],
        },
        cv=3,
        n_jobs=16,
        scoring="f1_macro",
    )
    search.fit(X_train, y_train)
    print(f"Best parameters: {search.best_params_}")
    return search.best_estimator_

def compute_shap(clf, X_train, X_test):
    # ``clf`` scales its input internally, so the explainer must see raw X.
    explainer = shap.Explainer(clf.predict_proba, X_train)
    return explainer(X_test).values[:, :, -1]

def compute_explainerpfn(X_train, X_test, y_train_scores, y_test_scores):
    xai = ExplainerPFN(
        model_path=resolve_model_path(),
        random_state=RANDOM_STATE,
        inference_precision=torch.float32,
    )
    xai.fit(pd.concat([X_train, X_test], axis=0).values, np.concatenate([y_train_scores, y_test_scores], axis=0))
    exp_values = xai.predict(X_test.values, y_test_scores)
    return xai.apply_correction(
        y_test_scores, exp_values, kind=["statistical", "additive"]
    )

def select_instances(shap_vals, epfn_vals, X_test, n_high=1, n_low=1):
    """Select high- and low-agreement instances.
    
    High agreement:  maximum positive Pearson r among instances with >50% sign agreement.
    Low agreement:   minimum negative Pearson r among instances with <=50% sign agreement.
    This ensures the selected instances are visually meaningful extremes rather than
    sign-inverted cases that still correlate strongly in absolute terms."""
    per_instance_corr = []
    sign_agreement_frac = []
    for i in range(len(X_test)):
        r, _ = pearsonr(shap_vals[i], epfn_vals[i])
        per_instance_corr.append(r)
        # fraction of features with matching signs (ignoring exact zeros)
        s_shap = np.sign(shap_vals[i])
        s_epfn = np.sign(epfn_vals[i])
        signs_match = (s_shap == s_epfn) | ((shap_vals[i] == 0) & (epfn_vals[i] == 0))
        sign_agreement_frac.append(signs_match.mean())
    per_instance_corr = np.array(per_instance_corr)
    sign_agreement_frac = np.array(sign_agreement_frac)

    # High agreement: max positive r with majority sign agreement
    high_candidates = np.where(sign_agreement_frac > 0.5)[0]
    if len(high_candidates) == 0:
        high_candidates = np.arange(len(per_instance_corr))
    high_idx = high_candidates[np.argsort(per_instance_corr[high_candidates])[-n_high:][::-1]]

    # Low agreement: min negative r with minority/mixed sign agreement
    low_candidates = np.where(sign_agreement_frac <= 0.5)[0]
    if len(low_candidates) == 0:
        low_candidates = np.arange(len(per_instance_corr))
    low_idx = low_candidates[np.argsort(per_instance_corr[low_candidates])[:n_low]]

    return high_idx, low_idx, per_instance_corr

def plot_waterfall(instance_idx, shap_vals, epfn_vals, features, feature_names, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sort by average absolute value across SHAP and EPFN for unified ordering
    avg_abs = (np.abs(shap_vals) + np.abs(epfn_vals)) / 2
    order = np.argsort(avg_abs)[::-1]

    for ax, vals, label in zip(axes, [shap_vals, epfn_vals], ["SHAP", "ExplainerPFN"]):
        sorted_vals = vals[order]
        sorted_names = [feature_names[i] for i in order]
        colors = ["#1f77b4" if v >= 0 else "#d62728" for v in sorted_vals]

        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(label)
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved waterfall chart to {output_path}")
    plt.close()

def main():
    ensure_results_dirs()
    cache_path = CACHES_DIR / "acs_case_study_cache.pkl"

    if cache_path.exists():
        print("Loading cached results...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        X_test = cache["X_test"]
        shap_values = cache["shap_values"]
        exp_corr = cache["exp_corr"]
    else:
        print("Loading ACS data...")
        df, y = load_acs_data()
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=2000, random_state=RANDOM_STATE
        )
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=2000, random_state=RANDOM_STATE
        )

        print("Training model...")
        clf = train_model(X_train, y_train)
        y_test_scores = clf.predict_proba(X_test)[:, -1]
        y_train_scores = clf.predict_proba(X_train)[:, -1]

        print("Computing SHAP...")
        shap_values = compute_shap(clf, X_train, X_test)

        print("Computing ExplainerPFN...")
        exp_corr = compute_explainerpfn(X_train, X_test, y_train_scores, y_test_scores)

        cache = {
            "X_test": X_test,
            "shap_values": shap_values,
            "exp_corr": exp_corr,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Cached results to {cache_path}")

    feature_names = list(X_test.columns)
    features = X_test.values

    print("Selecting instances...")
    high_idx, low_idx, per_corr = select_instances(shap_values, exp_corr, X_test, n_high=1, n_low=1)
    print(f"High agreement instance: idx={high_idx[0]}, r={per_corr[high_idx[0]]:.3f}")
    print(f"Low agreement instance:  idx={low_idx[0]}, r={per_corr[low_idx[0]]:.3f}")

    # Print feature values for selected instances
    print("\nHigh agreement instance features:")
    for fname, fval in zip(feature_names, features[high_idx[0]]):
        print(f"  {fname}: {fval}")
    print("\nLow agreement instance features:")
    for fname, fval in zip(feature_names, features[low_idx[0]]):
        print(f"  {fname}: {fval}")

    plot_waterfall(
        high_idx[0],
        shap_values[high_idx[0]],
        exp_corr[high_idx[0]],
        features[high_idx[0]],
        feature_names,
        FIGURES_DIR / "acs_waterfall_high_agreement.pdf",
    )
    plot_waterfall(
        low_idx[0],
        shap_values[low_idx[0]],
        exp_corr[low_idx[0]],
        features[low_idx[0]],
        feature_names,
        FIGURES_DIR / "acs_waterfall_low_agreement.pdf",
    )

    # Also save per-instance correlation distribution plot
    plt.figure(figsize=(8, 4))
    plt.hist(per_corr, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(per_corr[high_idx[0]], color="green", linestyle="--", linewidth=2, label=f"High agreement (r={per_corr[high_idx[0]]:.2f})")
    plt.axvline(per_corr[low_idx[0]], color="red", linestyle="--", linewidth=2, label=f"Low agreement (r={per_corr[low_idx[0]]:.2f})")
    plt.xlabel("Per-Instance Pearson Correlation (SHAP vs. ExplainerPFN)")
    plt.ylabel("Count")
    plt.title("Distribution of Instance-Level Agreement on ACS Public Coverage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "acs_per_instance_correlation_distribution.pdf", dpi=300, bbox_inches="tight")
    print(f"Saved distribution plot to {FIGURES_DIR / 'acs_per_instance_correlation_distribution.pdf'}")
    plt.close()

if __name__ == "__main__":
    main()
