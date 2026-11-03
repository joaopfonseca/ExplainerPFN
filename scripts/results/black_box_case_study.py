"""Third-party (black-box) case study.

Applies the frozen ExplainerPFN to a breast-cancer classifier we did not
train, downloaded from HuggingFace
(DeepNeural/breast_cancer_classifier_model_v1). The model is a
Pipeline(ColumnTransformer + RFE + LogisticRegression) trained by a third
party on the UCI Wisconsin Breast Cancer dataset (30 features). RFE selects
6 features internally; the final LogisticRegression operates on those 6.

We extract the 6 RFE-selected features and the model's probability scores,
then feed only (X_selected, y_hat) to ExplainerPFN -- it never queries the
model.

Two framings:

* Pure black-box: feed only (X, y_hat) to ExplainerPFN; validate with
  model-free checks -- feature-prediction mutual information, domain-knowledge
  ranking (concave points and area are dominant malignancy predictors), and
  stability across background samples.
* Blinded protocol: we *do* have the model, so we also compute SHAP via
  ``shap.explainers.Exact`` on the LogisticRegression and compare it with the
  ExplainerPFN attributions (Pearson/Spearman/Jaccard).

Outputs go to ``notebooks/paper_results/``:
    pickles/black_box_case_study_results.pkl
    csv/black_box_case_study.csv
    tables/black_box_case_study_summary.tex
    figures/black_box_case_study_*.pdf
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.datasets import load_breast_cancer
from huggingface_hub import hf_hub_download
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from _common import (  # noqa: E402
    CSV_DIR,
    FIGURES_DIR,
    PICKLES_DIR,
    TABLES_DIR,
    add_repo_to_path,
    ensure_results_dirs,
    resolve_model_path,
)

add_repo_to_path()

from explainerpfn.base import ExplainerPFN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

HF_REPO = "DeepNeural/breast_cancer_classifier_model_v1"
HF_FILENAME = "breast_cancer_classifier_model_v1.joblib"

# The model expects features named: radius1, texture1, ..., fractal_dimension3
# (mean=1, se=2, worst=3 suffixes)
EXPECTED_FEATURES = [
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1",
    "fractal_dimension1", "radius2", "texture2", "perimeter2", "area2",
    "smoothness2", "compactness2", "concavity2", "concave_points2",
    "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3",
    "area3", "smoothness3", "compactness3", "concavity3", "concave_points3",
    "symmetry3", "fractal_dimension3",
]

# Domain knowledge: worst concave points (concave_points3) is the established
# dominant predictor of malignancy (irregular contour). Mean area (area1) and
# worst compactness (compactness3) are also top-tier predictors. SE features
# (suffix 2) are generally weaker.
DOMAIN_RANKING = [
    "concave_points3", "area1", "compactness3", "concavity3",
    "radius3", "perimeter3", "texture3", "concave_points1",
    "area3", "compactness2",
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.labelsize": 14, "font.size": 14, "legend.fontsize": 12,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "font.family": "serif",
})


def load_breast_cancer_data():
    """Load UCI Wisconsin Breast Cancer and rename to model's expected format."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=EXPECTED_FEATURES)
    y = data.target  # 0 = malignant, 1 = benign
    return df, y


def load_hf_model():
    """Download the third-party breast cancer model from HuggingFace."""
    model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)
    model = joblib.load(model_path)
    return model


def top_k_jaccard_similarity(a, b, k):
    n, m = a.shape
    jaccards = []
    for row in range(n):
        top_a = set(np.argpartition(a[row], -k)[-k:])
        top_b = set(np.argpartition(b[row], -k)[-k:])
        inter = len(top_a & top_b)
        union = len(top_a | top_b)
        jaccards.append(inter / union if union else 1.0)
    return jaccards


def per_instance_pearson(shap_arr, pred_arr):
    n = shap_arr.shape[0]
    corrs = []
    for i in range(n):
        s, p = shap_arr[i], pred_arr[i]
        if np.std(s) < 1e-12 or np.std(p) < 1e-12:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(s, p)[0, 1]))
    return np.array(corrs)


def per_instance_sign_agreement(shap_arr, pred_arr):
    return np.mean(np.sign(shap_arr) == np.sign(pred_arr), axis=1)


def main():
    ensure_results_dirs()
    model_path = resolve_model_path()
    print(f"Device: {DEVICE}")

    # ---- Load data + model ----
    print("Loading UCI Wisconsin Breast Cancer dataset...")
    df, y = load_breast_cancer_data()
    print(f"  shape: {df.shape}, target (0=malignant, 1=benign): {np.bincount(y)}")

    print("\nDownloading HuggingFace model...")
    model = load_hf_model()
    print(f"  model type: {type(model).__name__}")
    print(f"  steps: {[(n, type(s).__name__) for n, s in model.steps]}")

    # ---- Extract RFE-selected features ----
    rfe = model.named_steps["rfe"]
    lgr = model.named_steps["lgr"]
    selected_mask = rfe.get_support()
    selected_features = [EXPECTED_FEATURES[i] for i in range(30) if selected_mask[i]]
    print(f"\nRFE selected {len(selected_features)} features: {selected_features}")

    # Get probability scores from the full pipeline
    y_scores = model.predict_proba(df)[:, 1]

    # Get the preprocessed + RFE-selected features (the space the LogisticRegression sees)
    X_pre = model.named_steps["preprocess"].transform(df)
    X_sel = rfe.transform(X_pre)
    print(f"  Selected feature matrix: {X_sel.shape}")
    print(f"  y_scores range: [{y_scores.min():.4f}, {y_scores.max():.4f}] "
          f"std={y_scores.std():.4f}")

    # ---- Split ----
    X_train, X_test, y_train, y_test, y_train_scores, y_test_scores = \
        train_test_split(X_sel, y, y_scores, test_size=0.3, random_state=RANDOM_STATE)
    feature_names = selected_features
    print(f"\nSplit: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # ---- ExplainerPFN (frozen, zero-shot) ----
    print("\nRunning ExplainerPFN (zero-shot on (X, y_hat))...")
    xai = ExplainerPFN(model_path=model_path, random_state=RANDOM_STATE, device=DEVICE)
    xai.fit(np.concatenate([X_train, X_test], axis=0),
            np.concatenate([y_train_scores, y_test_scores], axis=0))
    epfn_raw = xai.predict(X_test, y_test_scores)
    epfn_full = xai.apply_correction(y_test_scores, epfn_raw,
                                     kind=["statistical", "additive"])
    print(f"  EPFN attributions shape: {epfn_full.shape}")

    # ---- SHAP ground truth (blinded protocol) ----
    print("\nComputing SHAP via Exact explainer on LogisticRegression (blinded)...")
    shap_explainer = shap.explainers.Exact(lgr.predict_proba, X_train)
    shap_values = shap_explainer(X_test).values[:, :, -1]
    print(f"  SHAP values shape: {shap_values.shape}")

    # ---- Blinded comparison metrics ----
    pearson = float(np.corrcoef(epfn_full.ravel(), shap_values.ravel())[0, 1])
    spearman = float(spearmanr(epfn_full.ravel(), shap_values.ravel())[0])
    k = max(1, int(len(feature_names) / 3))
    jaccard = float(np.mean(top_k_jaccard_similarity(
        np.abs(epfn_full), np.abs(shap_values), k)))
    print("\nBlinded comparison (EPFN full pipeline vs SHAP):")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  Jaccard (top-{k}): {jaccard:.4f}")

    # ---- Per-instance correlations ----
    inst_corr = per_instance_pearson(shap_values, epfn_full)
    inst_sign = per_instance_sign_agreement(shap_values, epfn_full)
    print(f"\nPer-instance: mean Pearson={inst_corr.mean():.3f}, "
          f"mean sign-agreement={inst_sign.mean():.3f}")

    # ---- Model-free validation (pure black-box) ----
    print("\n--- Pure black-box validation ---")

    # 1. Feature-prediction mutual information ranking
    mi = mutual_info_classif(X_test, y_test, random_state=RANDOM_STATE)
    mi_ranking = np.argsort(-mi)
    mi_ranked_names = [feature_names[i] for i in mi_ranking]
    print(f"  MI feature ranking: {mi_ranked_names}")

    epfn_importance = np.mean(np.abs(epfn_full), axis=0)
    epfn_ranking = np.argsort(-epfn_importance)
    epfn_ranked_names = [feature_names[i] for i in epfn_ranking]
    print(f"  EPFN feature ranking: {epfn_ranked_names}")

    mi_ranks = np.argsort(np.argsort(-mi))
    epfn_ranks = np.argsort(np.argsort(-epfn_importance))
    mi_epfn_spearman = float(spearmanr(mi_ranks, epfn_ranks)[0])
    print(f"  MI vs EPFN rank Spearman: {mi_epfn_spearman:.3f}")

    # 2. Domain-knowledge check (only for selected features)
    domain_rank_map = {name: i for i, name in enumerate(DOMAIN_RANKING)}
    epfn_domain_ranks = []
    for fn in epfn_ranked_names:
        epfn_domain_ranks.append(domain_rank_map.get(fn, len(DOMAIN_RANKING)))
    domain_spearman = float(spearmanr(
        epfn_domain_ranks, list(range(len(epfn_domain_ranks))))[0])
    print(f"  EPFN vs domain-knowledge rank Spearman: {domain_spearman:.3f}")
    print(f"  EPFN actual:     {epfn_ranked_names}")
    print(f"  Domain top-10:   {DOMAIN_RANKING}")

    # 3. Stability across background subsamples and seeds
    print("\n  Stability check (5 background subsamples × 3 seeds)...")
    stability_rankings = []
    rng = np.random.default_rng(RANDOM_STATE)
    n_train = X_train.shape[0]
    for _ in range(5):
        bg_idx = rng.choice(n_train, size=min(200, n_train), replace=False)
        X_bg = X_train[bg_idx]
        y_bg = y_train_scores[bg_idx]
        for seed in [42, 123, 777]:
            xai_s = ExplainerPFN(model_path=model_path, random_state=seed, device=DEVICE)
            xai_s.fit(np.concatenate([X_bg, X_test]),
                      np.concatenate([y_bg, y_test_scores]))
            ep_s = xai_s.predict(X_test, y_test_scores)
            ep_s = xai_s.apply_correction(y_test_scores, ep_s,
                                          kind=["statistical", "additive"])
            imp_s = np.mean(np.abs(ep_s), axis=0)
            stability_rankings.append(np.argsort(-imp_s))
    stab_corrs = []
    for i in range(len(stability_rankings)):
        for j in range(i + 1, len(stability_rankings)):
            r1 = np.argsort(np.argsort(stability_rankings[i]))
            r2 = np.argsort(np.argsort(stability_rankings[j]))
            stab_corrs.append(spearmanr(r1, r2)[0])
    stab_corrs = np.array(stab_corrs)
    print(f"  Stability: mean pairwise rank Spearman = {stab_corrs.mean():.3f} "
          f"(std={stab_corrs.std():.3f})")

    # ---- Save results ----
    results = {
        "X_test": X_test, "y_test": y_test, "y_test_scores": y_test_scores,
        "shap": shap_values, "epfn_full": epfn_full, "epfn_raw": epfn_raw,
        "feature_names": feature_names,
        "pearson": pearson, "spearman": spearman, "jaccard": jaccard,
        "inst_corr": inst_corr, "inst_sign": inst_sign,
        "mi": mi, "mi_ranking": mi_ranking, "epfn_ranking": epfn_ranking,
        "mi_epfn_spearman": mi_epfn_spearman,
        "domain_spearman": domain_spearman,
        "stability_corrs": stab_corrs,
        "model_type": "Pipeline(ColumnTransformer + RFE + LogisticRegression)",
        "hf_repo": HF_REPO,
    }
    with open(PICKLES_DIR / "black_box_case_study_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved {PICKLES_DIR / 'black_box_case_study_results.pkl'}")

    # CSV
    df_csv = pd.DataFrame(X_test, columns=feature_names)
    df_csv["y_score"] = y_test_scores
    for j, fn in enumerate(feature_names):
        df_csv[f"shap_{fn}"] = shap_values[:, j]
        df_csv[f"epfn_{fn}"] = epfn_full[:, j]
    df_csv["inst_pearson"] = inst_corr
    df_csv["inst_sign_agreement"] = inst_sign
    df_csv.to_csv(CSV_DIR / "black_box_case_study.csv", index=False)
    print(f"Saved {CSV_DIR / 'black_box_case_study.csv'}")

    # ---- LaTeX summary table ----
    caption = (
        "Black-box case study: ExplainerPFN applied to a third-party breast cancer "
        "model (DeepNeural/breast\\_cancer\\_classifier\\_model\\_v1, "
        "HuggingFace) on the UCI Wisconsin Breast Cancer dataset. The model is "
        "a \\texttt{Pipeline(ColumnTransformer + RFE + LogisticRegression)} "
        "trained by a third party on 30 features; RFE internally selects 6 "
        "features. ExplainerPFN uses only the 6 selected features and logged "
        "$\\hat y$; it never queries the model. \\textbf{Blinded comparison}: "
        "SHAP via \\texttt{shap.explainers.Exact} on the LogisticRegression "
        "(computed post-hoc, not used by EPFN). \\textbf{Model-free validation}: "
        "mutual information ranking vs EPFN ranking (Spearman), and EPFN ranking "
        "vs domain-knowledge ranking (Spearman). \\textbf{Stability}: mean "
        "pairwise rank Spearman across 15 background/seed combinations."
    )
    summary_tex = (
        "\\begin{table}[ht]\n\\centering\n"
        f"\\caption{{{caption}}}\n"
        "\\label{tab:black_box_case_study}\n"
        "\\begin{tabular}{lc}\n\\toprule\n"
        "Metric & Value \\\\\n\\midrule\n"
        f"Blinded Pearson (EPFN vs SHAP) & {pearson:.3f} \\\\\n"
        f"Blinded Spearman (EPFN vs SHAP) & {spearman:.3f} \\\\\n"
        f"Blinded Jaccard (top-{k}) & {jaccard:.3f} \\\\\n"
        f"Per-instance Pearson (mean) & {inst_corr.mean():.3f} \\\\\n"
        f"Per-instance sign agreement (mean) & {inst_sign.mean():.3f} \\\\\n"
        f"MI vs EPFN ranking (Spearman) & {mi_epfn_spearman:.3f} \\\\\n"
        f"EPFN vs domain knowledge (Spearman) & {domain_spearman:.3f} \\\\\n"
        f"Stability (mean pairwise rank Spearman) & {stab_corrs.mean():.3f} \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    with open(TABLES_DIR / "black_box_case_study_summary.tex", "w") as f:
        f.write(summary_tex)
    print(f"Saved {TABLES_DIR / 'black_box_case_study_summary.tex'}")

    # ---- Figures ----
    # Fig 1: scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
    for j, fn in enumerate(feature_names):
        ax.scatter(shap_values[:, j], epfn_full[:, j], s=10, alpha=0.4,
                   color=colors[j], label=fn)
    lim = max(np.abs(shap_values).max(), np.abs(epfn_full).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("SHAP value"); ax.set_ylabel("ExplainerPFN attribution")
    ax.set_title("EPFN vs SHAP (breast cancer, third-party LogReg)")
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "black_box_case_study_shap_vs_epfn_scatter.pdf")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'black_box_case_study_shap_vs_epfn_scatter.pdf'}")

    # Fig 2: waterfalls
    high_idx = int(np.argmax(inst_corr))
    low_idx = int(np.argmin(inst_corr))
    for label, idx, fname in [
        ("high", high_idx, "black_box_case_study_waterfall_high_agreement.pdf"),
        ("low", low_idx, "black_box_case_study_waterfall_low_agreement.pdf"),
    ]:
        sv = shap_values[idx]
        ev = epfn_full[idx]
        order = np.argsort(-np.mean(np.abs(np.column_stack([sv, ev])), axis=1))
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax_i, (vals, title) in enumerate([(sv, "SHAP"), (ev, "ExplainerPFN")]):
            colors_h = ["#d62728" if v < 0 else "#1f77b4" for v in vals[order]]
            axes[ax_i].barh(range(len(order))[::-1], vals[order], color=colors_h)
            axes[ax_i].set_yticks(range(len(order))[::-1])
            axes[ax_i].set_yticklabels([feature_names[o] for o in order])
            axes[ax_i].set_title(f"{title} (inst {idx}, r={inst_corr[idx]:.2f})")
            axes[ax_i].axvline(0, color="k", lw=0.5)
        fig.suptitle(f"{label.capitalize()}-agreement instance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / fname)
        plt.close(fig)
        print(f"Saved {FIGURES_DIR / fname}")

    # Fig 3: feature importance
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = np.arange(len(feature_names))
    width = 0.25
    def norm(v):
        v = np.asarray(v, float)
        return v / v.max() if v.max() > 0 else v
    ax.bar(x_pos - width, norm(epfn_importance), width, label="EPFN (mean |attr|)")
    ax.bar(x_pos, norm(mi), width, label="Mutual Information")
    domain_imp = np.array([len(feature_names) - domain_rank_map.get(fn, len(DOMAIN_RANKING))
                           for fn in feature_names], float)
    ax.bar(x_pos + width, norm(domain_imp), width, label="Domain knowledge rank")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_ylabel("Normalized importance")
    ax.set_title("Feature importance: EPFN vs MI vs domain knowledge")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "black_box_case_study_feature_importance_domain_check.pdf")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'black_box_case_study_feature_importance_domain_check.pdf'}")

    # Fig 4: per-instance correlation
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(inst_corr, bins=25, edgecolor="black", alpha=0.7, color="#1f77b4")
    ax.axvline(inst_corr.mean(), color="red", ls="--", lw=1.5,
               label=f"mean={inst_corr.mean():.3f}")
    ax.set_xlabel("Per-instance Pearson (EPFN vs SHAP)")
    ax.set_ylabel("Count")
    ax.set_title("Per-instance correlation distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "black_box_case_study_per_instance_correlation.pdf")
    plt.close(fig)
    print(f"Saved {FIGURES_DIR / 'black_box_case_study_per_instance_correlation.pdf'}")

    # ---- Final summary ----
    print(f"\n{'=' * 60}")
    print("BLACK-BOX CASE STUDY SUMMARY")
    print(f"{'=' * 60}")
    print(f"Third-party model: {HF_REPO}")
    print("Model type: Pipeline(ColumnTransformer + RFE + LogisticRegression)")
    print("Dataset: UCI Wisconsin Breast Cancer (569 rows, 30 features → 6 selected)")
    print("\nBlinded comparison (EPFN vs SHAP):")
    print(f"  Pearson:  {pearson:.4f}")
    print(f"  Spearman: {spearman:.4f}")
    print(f"  Jaccard:  {jaccard:.4f}")
    print("\nModel-free validation:")
    print(f"  MI vs EPFN ranking:   {mi_epfn_spearman:.4f}")
    print(f"  EPFN vs domain know:  {domain_spearman:.4f}")
    print(f"  Stability (mean):     {stab_corrs.mean():.4f}")
    print("\nFeature rankings:")
    print(f"  EPFN:      {epfn_ranked_names}")
    print(f"  MI:        {mi_ranked_names}")


if __name__ == "__main__":
    main()