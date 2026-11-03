"""Broadened evaluation-time base-model families.

Runs the frozen ExplainerPFN against SHAP ground truth computed from base-model
families other than the RF and MLP used in the main tables, to test that the
learned prior is not locked to a single supervision family:

    * XGBoost (gradient-boosted trees -- a different tree family than RF)
    * Deep regularized MLP ((124, 124) + alpha=1e-4 + early stopping)
    * SVM, logistic regression, naive Bayes (optional runs)

All use ``shap.explainers.Exact`` (the same SHAP method as the RF/MLP runs), so
the only varying axis is the base-model family.

The harness is a faithful refactor of
``notebooks/8-analysis-across-datasets.ipynb`` (cells 5-7): same 70/30 split,
``random_state=42``, ``shap.Exact`` on ``predict_proba[:, 1]``, the same
few-shot predictor dict and the same ``few_shot_explanations`` routine, so the
new result pickles are schema-identical to the existing RF/MLP ones.

Outputs go to ``notebooks/paper_results/``:
    pickles/results_across_datasets_<base_model>.pkl
    tables/results_{corr,spearman,jaccard}_across_datasets_<base_model>.tex
    tables/base_model_families_cross_family_table.tex
    csv/base_model_families_per_base_model.csv
    figures/base_model_families_epfn_vs_shap_<base_model>.pdf
"""
import sys
import time
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from tqdm.auto import tqdm
import torch
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    CSV_DIR,
    FIGURES_DIR,
    PICKLES_DIR,
    TABLES_DIR,
    TABPFN_MODEL,
    add_repo_to_path,
    ensure_results_dirs,
    resolve_model_path,
)

add_repo_to_path()

from mlresearch.datasets import BinaryDatasets, ContinuousCategoricalDatasets  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

from explainerpfn.base import ExplainerPFN  # noqa: E402
from make_tables import (
    generate_table, df_to_latex,
    DATASET_ORDER, DATASET_ABBREV, METHOD_ORDER, METHOD_LABELS, N_SAMPLES,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
MAX_SAMPLES = 20000
MAX_FEATURES = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# method_label -> key in the results pickle (matches bootstrap_cis.py).
METHODS = {"EPFN": "explainerpfn_statistical+additive", "EPFN (raw)": "explainerpfn"}
for predictor in ["mlp", "rfr", "tabpfn"]:
    for n in [2, 4, 6, 8, 10]:
        METHODS[f"{predictor.upper()}_n{n}"] = f"{predictor}_n{n}"

# base_model pkl-label -> display name used in the paper tables pipeline.
BASE_MODEL_DISPLAY = {
    "random_forest": "Random Forest",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "deep_mlp": "Deep MLP",
    "svm": "SVM",
    "logreg": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
}
# display name -> filename stem used by make_tables.py.
BASE_MODEL_STEM = {
    "Random Forest": "random_forest",
    "MLP": "mlp",
    "XGBoost": "xgboost",
    "Deep MLP": "deep_mlp",
    "SVM": "svm",
    "Logistic Regression": "logreg",
    "Naive Bayes": "naive_bayes",
}

FEW_SHOT_SAMPLE_SIZE = [2, 4, 6, 8, 10]

# Reused exactly as in notebook 8 (cells 7/18) so few-shot keys are comparable.
FEW_SHOT_PREDICTORS = {
    "tabpfn": TabPFNRegressor(
        model_path=str(TABPFN_MODEL),
        random_state=RANDOM_STATE,
        device=DEVICE,
    ),
    "mlp": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(12, 12), max_iter=5000, random_state=RANDOM_STATE)),
        ]
    ),
    "rfr": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("rfr", RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE)),
        ]
    ),
}

# Additional base-model families.
BASE_MODELS = {
    "xgboost": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("xgb", XGBClassifier(
                n_estimators=500, random_state=RANDOM_STATE,
                eval_metric="logloss", n_jobs=1)),
        ]
    ),
    "deep_mlp": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(124, 124), alpha=1e-4,
                early_stopping=True, max_iter=10000, random_state=RANDOM_STATE)),
        ]
    ),
    "svm": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    ),
    "logreg": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                max_iter=5000, random_state=RANDOM_STATE)),
        ]
    ),
    "naive_bayes": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("gnb", GaussianNB()),
        ]
    ),
}

# Metric config
N_BOOTSTRAP = 50
CI_LEVEL = 0.95


def load_datasets():
    datasets = BinaryDatasets().download()
    datasets.content_ += ContinuousCategoricalDatasets().download().content_
    content_processed = []
    for name, df in datasets:
        if name == "CREDIT APPROVAL":
            continue
        df.columns = df.columns.astype(str)
        df = df.loc[:, ~df.columns.str.startswith("cat")].copy()
        if df.target.nunique() > 2:
            df.target = (df.target == df.target.mode().iloc[0]).astype(int)
        if df.shape[1] < MAX_FEATURES and df.shape[1] > 2 and df.shape[0] < MAX_SAMPLES:
            content_processed.append((name, df))
    datasets.content_ = content_processed
    return datasets


def few_shot_explanations(X_train, y_train, base_model, shap_explanations,
                          X_test, y_test, model, n_samples=5, random_state=42):
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, train_size=n_samples, random_state=random_state
    )
    explanations_sample = shap_explanations(
        base_model.named_steps["scaler"].transform(X_train_sample)
    ).values[:, :, -1]

    Xy_sample = np.concatenate([X_train_sample, y_train_sample.reshape(-1, 1)], axis=1)
    Xy_test = np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1)

    tabpfn_preds = np.zeros(X_test.shape, dtype=float)
    for i in range(X_test.shape[1]):
        exp_target = explanations_sample[:, i]
        model.fit(Xy_sample, exp_target)
        pred = model.predict(Xy_test)
        tabpfn_preds[:, i] = pred
    return tabpfn_preds


def run_experiments_across_datasets(datasets, model_path, base_model,
                                    few_shot_sample_size, few_shot_predictors,
                                    random_state=42, device=DEVICE):
    results = {}
    for dataset_name, df in tqdm(datasets, desc="datasets"):
        results[dataset_name] = {}
        X = df.drop(columns=["target"]).values
        y = df["target"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=random_state)
        # Remap labels to {0, 1} (some UCI sets use {1, 2}); XGBoost requires it,
        # and it is a no-op for sklearn estimators that already accept {0, 1}.
        y_train = np.where(y_train == y_train.min(), 0, 1).astype(int)
        y_test = np.where(y_test == y_test.min(), 0, 1).astype(int)
        base_model_ = clone(base_model)
        base_model_.fit(X_train, y_train)
        y_train_scores = base_model_.predict_proba(X_train)[:, 1]
        y_test_scores = base_model_.predict_proba(X_test)[:, 1]
        results[dataset_name]["X_train"] = X_train
        results[dataset_name]["X_test"] = X_test
        results[dataset_name]["y_train"] = y_train
        results[dataset_name]["y_test"] = y_test
        results[dataset_name]["base_model"] = base_model_
        results[dataset_name]["y_train_scores"] = y_train_scores
        results[dataset_name]["y_test_scores"] = y_test_scores

        # SHAP ground truth (Exact — same as RF/MLP runs)
        t0 = time.time()
        shap_explanations = shap.explainers.Exact(
            base_model_.steps[-1][1].predict_proba,
            base_model_.named_steps["scaler"].transform(X_train))
        shap_values_test = shap_explanations(
            base_model_.named_steps["scaler"].transform(X_test)).values[:, :, -1]
        results[dataset_name]["shap"] = shap_values_test
        results[dataset_name]["shap_time"] = time.time() - t0

        # ExplainerPFN (frozen, model-agnostic — consumes only X, y_scores)
        t0 = time.time()
        xai = ExplainerPFN(
            model_path=model_path,
            random_state=random_state,
            device=device,
            inference_precision=torch.float32,
        )
        xai.fit(np.concatenate([X_train, X_test], axis=0),
                np.concatenate([y_train_scores, y_test_scores], axis=0))
        exp_pfn_values = xai.predict(X_test, y_test_scores)
        results[dataset_name]["explainerpfn_time"] = time.time() - t0
        results[dataset_name]["explainerpfn"] = exp_pfn_values / np.sqrt(X.shape[1])
        results[dataset_name]["explainerpfn_statistical+additive"] = xai.apply_correction(
            y_test_scores, exp_pfn_values, kind=["statistical", "additive"])
        results[dataset_name]["explainerpfn_additive"] = xai.apply_correction(
            y_test_scores, exp_pfn_values - exp_pfn_values.mean(axis=0), kind=["additive"])

        # Few-shot surrogates (15 keys, schema-identical to existing pkls)
        for n_samples, (model_name, model) in product(
                few_shot_sample_size, few_shot_predictors.items()):
            model = clone(model)
            t0 = time.time()
            exp_few_shot = few_shot_explanations(
                X_train, y_train, base_model_, shap_explanations,
                X_test, y_test, model, n_samples=n_samples, random_state=random_state)
            results[dataset_name][f"{model_name}_n{n_samples}_time"] = time.time() - t0
            results[dataset_name][f"{model_name}_n{n_samples}"] = exp_few_shot
    return results


# --------------------------------------------------------------------------- #
# Metrics (mirror notebook 8 cell 12 + bootstrap_cis.py)
# --------------------------------------------------------------------------- #
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


def dataset_metrics(shap_arr, pred_arr, jaccard_frac=1 / 3):
    k = max(1, int(shap_arr.shape[1] * jaccard_frac))
    pearson = np.corrcoef(pred_arr.ravel(), shap_arr.ravel())[0, 1]
    spearman = spearmanr(pred_arr.ravel(), shap_arr.ravel())[0]
    jaccard = np.mean(top_k_jaccard_similarity(np.abs(pred_arr), np.abs(shap_arr), k))
    return pearson, spearman, jaccard


def bootstrap_correlation_ci(shap_arr, pred_arr, n_bootstrap=50,
                            random_state=42, metric="pearson", ci=0.95):
    rng = np.random.default_rng(random_state)
    n = shap_arr.shape[0]
    corrs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
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


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run_base_model(label, base_model, datasets):
    print(f"\n{'=' * 70}\n{label} base-model run\n{'=' * 70}")
    results = run_experiments_across_datasets(
        datasets=datasets, model_path=resolve_model_path(), base_model=base_model,
        few_shot_sample_size=FEW_SHOT_SAMPLE_SIZE,
        few_shot_predictors=FEW_SHOT_PREDICTORS,
        random_state=RANDOM_STATE, device=DEVICE)
    out_pkl = PICKLES_DIR / f"results_across_datasets_{label}.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved {out_pkl}")
    return results


def build_metrics_df(all_results):
    """Per-dataset Pearson/Spearman/Jaccard for EPFN (full pipeline) vs SHAP,
    plus bootstrap CIs. Returns a long-format DataFrame."""
    rows = []
    for base_label, results in all_results.items():
        for ds_name, data in tqdm(results.items(), desc=f"metrics {base_label}"):
            shap_arr = data["shap"]
            pred = data["explainerpfn_statistical+additive"]
            pear, spear, jac = dataset_metrics(shap_arr, pred)
            pci = bootstrap_correlation_ci(shap_arr, pred, N_BOOTSTRAP, RANDOM_STATE, "pearson", CI_LEVEL)
            sci = bootstrap_correlation_ci(shap_arr, pred, N_BOOTSTRAP, RANDOM_STATE, "spearman", CI_LEVEL)
            rows.append({
                "base_model": base_label, "dataset": ds_name,
                "pearson": pear, "spearman": spear, "jaccard": jac,
                "pearson_ci_lower": pci["lower"], "pearson_ci_upper": pci["upper"],
                "spearman_ci_lower": sci["lower"], "spearman_ci_upper": sci["upper"],
            })
    return pd.DataFrame(rows)


def build_all_methods_cis(all_results):
    """Bootstrap CIs for ALL 16 methods (EPFN + 15 few-shot) per base model,
    matching the schema bootstrap_cis.py produces so that
    ``make_tables.generate_table()`` can consume it directly:
        base_model, dataset, method, pearson_mean/lower/upper/std,
        spearman_mean/lower/upper/std
    """
    rows = []
    for base_label, results in all_results.items():
        display = BASE_MODEL_DISPLAY.get(base_label, base_label)
        for ds_name, data in tqdm(results.items(), desc=f"CIs {base_label}"):
            shap_arr = data["shap"]
            for method_label, method_key in METHODS.items():
                if method_key not in data:
                    continue
                pred = data[method_key]
                pci = bootstrap_correlation_ci(shap_arr, pred, N_BOOTSTRAP,
                                                RANDOM_STATE, "pearson", CI_LEVEL)
                sci = bootstrap_correlation_ci(shap_arr, pred, N_BOOTSTRAP,
                                                RANDOM_STATE, "spearman", CI_LEVEL)
                rows.append({
                    "base_model": display, "dataset": ds_name,
                    "method": method_label,
                    "pearson_mean": pci["mean"],
                    "pearson_lower": pci["lower"], "pearson_upper": pci["upper"],
                    "pearson_std": pci["std"],
                    "spearman_mean": sci["mean"],
                    "spearman_lower": sci["lower"], "spearman_upper": sci["upper"],
                    "spearman_std": sci["std"],
                })
    return pd.DataFrame(rows)


def emit_paper_tables(df_cis, base_models):
    """Emit the 5 paper-style .tex files per base model, mirroring
    make_tables.py: compact Pearson, compact Spearman, appendix
    Pearson+CI, appendix Spearman+CI, and compact Jaccard (no CI in cells).

    Each table is written as an individual .tex file under the tables dir,
    using the same filename stem convention as the existing paper tables
    (results_corr_across_datasets_<model>.tex, etc.).

    Jaccard has no bootstrap CIs in the existing pipeline; it is computed
    pointwise per dataset/method and rendered compact (matching the notebook).
    """
    out_dir = TABLES_DIR

    for display in base_models:
        stem = BASE_MODEL_STEM[display]
        df_sub = df_cis[df_cis["base_model"] == display]
        if df_sub.empty:
            print(f"  skipping {display}: no rows")
            continue

        # Compact Pearson (main body)
        df_wide = generate_table(df_cis, display, metric="pearson", include_ci=False)
        latex = df_to_latex(df_wide)
        p = out_dir / f"results_corr_across_datasets_{stem}.tex"
        with open(p, "w") as f:
            f.write(latex)
        print(f"  wrote {p}")

        # Compact Spearman (main body)
        df_wide = generate_table(df_cis, display, metric="spearman", include_ci=False)
        latex = df_to_latex(df_wide)
        p = out_dir / f"results_spearman_across_datasets_{stem}.tex"
        with open(p, "w") as f:
            f.write(latex)
        print(f"  wrote {p}")

        # Appendix Pearson with CIs
        df_wide = generate_table(df_cis, display, metric="pearson", include_ci=True)
        latex = df_to_latex(df_wide, stacked=True)
        p = out_dir / f"results_corr_across_datasets_{stem}_with_ci.tex"
        with open(p, "w") as f:
            f.write(latex)
        print(f"  wrote {p}")

        # Appendix Spearman with CIs
        df_wide = generate_table(df_cis, display, metric="spearman", include_ci=True)
        latex = df_to_latex(df_wide, stacked=True)
        p = out_dir / f"results_spearman_across_datasets_{stem}_with_ci.tex"
        with open(p, "w") as f:
            f.write(latex)
        print(f"  wrote {p}")

        # Compact Jaccard (no CIs — computed pointwise from the pkls)
        results = None
        for bl, disp in BASE_MODEL_DISPLAY.items():
            if disp == display:
                # load the pkl for this base model
                pkl = PICKLES_DIR / f"results_across_datasets_{bl}.pkl"
                if pkl.exists():
                    with open(pkl, "rb") as f:
                        results = pickle.load(f)
                break
        if results is not None:
            jac_rows = []
            for method in METHOD_ORDER:
                method_key = METHODS[method]
                rec = {"Method": METHOD_LABELS[method], "Samples": str(N_SAMPLES[method])}
                for ds in DATASET_ORDER:
                    if ds not in results or method_key not in results[ds]:
                        rec[DATASET_ABBREV[ds]] = ""
                        continue
                    shap_arr = results[ds]["shap"]
                    pred = results[ds][method_key]
                    k = max(1, int(shap_arr.shape[1] / 3))
                    jac = float(np.mean(top_k_jaccard_similarity(
                        np.abs(pred), np.abs(shap_arr), k)))
                    rec[DATASET_ABBREV[ds]] = f"{jac:.3f}"
                jac_rows.append(rec)
            df_jac = pd.DataFrame(jac_rows)
            # bold best / underline second best per dataset column
            for col in DATASET_ABBREV.values():
                vals = []
                for _, r in df_jac.iterrows():
                    try:
                        vals.append(float(r[col]))
                    except (ValueError, TypeError):
                        vals.append(-999.0)
                best_i = vals.index(max(vals))
                sorted_pairs = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
                second_i = sorted_pairs[1][0] if len(sorted_pairs) > 1 and sorted_pairs[1][1] > -900 else None
                for i, (_, r) in enumerate(df_jac.iterrows()):
                    if i == best_i:
                        df_jac.at[r.name, col] = f"\\textbf{{{r[col]}}}"
                    elif second_i is not None and i == second_i:
                        df_jac.at[r.name, col] = f"\\underline{{{r[col]}}}"
            latex = df_to_latex(df_jac)
            p = out_dir / f"results_jaccard_across_datasets_{stem}.tex"
            with open(p, "w") as f:
                f.write(latex)
            print(f"  wrote {p}")


def load_existing_results():
    """Load existing RF + MLP pkls to put them into the same cross-family table."""
    out = {}
    for label, fn in [("random_forest", "results_across_datasets_random_forest.pkl"),
                      ("mlp", "results_across_datasets_mlp.pkl")]:
        p = PICKLES_DIR / fn
        if p.exists():
            with open(p, "rb") as f:
                out[label] = pickle.load(f)
    return out


def cross_family_table(df_metrics):
    """One row per dataset, one column block per base model (Pearson [95% CI])."""
    base_order = ["random_forest", "mlp", "xgboost", "deep_mlp", "svm",
                  "logreg", "naive_bayes"]
    base_labels = {"random_forest": "RF", "mlp": "MLP",
                   "xgboost": "XGBoost", "deep_mlp": "Deep-MLP",
                   "svm": "SVM", "logreg": "LogReg", "naive_bayes": "NB"}
    datasets = list(dict.fromkeys(df_metrics["dataset"]))
    rows = []
    for ds in datasets:
        row = {"Dataset": ds}
        for b in base_order:
            sub = df_metrics[(df_metrics["base_model"] == b) & (df_metrics["dataset"] == ds)]
            if len(sub) == 0:
                row[f"{base_labels[b]} Pearson"] = ""
                row[f"{base_labels[b]} 95% CI"] = ""
            else:
                r = sub.iloc[0]
                row[f"{base_labels[b]} Pearson"] = f"{r['pearson']:.3f}"
                row[f"{base_labels[b]} 95% CI"] = f"[{r['pearson_ci_lower']:.3f}, {r['pearson_ci_upper']:.3f}]"
        rows.append(row)
    # Add a mean row
    mean_row = {"Dataset": "Mean"}
    for b in base_order:
        sub = df_metrics[df_metrics["base_model"] == b]
        if len(sub):
            mean_row[f"{base_labels[b]} Pearson"] = f"{sub['pearson'].mean():.3f}"
            mean_row[f"{base_labels[b]} 95% CI"] = ""
    rows.append(mean_row)
    return pd.DataFrame(rows)


def scatter_epfn_vs_shap(results, label):
    """One scatter PDF per base model: EPFN (full pipeline) vs SHAP, pooled across datasets."""
    all_shap, all_epfn = [], []
    for ds, data in results.items():
        all_shap.append(data["shap"].ravel())
        all_epfn.append(data["explainerpfn_statistical+additive"].ravel())
    s = np.concatenate(all_shap)
    e = np.concatenate(all_epfn)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(s, e, s=1, alpha=0.2)
    lim = max(abs(s).max(), abs(e).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("SHAP value"); ax.set_ylabel("ExplainerPFN (full pipeline)")
    ax.set_title(f"EPFN vs SHAP -- {label} (pooled, n={s.size})")
    plt.tight_layout()
    out = FIGURES_DIR / f"base_model_families_epfn_vs_shap_{label}.pdf"
    plt.savefig(out); plt.close(fig)
    print(f"Saved {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-only", action="store_true",
                    help="Skip base-model training; load existing pkls and only "
                         "recompute metrics + emit tables.")
    ap.add_argument("--base-models", nargs="+", default=None,
                    help="Subset of base models to train (e.g. svm). Existing "
                         "pkls for the others are loaded. Default: all in BASE_MODELS.")
    args = ap.parse_args()

    ensure_results_dirs()
    print(f"Device: {DEVICE}")
    all_results = load_existing_results()  # RF + MLP for the cross-family table

    # Load existing family pkls first (so --base-models svm doesn't redo xgboost).
    for label in BASE_MODELS:
        pkl = PICKLES_DIR / f"results_across_datasets_{label}.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                all_results[label] = pickle.load(f)

    to_train = args.base_models or list(BASE_MODELS.keys())
    if args.metrics_only:
        print("--metrics-only: no training")
    else:
        datasets = load_datasets()
        print(f"Loaded {len(datasets.content_)} datasets")
        for label in to_train:
            if label not in BASE_MODELS:
                print(f"  unknown base model '{label}', skipping")
                continue
            res = run_base_model(label, BASE_MODELS[label], datasets)
            all_results[label] = res
            scatter_epfn_vs_shap(res, label)

    print("\nComputing metrics...")
    df_metrics = build_metrics_df(all_results)
    df_metrics.to_csv(CSV_DIR / "base_model_families_per_base_model.csv", index=False)
    print(f"Saved {CSV_DIR / 'base_model_families_per_base_model.csv'}")

    cf = cross_family_table(df_metrics)
    cf.to_latex(TABLES_DIR / "base_model_families_cross_family_table.tex", index=False)
    print(f"Saved {TABLES_DIR / 'base_model_families_cross_family_table.tex'}")
    print("\nCross-family table:")
    print(cf.to_string(index=False))

    # Per-base-model mean summary
    summary = df_metrics.groupby("base_model").agg(
        pearson_mean=("pearson", "mean"),
        spearman_mean=("spearman", "mean"),
        jaccard_mean=("jaccard", "mean"),
    ).round(3)
    print("\nPer-base-model means (EPFN full pipeline vs SHAP):")
    print(summary.to_string())

    # ---- Paper-style per-base-model tables (Pearson/Spearman compact + with-CI,
    #      Jaccard compact) for the new base models, matching the existing
    #      results_corr_across_datasets_<model>.tex convention. ----
    print("\nComputing bootstrap CIs for all 16 methods (EPFN + few-shot)...")
    df_cis = build_all_methods_cis(all_results)
    df_cis.to_csv(CSV_DIR / "base_model_families_bootstrap_cis.csv", index=False)
    df_cis.to_pickle(PICKLES_DIR / "base_model_families_bootstrap_cis.pkl")
    print(f"Saved {CSV_DIR / 'base_model_families_bootstrap_cis.csv'}")

    # Emit tables for the new base models only (RF/MLP already in the paper).
    new_display = [BASE_MODEL_DISPLAY[l] for l in BASE_MODELS]
    print(f"\nEmitting paper-style tables for {new_display}...")
    emit_paper_tables(df_cis, new_display)


if __name__ == "__main__":
    main()
