"""Generate the paper's LaTeX tables from the bootstrap CI results.

Produces compact Pearson/Spearman tables for the main text and the same
tables with 95% CIs for the appendix, for both base predictors.

Usage::

    python scripts/results/make_tables.py \
        [--input notebooks/paper_results/pickles/bootstrap_cis_all_methods.pkl] \
        [--paper-tables-dir ../arxiv-v2/tables]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    DEFAULT_RESULTS,
    PICKLES_DIR,
    TABLES_DIR,
    ensure_results_dirs,
    top_k_jaccard_similarity,
)

# Dataset ordering and abbreviations from existing tables
DATASET_ORDER = [
    "BANKNOTE AUTHENTICATION",
    "ABALONE",
    "ANNEALING",
    "CONTRACEPTIVE",
    "ECHOCARDIOGRAM",
    "FLAGS",
    "GERMAN CREDIT",
    "HEART",
    "HEART DISEASE",
    "HEPATITIS",
    "THYROID",
]

DATASET_ABBREV = {
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

METHOD_ORDER = [
    "EPFN",
    "MLP_n2",
    "MLP_n4",
    "MLP_n6",
    "MLP_n8",
    "MLP_n10",
    "RFR_n2",
    "RFR_n4",
    "RFR_n6",
    "RFR_n8",
    "RFR_n10",
    "TABPFN_n2",
    "TABPFN_n4",
    "TABPFN_n6",
    "TABPFN_n8",
    "TABPFN_n10",
]

METHOD_LABELS = {
    "EPFN": "EPFN",
    "MLP_n2": "MLP",
    "MLP_n4": "MLP",
    "MLP_n6": "MLP",
    "MLP_n8": "MLP",
    "MLP_n10": "MLP",
    "RFR_n2": "RF",
    "RFR_n4": "RF",
    "RFR_n6": "RF",
    "RFR_n8": "RF",
    "RFR_n10": "RF",
    "TABPFN_n2": "TabPFN",
    "TABPFN_n4": "TabPFN",
    "TABPFN_n6": "TabPFN",
    "TABPFN_n8": "TabPFN",
    "TABPFN_n10": "TabPFN",
}

N_SAMPLES = {
    "EPFN": 0,
    "MLP_n2": 2,
    "MLP_n4": 4,
    "MLP_n6": 6,
    "MLP_n8": 8,
    "MLP_n10": 10,
    "RFR_n2": 2,
    "RFR_n4": 4,
    "RFR_n6": 6,
    "RFR_n8": 8,
    "RFR_n10": 10,
    "TABPFN_n2": 2,
    "TABPFN_n4": 4,
    "TABPFN_n6": 6,
    "TABPFN_n8": 8,
    "TABPFN_n10": 10,
}


# Table method label -> key in the result pickles.
METHOD_KEYS = {"EPFN": "explainerpfn_statistical+additive"}
for _predictor in ("mlp", "rfr", "tabpfn"):
    for _n in (2, 4, 6, 8, 10):
        METHOD_KEYS[f"{_predictor.upper()}_n{_n}"] = f"{_predictor}_n{_n}"


def build_jaccard_df(results, base_model, jaccard_perc=1 / 3):
    """Per-dataset top-k Jaccard for every method in a results pickle."""
    rows = []
    for dataset, data in results.items():
        shap = np.asarray(data["shap"], float)
        k = max(1, int(shap.shape[1] * jaccard_perc))
        for method in METHOD_ORDER:
            predictions = data.get(METHOD_KEYS[method])
            if predictions is None:
                continue
            value = float(
                np.mean(
                    top_k_jaccard_similarity(
                        np.abs(np.asarray(predictions, float)), np.abs(shap), k=k
                    )
                )
            )
            rows.append(
                {
                    "dataset": dataset,
                    "base_model": base_model,
                    "method": method,
                    "jaccard_mean": value,
                }
            )
    return pd.DataFrame(rows)


def format_cell_compact(row, metric="pearson"):
    """Format a table cell with just the mean value."""
    mean = row[f"{metric}_mean"]
    return (float(mean), f"{mean:.3f}")


def format_cell_with_ci(row, metric="pearson"):
    """Format a table cell with mean and CI stacked vertically via makecell."""
    mean = row[f"{metric}_mean"]
    lower = row[f"{metric}_lower"]
    upper = row[f"{metric}_upper"]
    ci_str = f"({lower:.3f}--{upper:.3f})"
    return (float(mean), ci_str, f"{mean:.3f}")


def generate_table(df, base_model, metric="pearson", include_ci=False):
    """Generate a LaTeX table for a given base model and metric."""
    df = df[df["base_model"] == base_model].copy()

    # Pivot to wide format — store (raw_mean, display_parts) tuples
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        raw_means = {"Method": METHOD_LABELS[method], "Samples": N_SAMPLES[method]}
        display_parts = {"Method": METHOD_LABELS[method], "Samples": str(N_SAMPLES[method])}
        for dataset in DATASET_ORDER:
            abv = DATASET_ABBREV[dataset]
            ds_sub = sub[sub["dataset"] == dataset]
            if ds_sub.empty:
                raw_means[abv] = -999.0
                display_parts[abv] = ""
            else:
                row_data = ds_sub.iloc[0]
                if include_ci:
                    mean_val, ci_str, mean_fmt = format_cell_with_ci(row_data, metric=metric)
                    display_parts[abv] = (mean_fmt, ci_str)
                else:
                    mean_val, mean_fmt = format_cell_compact(row_data, metric=metric)
                    display_parts[abv] = (mean_fmt,)
                raw_means[abv] = mean_val
        rows.append((raw_means, display_parts))

    # Determine best / second-best per dataset column
    best = {}
    second = {}
    for col in DATASET_ABBREV.values():
        vals = [r[0][col] for r in rows]
        best_i = vals.index(max(vals))
        best[col] = best_i
        sorted_pairs = sorted(enumerate(vals), key=lambda x: x[1], reverse=True)
        if len(sorted_pairs) > 1 and sorted_pairs[1][1] > -900:
            second[col] = sorted_pairs[1][0]
        else:
            second[col] = None

    # Build the DataFrame with final formatted cells
    table_rows = []
    for i, (_, dp) in enumerate(rows):
        row_data = {"Method": dp["Method"], "Samples": dp["Samples"]}
        for col in DATASET_ABBREV.values():
            parts = dp[col]
            if parts == "" or parts[0] == "":
                row_data[col] = ""
                continue
            mean_fmt = parts[0]
            if i == best[col]:
                mean_fmt = f"\\textbf{{{mean_fmt}}}"
            elif second[col] is not None and i == second[col]:
                mean_fmt = f"\\underline{{{mean_fmt}}}"
            if len(parts) == 2:
                # CI included: stack vertically with makecell
                row_data[col] = f"\\makecell{{{mean_fmt} \\\\ \\tiny {parts[1]}}}"
            else:
                row_data[col] = mean_fmt
        table_rows.append(row_data)

    return pd.DataFrame(table_rows)


def df_to_latex(df_wide, stacked=False):
    """Convert a wide dataframe to a LaTeX tabular.

    If stacked=True, data cells use \\makecell for two-line content.
    Method labels are placed on the middle row of each group instead
    of using \\multirow (which misaligns with \\makecell cell heights).
    """
    lines = []
    lines.append("% Generated by scripts/results/make_tables.py")
    lines.append("\\begin{tabular}{" + "c" * len(df_wide.columns) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(df_wide.columns) + " \\\\")
    lines.append("\\midrule")

    if stacked:
        # Build groups: collect rows per method, then emit with label on middle row
        groups = []
        current_group = []
        current_method = None
        for _, row in df_wide.iterrows():
            method = row["Method"]
            if method != current_method:
                if current_group:
                    groups.append((current_method, current_group))
                current_group = [row]
                current_method = method
            else:
                current_group.append(row)
        if current_group:
            groups.append((current_method, current_group))

        first_group = True
        for method, group_rows in groups:
            if not first_group:
                lines.append("\\midrule")
            first_group = False
            n = len(group_rows)
            label_row = n // 2  # middle row index (0-based)
            for i, row in enumerate(group_rows):
                samples_cell = f"\\makecell{{{row['Samples']}}}"
                data = " & ".join(str(row[col]) for col in df_wide.columns[2:])
                if i == label_row:
                    lines.append(f"{method} & {samples_cell} & {data} \\\\")
                else:
                    lines.append(f" & {samples_cell} & {data} \\\\")
    else:
        # Count rows per method group for multirow
        method_rows = {}
        for _, row in df_wide.iterrows():
            m = row["Method"]
            method_rows[m] = method_rows.get(m, 0) + 1

        current_method = None
        for _, row in df_wide.iterrows():
            method = row["Method"]
            samples = row["Samples"]

            if method != current_method:
                if current_method is not None:
                    lines.append("\\midrule")
                nrows = method_rows[method]
                lines.append(f"\\multirow[c]{{{nrows}}}{{*}}{{{method}}} & {samples} & " + " & ".join(
                    str(row[col]) for col in df_wide.columns[2:]
                ) + " \\\\")
                current_method = method
            else:
                lines.append(f" & {samples} & " + " & ".join(
                    str(row[col]) for col in df_wide.columns[2:]
                ) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PICKLES_DIR / "bootstrap_cis_all_methods.pkl"),
        help="Bootstrap CI pickle produced by scripts/results/bootstrap_cis.py.",
    )
    parser.add_argument(
        "--paper-tables-dir",
        default=None,
        help="Optional copy target (e.g. the LaTeX source's tables/ folder).",
    )
    args = parser.parse_args()

    ensure_results_dirs()
    print("Loading bootstrap CI results...")
    df = pd.read_pickle(args.input)
    paper_tables_dir = Path(args.paper_tables_dir) if args.paper_tables_dir else TABLES_DIR
    paper_tables_dir.mkdir(parents=True, exist_ok=True)

    # MAIN BODY: Compact Pearson tables (no CIs in cells, same format as original)
    for base_model in ["Random Forest", "MLP"]:
        model_label = "random_forest" if base_model == "Random Forest" else "mlp"
        print(f"Generating compact Pearson table for {base_model}...")

        df_wide = generate_table(df, base_model, metric="pearson", include_ci=False)
        latex = df_to_latex(df_wide)

        output_path = paper_tables_dir / f"results_corr_across_datasets_{model_label}.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  Saved to {output_path}")

    # MAIN BODY: Compact Spearman tables
    for base_model in ["Random Forest", "MLP"]:
        model_label = "random_forest" if base_model == "Random Forest" else "mlp"
        print(f"Generating compact Spearman table for {base_model}...")

        df_wide = generate_table(df, base_model, metric="spearman", include_ci=False)
        latex = df_to_latex(df_wide)

        output_path = paper_tables_dir / f"results_spearman_across_datasets_{model_label}.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  Saved to {output_path}")

    # MAIN BODY: Compact Jaccard tables (computed from the result pickles)
    for base_model, results_key in [("Random Forest", "random_forest"), ("MLP", "mlp")]:
        model_label = "random_forest" if base_model == "Random Forest" else "mlp"
        print(f"Generating compact Jaccard table for {base_model}...")

        with open(DEFAULT_RESULTS[results_key], "rb") as handle:
            results = pickle.load(handle)
        df_jac = build_jaccard_df(results, base_model)
        latex = df_to_latex(generate_table(df_jac, base_model, metric="jaccard"))

        output_path = paper_tables_dir / f"results_jaccard_across_datasets_{model_label}.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  Saved to {output_path}")

    # APPENDIX: Full tables with CIs (both Pearson and Spearman)
    for base_model in ["Random Forest", "MLP"]:
        model_label = "random_forest" if base_model == "Random Forest" else "mlp"
        print(f"Generating appendix Pearson table with CIs for {base_model}...")

        df_wide = generate_table(df, base_model, metric="pearson", include_ci=True)
        latex = df_to_latex(df_wide, stacked=True)

        output_path = paper_tables_dir / f"results_corr_across_datasets_{model_label}_with_ci.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  Saved to {output_path}")

        print(f"Generating appendix Spearman table with CIs for {base_model}...")
        df_wide = generate_table(df, base_model, metric="spearman", include_ci=True)
        latex = df_to_latex(df_wide, stacked=True)

        output_path = paper_tables_dir / f"results_spearman_across_datasets_{model_label}_with_ci.tex"
        with open(output_path, "w") as f:
            f.write(latex)
        print(f"  Saved to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
