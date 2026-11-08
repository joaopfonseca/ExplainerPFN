"""Generate the dataset reports from the frozen registry.

Writes next to this script:
    datasets_summary.tex   LaTeX table (name, n, d, classes) grouped by bucket
    mapping_report.md      bucket counts, source breakdown, every skipped candidate + reason

    python scripts/datasets/report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_repo_to_path  # noqa: E402

add_repo_to_path()

HERE = Path(__file__).resolve().parent
BUCKET_ORDER = ["[3,7)", "[7,11)", "[11,15)"]
SOURCE_LABEL = {
    "legacy-cache": "legacy cache",
    "uci": "UCI via OpenML",
    "openml-suite": "OpenML curated suite",
    "openml-open": "OpenML open pool",
}


def tex_escape(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"), ("%", r"\%"),
                 ("#", r"\#"), ("$", r"\$")]:
        text = text.replace(a, b)
    return text


def display_name(name: str) -> str:
    return name.title() if name.isupper() else name


def summary_tex(inc: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & $n$ & $d$ & Classes \\",
    ]
    for bucket in BUCKET_ORDER:
        sub = inc[inc.bucket == bucket].sort_values("dataset", key=lambda s: s.str.lower())
        lines += [r"\midrule", rf"\multicolumn{{4}}{{l}}{{\emph{{$d \in {bucket}$ ({len(sub)} datasets)}}}} \\"]
        for r in sub.itertuples():
            lines.append(
                f"{tex_escape(display_name(r.dataset))} & {int(r.n):,} & {int(r.d)} & "
                f"{int(r.n_classes)} \\\\".replace(",", "{,}")
            )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def mapping_report(reg: pd.DataFrame) -> str:
    inc = reg[reg.status == "included"]
    skip = reg[reg.status == "skipped"]
    out = ["# Phase 0 dataset registry report\n",
           f"Included: **{len(inc)}**  |  skipped/recorded: {len(skip)}\n",
           "## Bucket counts (target 16 / 17 / 17)\n",
           "| bucket | included | target |", "|---|---|---|"]
    targets = dict(zip(BUCKET_ORDER, (16, 17, 17)))
    for b in BUCKET_ORDER:
        out.append(f"| {b} | {(inc.bucket == b).sum()} | {targets[b]} |")
    out += ["", "## Included by source\n", "| source | count |", "|---|---|"]
    for src, n in inc.source.value_counts().items():
        out.append(f"| {SOURCE_LABEL.get(src, src)} | {n} |")
    imputed = inc[inc.missing.fillna(0) > 0]
    out += ["", f"## Median-imputed datasets ({len(imputed)})\n"]
    out += [f"- {r.dataset}: {r.notes}" for r in imputed.itertuples()]
    uci_skips = skip[skip.source == "uci"]
    out += ["", f"## UCI candidates not in the pool ({len(uci_skips)})\n",
            "| candidate | UCI id | reason |", "|---|---|---|"]
    for r in uci_skips.sort_values("dataset").itertuples():
        uid = "" if pd.isna(r.uci_id) else int(r.uci_id)
        out.append(f"| {r.dataset} | {uid} | {r.notes} |")
    other = skip[skip.source != "uci"]
    out += ["", f"## Other skipped OpenML candidates ({len(other)})\n",
            "| dataset | OpenML id | reason |", "|---|---|---|"]
    for r in other.sort_values("dataset").itertuples():
        did = "" if pd.isna(r.openml_did) else int(r.openml_did)
        out.append(f"| {r.dataset} | {did} | {r.notes} |")
    out += ["", "## Legacy datasets outside the new pool (n < 200)\n",
            "Echocardiogram (n=61), Hepatitis (n=129), Flags (n=194) stay in the legacy "
            "per-dataset appendix tables only.", ""]
    return "\n".join(out)


def write_reports(reg: pd.DataFrame) -> None:
    """Write both reports from the full build table (included and skipped rows)."""
    inc = reg[reg.status == "included"]
    (HERE / "datasets_summary.tex").write_text(summary_tex(inc))
    (HERE / "mapping_report.md").write_text(mapping_report(reg))
    print(f"wrote {HERE / 'datasets_summary.tex'} and {HERE / 'mapping_report.md'}")
    print(inc.bucket.value_counts().reindex(BUCKET_ORDER).to_string())
