"""Progress of the ground-truth stage: finished pairs per family, errors, slowest pairs.

Counts the artifacts under ``pickles/ground_truth/<family>/`` against the frozen registry
and summarises ``logs/benchmarks/ground_truth_runs.csv``. Run it on the cluster (or after
syncing the results back) while the SLURM array is running:

    python scripts/results/ground_truth_status.py [--pending]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_repo_to_path  # noqa: E402

add_repo_to_path()

from compute_ground_truth import FAMILIES, GROUND_TRUTH_DIR, RUN_LOG, artifact_path  # noqa: E402
from explainerpfn.datasets import REGISTRY  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=str(GROUND_TRUTH_DIR))
    ap.add_argument("--log", default=str(RUN_LOG))
    ap.add_argument("--pending", action="store_true", help="list the unfinished pairs")
    args = ap.parse_args()

    datasets = [r["dataset"] for r in REGISTRY]
    total = done = 0
    pending = []
    print(f"{'family':15s} {'done':>9s}")
    for family in FAMILIES:
        missing = [d for d in datasets if not artifact_path(args.out_dir, family, d).exists()]
        pending += [(d, family) for d in missing]
        n_done = len(datasets) - len(missing)
        total, done = total + len(datasets), done + n_done
        print(f"{family:15s} {n_done:4d}/{len(datasets)}")
    print(f"{'all':15s} {done:4d}/{total}  ({100 * done / total:.0f}%)")

    log = Path(args.log)
    if log.exists():
        runs = pd.read_csv(log)
        errors = runs[runs.status != "ok"]
        print(f"\nlog rows: {len(runs)}, errors: {len(errors)}")
        if len(errors):
            print(errors[["dataset", "family", "status"]].to_string(index=False))
        ok = runs[runs.status == "ok"].copy()
        ok["total_time"] = ok.fit_time + ok.shap_time
        print("\nslowest finished pairs:")
        cols = ["dataset", "family", "d", "n_test", "shap_method", "fit_time", "shap_time"]
        print(ok.nlargest(5, "total_time")[cols].round(1).to_string(index=False))
    if args.pending:
        print("\npending:")
        for d, family in pending:
            print(f"  {d} / {family}")


if __name__ == "__main__":
    main()
