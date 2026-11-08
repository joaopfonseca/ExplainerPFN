"""Legacy validation gate for the frozen dataset registry.

For every legacy dataset in the registry: reload it through ``ExpandedDatasets``, rebuild the
70/30 split (``random_state=42``, labels remapped as in ``base_model_families.py``) and
require exact equality of ``X_train/X_test`` (and of the label partition) with the stored
results pickles (MLP and random forest). The stored SHAP arrays cannot be recomputed here, so
their shape is checked against ``X_test`` instead.

    python scripts/datasets/legacy_gate.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import PICKLES_DIR, add_repo_to_path  # noqa: E402

add_repo_to_path()

from explainerpfn.datasets import ExpandedDatasets  # noqa: E402

FAMILIES = ["mlp", "random_forest"]


def remap(y: np.ndarray) -> np.ndarray:
    return np.where(y == y.min(), 0, 1).astype(int)


def check(split: tuple, stored: dict) -> list[str]:
    """Exact X equality; labels compared after the {0,1} remap.

    Older pickles store raw UCI labels ({1, 2} for Heart and German Credit). The remap only
    renames them (larger label -> 1, the SHAP positive class), so the partition must match.
    """
    errors = []
    for key, got in zip(("X_train", "X_test", "y_train", "y_test"), split):
        want = stored[key]
        if key.startswith("y"):
            want = remap(want)
        if not np.array_equal(got, want):
            errors.append(f"{key} differs")
    if stored["shap"].shape != stored["X_test"].shape:
        errors.append("stored SHAP shape != X_test shape")
    return errors


def legacy_gate(data_home=None, pickles_dir: Path = PICKLES_DIR) -> list[tuple[str, list[str]]]:
    """Return ``[(dataset, errors)]`` for every legacy dataset (empty list = PASS)."""
    datasets = ExpandedDatasets(sources=["legacy-cache"], data_home=data_home).download()
    stored = {}
    for fam in FAMILIES:
        with open(Path(pickles_dir) / f"results_across_datasets_{fam}.pkl", "rb") as f:
            stored[fam] = pickle.load(f)
    registry = datasets.registry
    registry = registry.set_index(registry.dataset.map(datasets.display_name))
    results = []
    for name, frame in datasets:
        row = registry.loc[name]
        errors = []
        if (len(frame), frame.shape[1] - 1) != (row.n, row.d):
            errors.append("cache n/d != registry n/d")
        split = datasets.train_test_split(name)
        for fam in FAMILIES:
            if row.dataset not in stored[fam]:
                errors.append(f"{fam}: dataset missing from pickle")
                continue
            errors += [f"{fam}: {e}" for e in check(split, stored[fam][row.dataset])]
        results.append((name, errors))
    return results


def main() -> int:
    results = legacy_gate()
    for name, errors in results:
        print(f"{'PASS' if not errors else 'FAIL'}  {name:26s} {'; '.join(errors)}")
    failed = sum(bool(e) for _, e in results)
    print(f"\n{len(results) - failed}/{len(results)} legacy datasets reproduce the stored arrays")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
