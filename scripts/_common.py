"""Shared paths and helpers for the experiment scripts.

The scripts are grouped by what they do:

* ``train/``   -- training the foundation model and validating its data.
* ``eval/``    -- benchmarking a checkpoint or a baseline against SHAP values.
* ``results/`` -- turning evaluation outputs into the paper's tables/figures.

Every generated artifact goes to ``notebooks/paper_results/`` (git-ignored),
and the LaTeX source is expected next to the repository root (``../arxiv-v2``).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

RESULTS_DIR = NOTEBOOKS_DIR / "paper_results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
PICKLES_DIR = RESULTS_DIR / "pickles"
CSV_DIR = RESULTS_DIR / "csv"
CACHES_DIR = RESULTS_DIR / "caches"

LOGS_DIR = REPO_ROOT / "logs"
TABPFN_MODEL = NOTEBOOKS_DIR / "tabpfn-v2-regressor.ckpt"

PAPER_DIR = REPO_ROOT.parent / "arxiv-v2"
PAPER_TABLES_DIR = PAPER_DIR / "tables"

DEFAULT_RESULTS = {
    "mlp": PICKLES_DIR / "results_across_datasets_mlp.pkl",
    "random_forest": PICKLES_DIR / "results_across_datasets_random_forest.pkl",
}


def add_repo_to_path() -> None:
    """Make ``import explainerpfn`` work when a script is run by path."""
    for path in (REPO_ROOT, SCRIPTS_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def ensure_results_dirs() -> Path:
    """Create the artifact tree and return its root."""
    for path in (TABLES_DIR, FIGURES_DIR, PICKLES_DIR, CSV_DIR, CACHES_DIR):
        path.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def resolve_model_path(cli_value: str | os.PathLike | None = None) -> str:
    """Resolve an ExplainerPFN checkpoint.

    Priority: explicit CLI value, ``$EPFN_MODEL_PATH``, then the most recent
    ``checkpoints*/final_model.ckp`` found in the repository (falling back to
    ``best_model``). Both layouts are supported: ``checkpoints/...`` used
    locally and ``checkpoints_v5/...`` used on the cluster.
    """
    if cli_value:
        path = Path(cli_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {path}")
        return str(path)

    env_value = os.environ.get("EPFN_MODEL_PATH")
    if env_value:
        path = Path(env_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EPFN_MODEL_PATH points to a missing file: {path}"
            )
        return str(path)

    patterns = [
        "checkpoints/v*/final_model.ckp",
        "checkpoints_v*/final_model.ckp",
        "checkpoints*/final_model.ckp",
        "checkpoints/v*/best_model.ckp",
        "checkpoints_v*/best_model.ckp",
        "checkpoints*/best_model.ckp",
        "checkpoints/v*/final_model.pt",
        "checkpoints_v*/final_model.pt",
        "checkpoints*/final_model.pt",
    ]
    for pattern in patterns:
        matches = sorted(REPO_ROOT.glob(pattern), reverse=True)
        if matches:
            return str(matches[0])

    raise FileNotFoundError(
        "no ExplainerPFN checkpoint found; pass --model-path or set "
        "EPFN_MODEL_PATH. Searched: " + ", ".join(patterns)
    )


def safe_corr(a, b) -> float:
    """Pearson correlation, ``nan`` when either input is constant."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def top_k_jaccard_similarity(arr1, arr2, k: int = 3) -> np.ndarray:
    """Mean top-``k`` Jaccard similarity between the absolute values."""
    if arr1.ndim == 1:
        arr1 = arr1.reshape(1, -1)
    if arr2.ndim == 1:
        arr2 = arr2.reshape(1, -1)

    feats1 = np.argpartition(arr1, -k, axis=1)[:, -k:]
    feats2 = np.argpartition(arr2, -k, axis=1)[:, -k:]

    intersection = [
        np.intersect1d(row1, row2).shape[0]
        for row1, row2 in zip(feats1, feats2)
    ]
    union = [
        row1.shape[0] + row2.shape[0] - inter
        for row1, row2, inter in zip(feats1, feats2, intersection)
    ]
    return np.array(intersection) / np.array(union)


class ArtifactDir(Mapping):
    """Lazy ``{display name: artifact}`` view of a ground-truth directory.

    ``scripts/results/compute_ground_truth.py`` writes one pickle per dataset under
    ``pickles/ground_truth/<family>/``. Pickles are only loaded on access (a random-forest
    artifact can be large), and only datasets of the frozen registry that have an artifact
    appear as keys, so partially finished runs can already be evaluated.
    """

    def __init__(self, path: str | os.PathLike):
        add_repo_to_path()
        from explainerpfn.datasets import REGISTRY, ExpandedDatasets
        from explainerpfn.datasets.processing import normalize_name

        self.path = Path(path)
        self._files = {}
        for row in REGISTRY:
            file = self.path / f"{normalize_name(row['dataset'])}.pkl"
            if file.exists():
                self._files[ExpandedDatasets.display_name(row["dataset"])] = file

    def __getitem__(self, key):
        import pickle

        with open(self._files[key], "rb") as handle:
            return pickle.load(handle)

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)


def shard_items(items, shard: str | None):
    """Items of shard ``"i/N"`` (every N-th item); all items when ``shard`` is None."""
    items = list(items)
    if shard is None:
        return items
    index, count = (int(x) for x in shard.split("/"))
    if not 0 <= index < count:
        raise ValueError(f"shard {shard!r} must satisfy 0 <= i < N")
    return items[index::count]


def safe_spearman(a, b) -> float:
    """Pooled Spearman correlation, ``nan`` when either input is constant."""
    from scipy.stats import spearmanr

    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def write_pickle_atomic(path: str | os.PathLike, obj) -> None:
    """Pickle ``obj`` so that an interrupted job never leaves a truncated file."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(obj, handle)
    tmp.replace(path)


def load_results(name_or_path: str | os.PathLike = "mlp"):
    """Load experiment results: ``mlp``, ``random_forest``, a pickle, or an artifact directory.

    A directory (``pickles/ground_truth/<family>``) returns a lazy :class:`ArtifactDir`.
    """
    import pickle

    path = Path(name_or_path)
    if path.is_dir():
        return ArtifactDir(path)
    if not path.is_file():
        try:
            path = DEFAULT_RESULTS[str(name_or_path)]
        except KeyError as exc:
            raise FileNotFoundError(
                f"results pickle not found: {name_or_path}"
            ) from exc
    if not path.is_file():
        raise FileNotFoundError(
            f"results pickle not found: {path} "
            "(run the dataset-analysis notebook first)"
        )
    with open(path, "rb") as handle:
        return pickle.load(handle)
