"""Loading and processing of the expanded-benchmark datasets.

Everything needed to turn a pinned source (an OpenML id or an mlresearch legacy dataset)
into the benchmark's processed frame, plus the helpers used to fingerprint and cache it:
numeric features only (categorical, object and boolean columns dropped), ID and constant
columns dropped, at most 20% missing cells median-imputed, multiclass targets binarised
majority-vs-rest. This mirrors scripts/results/base_model_families.py::load_datasets.

The dataset curation (which datasets enter the pool) lives in
scripts/datasets/build_registry.py; the user-facing entry point is
:class:.
"""
from __future__ import annotations

import gzip
import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_HOME_ENV = "EXPLAINERPFN_DATA_HOME"


def default_data_home() -> Path:
    """Where processed datasets and OpenML/UCI caches live.

    Priority: ``$EXPLAINERPFN_DATA_HOME``, ``<repo>/data/expanded`` in a source checkout,
    then ``~/explainerpfn_data/expanded``.
    """
    env = os.environ.get(DATA_HOME_ENV)
    if env:
        return Path(env).expanduser()
    repo = PACKAGE_DIR.parents[1]
    if (repo / "scripts").is_dir() and (repo / "data").is_dir():
        return repo / "data" / "expanded"
    return Path.home() / "explainerpfn_data" / "expanded"


def resolve_data_home(data_home) -> Path:
    return Path(data_home).expanduser() if data_home is not None else default_data_home()


def cache_filename(dataset: str, openml_did) -> str:
    """Deterministic cache file name of one registry row."""
    did = "legacy" if openml_did in ("", None) or pd.isna(openml_did) else int(openml_did)
    return f"{normalize_name(dataset)}_{did}.csv.gz"
MIN_N, MAX_N = 200, 20000
MIN_D, MAX_D = 3, 14
MAX_MISSING = 0.20  # max fraction of missing cells in the retained numeric features
MAX_CLASSES = 20  # OpenML-loaded targets with more distinct values are regression
BUCKETS = [(3, 7), (7, 11), (11, 15)]

COLUMNS = [
    "dataset", "uci_id", "openml_did", "openml_version", "source", "n_uci",
    "d_uci", "n", "d", "bucket", "n_classes", "missing", "dropped_columns",
    "sha256", "status", "notes",
]

# Legacy dataset -> (mlresearch class, fetch key, UCI candidate it overlaps or None).
LEGACY = {
    "BANKNOTE AUTHENTICATION": ("BinaryDatasets", "banknote_authentication",
                                "Banknote Authentication"),
    "ABALONE": ("ContinuousCategoricalDatasets", "abalone", "Abalone"),
    "ANNEALING": ("ContinuousCategoricalDatasets", "annealing", None),
    "CONTRACEPTIVE": ("ContinuousCategoricalDatasets", "contraceptive",
                      "Contraceptive Method Choice"),
    "GERMAN CREDIT": ("ContinuousCategoricalDatasets", "german_credit",
                      "Statlog (German Credit Data)"),
    "HEART": ("ContinuousCategoricalDatasets", "heart", "Statlog (Heart)"),
    "HEART DISEASE": ("ContinuousCategoricalDatasets", "heart_disease", None),
    "THYROID": ("ContinuousCategoricalDatasets", "thyroid", "Thyroid Disease"),
}

# OpenML default targets that are not the UCI task: use this column instead.
TARGET_OVERRIDES = {46612: "death_event"}
ID_NAME = re.compile(r"^(id|index|idx|unnamed.*|row.*|seq.*|sequence.*|no|number|case)$", re.I)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------- #
def assign_bucket(d: int) -> str | None:
    """Dimensionality bucket label for ``d`` numeric features, ``None`` if out of range."""
    for lo, hi in BUCKETS:
        if lo <= d < hi:
            return f"[{lo},{hi})"
    return None


def binarize_target(y: pd.Series) -> np.ndarray:
    """Binary target: unchanged if 2 classes, majority-vs-rest otherwise.

    Binary labels are mapped by sorted value (min -> 0), as the evaluation
    protocol remaps ``y == y.min()`` to 0. Multiclass: mode -> 1, rest -> 0
    (ties resolved by ``Series.mode().iloc[0]``, as in the legacy loader).
    """
    classes = sorted(pd.unique(y))
    if len(classes) == 2:
        return (y == classes[1]).to_numpy().astype(int)
    return (y == y.mode().iloc[0]).to_numpy().astype(int)


def is_id_column(name: str, values: pd.Series) -> bool:
    """Explicit ID/index/sequence column: by name, or a unit-step integer sequence."""
    if ID_NAME.match(str(name)):
        return True
    if pd.api.types.is_integer_dtype(values) and len(values) > 2:
        v = values.to_numpy()
        return bool(np.all(np.diff(v) == 1))
    return False


def clean_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Keep numeric non-ID, non-constant columns. Returns (X, dropped_columns)."""
    dropped, keep = [], []
    for col in X.columns:
        s = X[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            dropped.append(f"{col}(cat)")
        elif is_id_column(col, s):
            dropped.append(f"{col}(id)")
        elif s.nunique(dropna=True) <= 1:
            dropped.append(f"{col}(const)")
        else:
            keep.append(col)
    return X[keep].astype(float), dropped


def missing_fraction(X: pd.DataFrame) -> float:
    """Fraction of missing cells in the feature matrix."""
    return float(X.isna().to_numpy().mean()) if X.size else 0.0


def impute_median(X: pd.DataFrame) -> pd.DataFrame:
    """Median-impute numeric columns (the protocol for datasets with <= MAX_MISSING NaNs)."""
    return X.fillna(X.median())


def canonical_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, float_format="%.17g").encode()


def sha256_of(df: pd.DataFrame) -> str:
    """Hash of the processed frame as deterministic CSV bytes."""
    return hashlib.sha256(canonical_csv(df)).hexdigest()


def write_cache(df: pd.DataFrame, path: Path) -> None:
    """Deterministic gzip CSV (fixed mtime) so rebuilds are byte-identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(
        filename="", fileobj=raw, mode="wb", mtime=0) as gz:
        gz.write(canonical_csv(df))


def read_cache(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class Entry:
    dataset: str
    source: str
    uci_id: str = ""
    openml_did: int | str = ""
    openml_version: int | str = ""
    n_uci: int | str = ""
    d_uci: int | str = ""
    n: int | str = ""
    d: int | str = ""
    bucket: str = ""
    n_classes: int | str = ""
    missing: int | str = ""
    dropped_columns: str = ""
    sha256: str = ""
    status: str = "skipped"
    notes: str = ""
    frame: pd.DataFrame | None = field(default=None, repr=False)

    def row(self) -> dict:
        return {c: getattr(self, c) for c in COLUMNS}


def finalize(entry: Entry, df: pd.DataFrame, n_classes: int, dropped: list[str],
             n_imputed: int = 0) -> Entry:
    """Apply the pool criteria to a processed frame and fill the registry fields."""
    d = df.shape[1] - 1
    entry.n, entry.d = len(df), d
    entry.n_classes = n_classes
    entry.missing = n_imputed  # cells median-imputed before the frame was frozen
    entry.dropped_columns = ";".join(dropped)
    entry.bucket = assign_bucket(d) or ""
    if df.isna().to_numpy().any():
        entry.notes = "missing values after preprocessing"
    elif not MIN_N <= len(df) <= MAX_N:
        entry.notes = f"n={len(df)} outside [{MIN_N},{MAX_N}]"
    elif not entry.bucket:
        entry.notes = f"d={d} outside [{MIN_D},{MAX_D}]"
    else:
        entry.status = "included"
        entry.sha256 = sha256_of(df)
        entry.frame = df
    return entry


def load_openml(did: int, version: int | None, data_home=None):
    """(frame, name): ``fetch_openml`` result with the default target, pinned by id."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=did, as_frame=True, parser="auto",
                         data_home=str(resolve_data_home(data_home) / "_openml_cache"))
    return bunch.frame, bunch.target.name if bunch.target is not None else None


def process_openml(did: int, version: int | None, entry: Entry, data_home=None) -> Entry:
    frame, target_name = load_openml(did, version, data_home)
    entry.openml_did, entry.openml_version = did, version or ""
    if did in TARGET_OVERRIDES:
        target_name = TARGET_OVERRIDES[did]
        entry.notes = f"target overridden to {target_name}"
    if target_name is None or target_name not in frame:
        entry.notes = "no default target attribute"
        return entry
    y_raw = frame[target_name]
    if y_raw.nunique() > MAX_CLASSES:  # many-valued target = regression, not classification
        entry.notes = f"regression-like target ({y_raw.nunique()} distinct values)"
        return entry
    entry.notes = entry.notes if did in TARGET_OVERRIDES else ""
    X, dropped = clean_features(frame.drop(columns=[target_name]))
    if y_raw.isna().any():
        entry.notes = "missing target values"
        return entry
    frac = missing_fraction(X)
    if frac > MAX_MISSING:
        entry.notes = f"{frac:.1%} missing exceeds {MAX_MISSING:.0%}"
        return entry
    n_imputed = int(X.isna().to_numpy().sum())
    if n_imputed:
        X = impute_median(X)
        entry.notes = (entry.notes + "; " if entry.notes else "") + (
            f"median-imputed {n_imputed} cells ({frac:.1%})")
    y = binarize_target(y_raw)
    df = X.assign(target=y)
    return finalize(entry, df, int(y_raw.nunique()), dropped, n_imputed)


def load_legacy_frame(cls_name: str, key: str) -> pd.DataFrame:
    """Raw mlresearch dataset (downloaded to ``~/mlresearch_data`` if missing)."""
    import mlresearch.datasets as mld

    content = getattr(mld, cls_name)(names=[key]).download().content_
    df = content[0][1].copy()
    df.columns = df.columns.astype(str)
    return df


def process_legacy(name: str, cls_name: str, key: str) -> Entry:
    """Legacy protocol: drop ``cat*`` columns, binarise multiclass majority-vs-rest."""
    df = load_legacy_frame(cls_name, key)
    dropped = [f"{c}(cat)" for c in df.columns if c.startswith("cat")]
    df = df.loc[:, ~df.columns.str.startswith("cat")].copy()
    n_classes = int(df["target"].nunique())
    if n_classes > 2:
        df["target"] = (df["target"] == df["target"].mode().iloc[0]).astype(int)
    entry = Entry(dataset=name, source="legacy-cache")
    return finalize(entry, df, n_classes, dropped)
