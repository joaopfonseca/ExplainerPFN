"""User-facing access to the frozen expanded-benchmark datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from mlresearch.datasets.base import Datasets
from sklearn.model_selection import train_test_split

from .processing import (
    LEGACY,
    BUCKETS,
    Entry,
    resolve_data_home,
    assign_bucket,
    cache_filename,
    normalize_name,
    process_legacy,
    process_openml,
    read_cache,
    sha256_of,
    write_cache,
)
from .registry import REGISTRY

BUCKET_LABELS = [assign_bucket(lo) for lo, _ in BUCKETS]


class ExpandedDatasets(Datasets):
    """The 46 pre-registered datasets of the expanded ExplainerPFN benchmark.

    Same interface as the ``mlresearch`` dataset classes (e.g. ``BinaryDatasets``):
    ``download()`` populates ``content_`` with ``(NAME, DataFrame)`` tuples, where each
    frame holds the numeric features plus a binary ``target`` column, and the object can be
    iterated, indexed by position or name, and queried with ``keys``/``values``/``items``.

    The pool is frozen in ``explainerpfn/datasets/registry.py`` (dataset, source,
    pinned OpenML id, dimensionality bucket, sha256 of the processed data). Processed data
    is cached under ``data_home``; a missing or modified cache file is rebuilt from the
    pinned source and must reproduce the registry sha256, otherwise an error is raised.

    Preprocessing (identical for all datasets): numeric features only (categorical, object
    and boolean columns dropped), ID and constant columns dropped, at most 20% missing cells
    median-imputed, multiclass targets binarised majority-vs-rest.

    Parameters
    ----------
    names : str or list, default="all"
        Dataset names (case-insensitive, e.g. ``"banknote authentication"``) or ``"all"``.

    data_home : str or Path, default=None
        Cache directory. Defaults to ``$EXPLAINERPFN_DATA_HOME``, ``<repo>/data/expanded``
        in a source checkout, or ``~/explainerpfn_data/expanded``.

    download_if_missing : bool, default=True
        Rebuild datasets that are not cached from their pinned source.

    buckets : list of str, default=None
        Keep only these dimensionality buckets: ``"[3,7)"``, ``"[7,11)"``, ``"[11,15)"``.

    sources : list of str, default=None
        Keep only these registry sources (``legacy-cache``, ``uci``, ``openml-suite``,
        ``openml-open``).

    verify : bool, default=True
        Check the sha256 of every loaded dataset against the registry.

    entries : sequence of dict, default=None
        Alternative registry rows (mainly for testing); defaults to ``REGISTRY``.

    Attributes
    ----------
    data_home_ : Path
        Directory used for the cache.

    content_ : list
        List of ``(NAME, DataFrame)`` tuples.

    Examples
    --------
    >>> from explainerpfn.datasets import ExpandedDatasets
    >>> datasets = ExpandedDatasets(buckets=["[11,15)"]).download()
    >>> for name, df in datasets:
    ...     X, y = df.drop(columns="target").values, df["target"].values
    """

    def __init__(
        self,
        names: Union[str, list] = "all",
        data_home: Optional[Union[str, Path]] = None,
        download_if_missing: bool = True,
        buckets: Optional[list] = None,
        sources: Optional[list] = None,
        verify: bool = True,
        entries: Optional[Sequence[dict]] = None,
    ):
        super().__init__(
            names=names,
            data_home=None if data_home is None else str(data_home),
            download_if_missing=download_if_missing,
        )
        self.buckets = buckets
        self.sources = sources
        self.verify = verify
        self.entries = entries

    # ------------------------------------------------------------------ registry
    @staticmethod
    def display_name(dataset: str) -> str:
        """Name used in ``content_`` (upper case, as in the mlresearch classes)."""
        return dataset.upper().replace("_", " ")

    @property
    def registry(self) -> pd.DataFrame:
        """Registry rows (included datasets only) selected by the constructor arguments."""
        reg = pd.DataFrame(REGISTRY if self.entries is None else list(self.entries))
        if self.buckets is not None:
            unknown = set(self.buckets) - set(BUCKET_LABELS)
            if unknown:
                raise ValueError(f"unknown buckets {sorted(unknown)}; use {BUCKET_LABELS}")
            reg = reg[reg.bucket.isin(self.buckets)]
        if self.sources is not None:
            reg = reg[reg.source.isin(self.sources)]
        if self.names != "all":
            wanted = [self.names] if isinstance(self.names, str) else list(self.names)
            keys = {normalize_name(self.display_name(d)): d for d in reg.dataset}
            missing = [n for n in wanted if normalize_name(n) not in keys]
            if missing:
                raise KeyError(f"datasets not in the (filtered) registry: {missing}")
            reg = reg[reg.dataset.isin([keys[normalize_name(n)] for n in wanted])]
        return reg.sort_values(["bucket", "dataset"], kind="stable").reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        """One row per dataset: name, n, d, bucket, classes, source, imputed cells."""
        reg = self.registry
        return pd.DataFrame({
            "name": [self.display_name(d) for d in reg.dataset],
            "n": reg.n.astype(int), "d": reg.d.astype(int), "bucket": reg.bucket,
            "n_classes": reg.n_classes.astype(int), "source": reg.source,
            "imputed_cells": reg.missing.fillna(0).astype(int),
        })

    # ------------------------------------------------------------------- loading
    def _rebuild(self, row) -> pd.DataFrame:
        if row.source == "legacy-cache":
            cls_name, key, _ = LEGACY[row.dataset]
            entry = process_legacy(row.dataset, cls_name, key)
        else:
            version = None if pd.isna(row.openml_version) else int(row.openml_version)
            entry = process_openml(int(row.openml_did), version,
                                   Entry(dataset=row.dataset, source=row.source),
                                   self.data_home_)
        if entry.status != "included" or entry.sha256 != row.sha256:
            raise RuntimeError(
                f"{row.dataset}: rebuilt data does not reproduce the frozen registry "
                f"(status={entry.status}, notes={entry.notes!r}); the upstream source changed")
        return entry.frame

    def _load_one(self, row) -> pd.DataFrame:
        path = self.data_home_ / cache_filename(row.dataset, row.openml_did)
        if path.exists():
            df = read_cache(path)
            if not self.verify or sha256_of(df) == row.sha256:
                return df
            if not self.download_if_missing:
                raise ValueError(f"{path} does not match the registry sha256")
        elif not self.download_if_missing:
            raise FileNotFoundError(f"{path} not found and download_if_missing=False")
        df = self._rebuild(row)
        write_cache(df, path)
        return read_cache(path)

    def download(self):
        """Load the selected datasets into ``content_``."""
        self.data_home_ = resolve_data_home(self.data_home)
        self.data_home_.mkdir(parents=True, exist_ok=True)
        try:
            from tqdm.auto import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda it, **kw: it  # noqa: E731
        rows = list(self.registry.itertuples())
        self.content_ = [
            (self.display_name(r.dataset), self._load_one(r))
            for r in tqdm(rows, desc="Datasets")
        ]
        return self

    # ------------------------------------------------------------------ protocol
    MAX_TEST = 2000  # cap on the test-set size (bounds the cost of exact SHAP)

    @classmethod
    def test_size(cls, n: int):
        """Benchmark test-set size for ``n`` rows: ``min(30%, MAX_TEST rows)``.

        Returns the float 0.3 while the 30% test set fits under the cap (identical to the
        plain 70/30 split used for the original datasets) and the integer ``MAX_TEST``
        otherwise.
        """
        return 0.3 if int(np.ceil(0.3 * n)) <= cls.MAX_TEST else cls.MAX_TEST

    def train_test_split(self, name: str, test_size=None, random_state: int = 42):
        """Benchmark split of one dataset: ``X_train, X_test, y_train, y_test``.

        Same protocol as ``scripts/results/base_model_families.py``, with the test set
        capped at ``MAX_TEST`` rows (``test_size=None``; see :meth:`test_size`):
        ``random_state=42``, labels remapped to {0, 1} (smaller label -> 0).
        """
        df = self[name.upper().replace("_", " ")]
        X = df.drop(columns="target").values
        y = df["target"].values
        if test_size is None:
            test_size = self.test_size(len(df))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        y_tr = np.where(y_tr == y_tr.min(), 0, 1).astype(int)
        y_te = np.where(y_te == y_te.min(), 0, 1).astype(int)
        return X_tr, X_te, y_tr, y_te
