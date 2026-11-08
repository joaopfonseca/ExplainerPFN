"""Rebuild the frozen dataset registry and the processed-data cache.

This is the curation step: it decides which datasets enter the benchmark pool (UCI
candidates through pinned OpenML mirrors, the eligible legacy datasets, and a seeded,
bucket-stratified top-up from curated OpenML suites and, for the sparsest bucket, the open
OpenML pool), applies the pool criteria and writes
explainerpfn/datasets/registry.py (the included rows as a Python literal) plus the reports
in this directory. Re-running reproduces that file byte for
byte (the OpenML listing, the curated suites and the UCI metadata are pinned under the data
home). Loading the frozen pool is done by explainerpfn.datasets.ExpandedDatasets.

    python scripts/datasets/build_registry.py [--refresh-listing] [--data-home DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_repo_to_path  # noqa: E402

add_repo_to_path()

import report  # noqa: E402
from candidates import (  # noqa: E402
    LEGACY_ADDITIONS,
    LEGACY_UCI_NAME,
    SIMULATED_EXCLUSIONS,
    UCI_CANDIDATES,
    UCI_ID_OVERRIDES,
)

from explainerpfn.datasets.processing import (  # noqa: E402
    BUCKETS,
    COLUMNS,
    LEGACY,
    MAX_CLASSES,
    MAX_D,
    MAX_MISSING,
    MIN_D,
    Entry,
    assign_bucket,
    cache_filename,
    normalize_name,
    process_legacy,
    process_openml,
    resolve_data_home,
    write_cache,
)

OPENML_LIST_URL = (
    "https://www.openml.org/api/v1/json/data/list/status/active/"
    "number_instances/200..20000/number_features/4..400/"
    "number_classes/2..50/limit/10000"
)

SEED = 42
TOTAL_TARGET = 50
BUCKET_TARGETS = {(3, 7): 16, (7, 11): 17, (11, 15): 17}

# UCI candidate -> pinned OpenML (did, version). Hand-verified; no fuzzy matching.
# Candidates without an entry are recorded as skipped ("no OpenML mirror").
UCI_ALIASES = {
    "Maternal Health Risk": (46600, 1),
    "Wholesale customers": (1511, 1),
    "Contraceptive Method Choice": (23, 1),
    "User Knowledge Modeling": (1508, 1),
    "Banknote Authentication": (1462, 1),
    "Blood Transfusion Service Center": (1464, 1),
    "Haberman's Survival": (43, 1),
    "Vertebral Column": (1523, 1),
    "ILPD (Indian Liver Patient Dataset)": (1480, 1),
    "MAGIC Gamma Telescope": (1120, 1),
    "Page Blocks Classification": (30, 1),
    "Statlog (Heart)": (53, 1),
    "Glass Identification": (41, 1),
    "Statlog (German Credit Data)": (31, 1),
    "Website Phishing": (46963, 1),
    "Energy Efficiency": (1472, 1),
    "HTRU2": (45558, 2),
    "Yeast": (181, 1),
    "Abalone": (183, 1),
    "Auction Verification": (44958, 7),
    "Ecoli": (39, 1),
    "NHANES Age Prediction Subset": (46946, 1),
    "Algerian Forest Fires": (46279, 1),
    "Iranian Churn": (None, None),
    "Thoracic Surgery Data": (46595, 1),
    "Risk Factor Prediction of Chronic Kidney Disease": (46602, 1),
    "Estimation of Obesity Levels": (46597, 1),
    "Heart Failure Clinical Records": (46612, 1),
    "Wine Quality": (40498, 1),  # white wine: n=4898 matches the UCI page
}

# Curated OpenML suites the top-up is drawn from (pre-registered by study id).
SUITES = {
    "OpenML-CC18": 99,
    "tabular-numerical-classification": 337,
    "AutoML-benchmark-classification": 271,
    "TabArena-v0.1": 457,
}
STUDY_URL = "https://www.openml.org/api/v1/json/study/{}"

# Normalised names of suite datasets that are the same source as a legacy/UCI entry.
SAME_SOURCE = {
    "creditg", "cmc", "banknoteauthentication", "abalone", "heartstatlog", "anneal",
    "bloodtransfusionservicecenter", "pageblocks", "glass", "ecoli", "yeast", "ilpd",
    "magictelescope", "wine", "vertebracolumn", "haberman", "userknowledge",
    "winequalitywhite", "winequalityred",
}

# Stage-2 (open pool, [11,15) only) manual vetting, frozen here: time-series derived,
# synthetic or unvetted sets, same-source variants of included datasets, and regression
# sets whose target would be binarised.
OPEN_POOL_DENY = re.compile(
    r"^(JapaneseVowels|chatfield_4|calendarDOW|mental_health_detection|boston|bodyfat|"
    r"cpu_small|wind)$|heart|wine|bwin|chronic.?kidney|^FOREX_|^fri_|volcanoes",
    re.I,
)

TOPUP_NAME_DENY = re.compile(
    r"_seed_|stock|price|TESTFORDESCRIPTION|simulat|synthetic|artificial|"
    r"^monk|rent|housing|covid|bitcoin|crypto",
    re.I,
)


def content_key(df: pd.DataFrame) -> str:
    """Row/column-order-invariant identity, used to detect duplicate datasets."""
    cols = sorted(
        hashlib.sha1(np.sort(df[c].to_numpy()).tobytes()).hexdigest()
        for c in df.columns if c != "target"
    )
    tgt = hashlib.sha1(np.sort(df["target"].to_numpy()).tobytes()).hexdigest()
    return hashlib.sha1("|".join([str(df.shape), tgt, *cols]).encode()).hexdigest()


def near_key(df: pd.DataFrame) -> tuple:
    """Coarse identity (n, d, #positives): catches the same data re-coded on another mirror."""
    return (len(df), df.shape[1] - 1, int(df["target"].sum()))


def plan_topup(deficits_by_bucket: dict[str, int], total_needed: int,
               pool_by_bucket: dict[str, list]) -> dict[str, int]:
    """Round-robin allocation of top-up slots over under-target buckets.

    ``deficits_by_bucket``: slots missing to reach the per-bucket target (>= 0).
    Stops at ``total_needed`` or when the pools run out.
    """
    alloc = {b: 0 for b in deficits_by_bucket}
    left = total_needed
    progress = True
    while left > 0 and progress:
        progress = False
        for b in sorted(deficits_by_bucket, key=lambda b: (-deficits_by_bucket[b] + alloc[b], b)):
            if left > 0 and alloc[b] < deficits_by_bucket[b] and alloc[b] < len(pool_by_bucket[b]):
                alloc[b] += 1
                left -= 1
                progress = True
    return alloc


# --------------------------------------------------------------------------- #
# I/O: OpenML + legacy cache
# --------------------------------------------------------------------------- #
def openml_listing(refresh: bool = False, data_home=None) -> list[dict]:
    """Pinned copy of the OpenML dataset listing (cached under data/expanded)."""
    path = resolve_data_home(data_home) / "openml_list.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    with urllib.request.urlopen(OPENML_LIST_URL, timeout=180) as resp:
        raw = json.load(io.TextIOWrapper(resp))["data"]["dataset"]
    rows = []
    for r in raw:
        q = {x["name"]: float(x["value"]) for x in r["quality"]}
        rows.append({
            "did": int(r["did"]), "name": r["name"], "version": int(r["version"]),
            "n": int(q.get("NumberOfInstances", 0)),
            "num": int(q.get("NumberOfNumericFeatures", 0)),
            "classes": int(q.get("NumberOfClasses", 0)),
            "missing": int(q.get("NumberOfMissingValues", 0)),
        })
    rows.sort(key=lambda r: r["did"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return rows


def uci_metadata(refresh: bool = False, data_home=None) -> dict[str, dict]:
    """{UCI name: {uci_id, n, d}} from the UCI API (cached, pinned under data/expanded)."""
    path = resolve_data_home(data_home) / "uci_meta.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    meta = {}
    for name in [*UCI_CANDIDATES, *LEGACY_ADDITIONS]:
        query = (f"id={UCI_ID_OVERRIDES[name]}" if name in UCI_ID_OVERRIDES
                 else "name=" + urllib.parse.quote(name))
        with urllib.request.urlopen(f"https://archive.ics.uci.edu/api/dataset?{query}",
                                    timeout=60) as resp:
            data = json.load(io.TextIOWrapper(resp)).get("data") or {}
        meta[name] = {"uci_id": data.get("uci_id"), "n": data.get("num_instances"),
                      "d": data.get("num_features")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def verify_against_uci(entry: Entry) -> Entry:
    """A mapping is accepted only if it reproduces the UCI page: n within +-1 (header or
    duplicate-row artefacts) and no more numeric features than UCI lists."""
    if entry.status != "included" or entry.n_uci == "":
        return entry
    why = None
    if abs(int(entry.n) - int(entry.n_uci)) > 1:
        why = f"n={entry.n} does not match UCI n={entry.n_uci}"
    elif int(entry.d) > int(entry.d_uci):
        why = f"d={entry.d} exceeds UCI d={entry.d_uci}"
    if why:
        entry.status, entry.frame, entry.notes = "skipped", None, "mapping rejected: " + why
    return entry


def suite_dids(refresh: bool = False, data_home=None) -> dict[int, list[str]]:
    """{did: [suite names]} for the pre-registered curated suites (cached, pinned)."""
    out: dict[int, list[str]] = {}
    for name, sid in SUITES.items():
        path = resolve_data_home(data_home) / f"study_{sid}.json"
        if path.exists() and not refresh:
            ids = json.loads(path.read_text())
        else:
            with urllib.request.urlopen(STUDY_URL.format(sid), timeout=120) as resp:
                ids = sorted(int(i) for i in json.load(io.TextIOWrapper(resp))
                             ["study"]["data"]["data_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(ids))
        for did in ids:
            out.setdefault(did, []).append(name)
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build(refresh: bool = False, log=print, data_home=None) -> list[Entry]:
    entries: list[Entry] = []
    seen_keys: dict = {}  # content_key (str) and near_key (tuple) -> dataset
    seen_names: set[str] = set()

    def register(e: Entry) -> Entry:
        if e.status == "included":
            key, near = content_key(e.frame), near_key(e.frame)
            dup = seen_keys.get(key) or seen_keys.get(near)
            if dup:
                e.status, e.frame = "skipped", None
                e.notes = f"duplicate of {dup}"
            else:
                seen_keys[key] = seen_keys[near] = e.dataset
        entries.append(e)
        return e

    # 1. legacy (precedence: the cache that produced the paper's numbers)
    meta = uci_metadata(refresh, data_home)

    def fill_uci(e: Entry, uci_name: str | None) -> None:
        m = meta.get(uci_name or "")
        if m:
            e.uci_id, e.n_uci, e.d_uci = m["uci_id"], m["n"], m["d"]

    legacy_overlap = {}
    for name, (cls_name, key, uci) in LEGACY.items():
        e = register(process_legacy(name, cls_name, key))
        fill_uci(e, uci or LEGACY_UCI_NAME.get(name))
        if e.n_uci != "" and e.n != e.n_uci:
            e.notes = f"legacy n={e.n} differs from UCI n={e.n_uci}"
        if uci:
            legacy_overlap[uci] = name
        log(f"legacy {name}: {e.status} n={e.n} d={e.d} {e.notes}")

    # 2. UCI candidates through pinned OpenML ids
    for uci in UCI_CANDIDATES:
        e = Entry(dataset=uci, source="uci")
        fill_uci(e, uci)
        if uci in legacy_overlap:
            e.status, e.notes = "skipped", f"covered by legacy cache ({legacy_overlap[uci]})"
            entries.append(e)
            continue
        did, ver = UCI_ALIASES.get(uci, (None, None))
        if did is None:
            e.notes = "no OpenML mirror found"
            entries.append(e)
            continue
        try:
            e = verify_against_uci(process_openml(did, ver, e, data_home))
        except Exception as exc:  # network/parse failure is recorded, not fatal
            e.notes = f"load error: {type(exc).__name__}: {exc}"[:200]
        register(e)
        log(f"uci {uci}: {e.status} n={e.n} d={e.d} {e.notes}")
    for name, why in SIMULATED_EXCLUSIONS.items():
        entries.append(Entry(dataset=name, source="uci", notes=why))

    # 3. OpenML top-up, seeded and bucket-stratified
    included = [e for e in entries if e.status == "included"]
    for e in entries:
        seen_names.add(normalize_name(e.dataset))
    counts = {assign_bucket(lo): 0 for lo, _ in BUCKETS}
    for e in included:
        counts[e.bucket] += 1
    deficits = {assign_bucket(lo): max(0, BUCKET_TARGETS[(lo, hi)] - counts[assign_bucket(lo)])
                for lo, hi in BUCKETS}
    needed = max(0, TOTAL_TARGET - len(included))
    log(f"pool before top-up: {len(included)}, per bucket {counts}, need {needed}")

    suite = suite_dids(refresh, data_home)
    listing = {r["did"]: r for r in openml_listing(refresh, data_home)}
    pool = {}
    for did in sorted(suite):
        r = listing.get(did)
        if r is None:  # outside n/missing/class filters or inactive
            continue
        nm = normalize_name(r["name"])
        if (TOPUP_NAME_DENY.search(r["name"]) or nm in seen_names or nm in SAME_SOURCE
                or r["did"] in {int(v[0]) for v in UCI_ALIASES.values() if v[0]}):
            continue
        if not MIN_D <= r["num"] <= MAX_D + 1:  # loose pre-filter; exact d checked after loading
            continue
        pool.setdefault(nm, dict(r, suites=suite[did]))
        seen_names.add(nm)
    order = sorted(pool.values(), key=lambda r: r["did"])
    random.Random(SEED).shuffle(order)
    log(f"top-up candidate pool: {len(order)} (suites {sorted(SUITES)})")

    got = {b: 0 for b in counts}
    for r in order:
        if needed <= 0:
            break
        e = Entry(dataset=r["name"], source="openml-suite", notes="")
        approx = assign_bucket(r["num"])
        if approx and got[approx] >= deficits[approx] and any(got[b] < deficits[b] for b in got):
            e.openml_did, e.openml_version = r["did"], r["version"]
            e.notes = f"not drawn: bucket {approx} at target"
            entries.append(e)
            continue
        try:
            e = process_openml(r["did"], r["version"], e, data_home)
        except Exception as exc:
            e.notes = f"load error: {type(exc).__name__}: {exc}"[:200]
        if e.status == "included" and (got[e.bucket] >= deficits[e.bucket]
                                       and any(got[b] < deficits[b] for b in got)):
            e.status, e.frame, e.notes = "skipped", None, "bucket already at target"
        if e.status == "included":
            register(e)
            if e.status == "included":
                got[e.bucket] += 1
                needed -= 1
            log(f"topup {r['name']}: {e.status} n={e.n} d={e.d} {e.bucket}")
        else:
            log(f"topup {r['name']}: skipped ({e.notes})")
            entries.append(e)

    # Stage 2: the curated suites cannot fill [11,15); draw the shortfall for that bucket
    # alone from the open OpenML pool (same criteria, vetted denylist, same seed).
    hi = assign_bucket(BUCKETS[-1][0])
    need_hi = BUCKET_TARGETS[BUCKETS[-1]] - sum(
        1 for e in entries if e.status == "included" and e.bucket == hi)
    if need_hi > 0:
        taken = {normalize_name(e.dataset) for e in entries}
        taken_dids = {int(e.openml_did) for e in entries if e.openml_did != ""}
        pool2 = {}
        for r in sorted(openml_listing(refresh, data_home), key=lambda r: (r["name"], r["version"])):
            nm = normalize_name(r["name"])
            frac = r["missing"] / max(r["n"] * max(r["num"], 1), 1)
            if (assign_bucket(r["num"]) != hi or r["classes"] > MAX_CLASSES or frac > MAX_MISSING
                    or TOPUP_NAME_DENY.search(r["name"]) or OPEN_POOL_DENY.search(r["name"])
                    or nm in taken or nm in SAME_SOURCE or r["did"] in taken_dids):
                continue
            pool2.setdefault(nm, r)
        order2 = sorted(pool2.values(), key=lambda r: r["did"])
        random.Random(SEED).shuffle(order2)
        log(f"stage 2: need {need_hi} in {hi}, open-pool candidates {len(order2)}")
        for r in order2:
            if need_hi <= 0:
                break
            e = Entry(dataset=r["name"], source="openml-open", notes="")
            try:
                e = process_openml(r["did"], r["version"], e, data_home)
            except Exception as exc:
                e.notes = f"load error: {type(exc).__name__}: {exc}"[:200]
            if e.status == "included" and e.bucket != hi:
                e.status, e.frame, e.notes = "skipped", None, f"d={e.d} not in {hi}"
            if e.status == "included":
                register(e)
                if e.status == "included":
                    need_hi -= 1
                log(f"open {r['name']}: {e.status} n={e.n} d={e.d} {e.bucket} {e.notes}")
            else:
                log(f"open {r['name']}: skipped ({e.notes})")
                entries.append(e)
    return entries


REGISTRY_MODULE = Path(__file__).resolve().parents[2] / "explainerpfn" / "datasets" / "registry.py"
REGISTRY_FIELDS = [c for c in COLUMNS if c != "status"]


def _py(value) -> str:
    """Python literal of one registry cell (blank -> None, numpy scalars -> int)."""
    if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "":
        return "None"
    if isinstance(value, str):
        return repr(value)
    return repr(int(value))


def registry_module_text(included: pd.DataFrame) -> str:
    """Source of ``explainerpfn/datasets/registry.py`` (deterministic)."""
    lines = [
        '"""Frozen registry of the expanded-benchmark datasets (generated, do not edit).',
        "",
        "Written by ``scripts/datasets/build_registry.py``. One entry per included dataset:",
        "pinned source, dimensionality bucket and sha256 of the processed data.",
        '"""',
        "# flake8: noqa: E501",
        "",
        "REGISTRY = (",
    ]
    for row in included[REGISTRY_FIELDS].itertuples(index=False):
        body = ", ".join(f"{c!r}: {_py(v)}" for c, v in zip(REGISTRY_FIELDS, row))
        lines.append(f"    {{{body}}},")
    lines.append(")")
    return "\n".join(lines) + "\n"


def write_outputs(entries: list[Entry], data_home=None,
                  registry_module: Path = REGISTRY_MODULE) -> pd.DataFrame:
    """Write the cache files, the registry module and the reports (deterministic)."""
    home = resolve_data_home(data_home)
    for e in entries:
        if e.status == "included":
            write_cache(e.frame, home / cache_filename(e.dataset, e.openml_did))
    df = pd.DataFrame([e.row() for e in entries], columns=COLUMNS)
    df["_s"] = df["status"].map({"included": 0, "skipped": 1})
    df = df.sort_values(["_s", "bucket", "dataset"], kind="stable").drop(columns="_s")
    registry_module.write_text(registry_module_text(df[df.status == "included"]))
    report.write_reports(df.replace("", np.nan))  # blanks as NaN, as the reports expect
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh-listing", action="store_true",
                    help="re-download the OpenML listing, suites and UCI metadata")
    ap.add_argument("--data-home", default=None, help="cache directory (default: auto)")
    args = ap.parse_args()

    entries = build(refresh=args.refresh_listing, data_home=args.data_home)
    df = write_outputs(entries, data_home=args.data_home)
    inc = df[df.status == "included"]
    print(f"\nincluded {len(inc)}, skipped {(df.status == 'skipped').sum()}")
    print(inc.bucket.value_counts().sort_index().to_string())
    print(f"registry -> {REGISTRY_MODULE}")


if __name__ == "__main__":
    main()
