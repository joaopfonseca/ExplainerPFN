# Phase 0 — Dataset collection and frozen registry

Status: **executed, pending user review of the final list** (see "Execution
outcome" at the end). No SHAP computation and no model training have been run.

Parent plan: `.opencode/plans/expanded-dataset-benchmark.md`.

## Goal

Build a reproducible, pre-registered pool of UCI datasets loaded through OpenML
mirrors, with identical preprocessing to the current analysis, and emit a
frozen registry plus a generated dataset-summary table. No SHAP computation and
no model training in this phase.

## Locked decisions

| Decision | Choice |
|---|---|
| Source | UCI candidate names, loaded via OpenML mirrors |
| Pool size | All no-missing eligible datasets (~40 before exclusions) |
| Small datasets | Uniform `n ≥ 200` for all datasets |
| SHAP test cap | 2,000 test rows (applied in Phase 1) |
| Simulated datasets | Excluded |
| Legacy datasets | Eligible ones included in the new pool |

Consequences:

- Echocardiogram (n=61) and Hepatitis (n=129) do **not** enter the new pool under
  uniform `n ≥ 200`. They remain in the legacy 11 per-dataset appendix tables.
- MONK's Problems and Electrical Grid Stability Simulated are excluded, leaving
  **38 UCI candidates** before OpenML mapping and post-load validation.
- Expected final N is **~38–42** after mapping, overlap removal, and validation.
  The exact N is frozen in the registry, not asserted in advance.

## Final pool definition

- Universe: the 40 no-missing UCI datasets found in the survey (listed below).
- Exclude explicitly simulated: MONK's Problems, Electrical Grid Stability
  Simulated.
- Include the legacy 11 that pass the criteria.
- Precedence: for a dataset already in the paper, the existing
  mlresearch-processed cache wins if the new OpenML loader does not reproduce it
  exactly. New datasets use the unified OpenML loader.

### Surveyed no-missing UCI candidates

`[3,7)` (13): Apartment for Rent Classified, MONK's Problems, Maternal Health
Risk, Vertebral Column, Wholesale customers, Contraceptive Method Choice,
Thyroid Disease, User Knowledge Modeling, Banknote Authentication, Blood
Transfusion Service Center, Assessing Mathematics Learning in Higher Education,
Haberman's Survival, Land Mines.

`[7,11)` (18): Forty Soybean Cultivars from Subsequent Harvests, ILPD (Indian
Liver Patient Dataset), MAGIC Gamma Telescope, Page Blocks Classification,
Statlog (Heart), Glass Identification, Statlog (German Credit Data), Website
Phishing, ApisTox, Energy Efficiency, HTRU2, Yeast, Abalone, Auction
Verification, Ecoli, NHANES Age Prediction Subset, Raisin, Rice (Cammeo and
Osmancik).

`[11,15)` (9): Algerian Forest Fires, Iranian Churn, Risk Factor Prediction of
Chronic Kidney Disease, Thoracic Surgery Data, Drug Consumption (Quantified),
Electrical Grid Stability Simulated Data, Estimation of Obesity Levels,
Heart Failure Clinical Records, Wine Quality.

After the simulated exclusions: 38 candidates.

## Steps

### 1. Candidate manifest

Store the 38 candidates with `uci_id`, UCI name, UCI `n`, UCI `d`, UCI class
count, UCI missing flag, and a `simulated` flag. Record the two exclusions with
the reason.

### 2. OpenML mapping

- Fetch metadata for the 424 coarse OpenML candidates and index by
  `original_data_url` and normalized name.
- Match each UCI candidate by UCI URL/name.
- Use a small manual alias table for the remainder (e.g., German Credit →
  `credit-g`, Banknote → `banknote-authentication`).
- Do **not** rely on fuzzy name matching. The survey produced false positives
  (Page Blocks → Glass Classification, Raisin → jasmine, Thyroid Disease → a
  1-feature dataset).
- Verify every mapping by loading it and checking `n`, `d`, target type, and
  class count against the UCI page.
- Record unmapped candidates as `skipped` with a reason.
- Pin the OpenML dataset id and version.

### 3. Implement `scripts/datasets/build_registry.py` (+ `explainerpfn/datasets/processing.py`)

- `fetch_openml(data_id=..., version=..., as_frame=True, parser="auto")`.
- Identify the target via `default_target_attribute`.
- Drop `category`/object columns (the `cat_` equivalent).
- Drop explicit ID/index/sequence columns and zero-variance columns.
- Require zero remaining NaNs; skip otherwise.
- Binarize multiclass targets majority-vs-rest, matching the current protocol.
- Compute actual `n`, `d`, class counts, dropped columns, and a content hash.
- Cache `data/expanded/<did>.csv.gz` (git-ignored) plus sha256.
- Emit `registry.py` with columns:
  `dataset, uci_id, openml_did, openml_version, source, n_uci, d_uci, n, d,
  bucket, n_classes, missing, dropped_columns, sha256, status, notes`.

### 4. Legacy validation gate

- Reconstruct the 70/30 split (`random_state=42`) for the 9 eligible legacy
  datasets and compare `X`, `y`, and stored SHAP arrays against the existing
  `notebooks/paper_results/pickles/results_across_datasets_*.pkl` files.
- Assert equality. If the unified OpenML loader differs, keep the legacy cache
  for those datasets and record the discrepancy in the registry notes.

### 5. Freeze and report

- Commit `registry.py` (metadata only; raw data stays git-ignored).
- Generate `datasets_summary.tex` (name, n, d, classes) for the full pool.
- Emit a bucket-count table and the mapping/skip report.
- **Stop for user review of the final dataset list before Phase 1.**

## Deliverables

- `scripts/datasets/build_registry.py` and `explainerpfn/datasets/processing.py` plus unit tests.
- `data/expanded/` cache (git-ignored).
- `registry.py` (committed).
- Generated `datasets_summary.tex`.
- Mapping/skip report.

## Acceptance criteria

- Re-running the build reproduces the manifest byte-for-byte.
- All 38 candidates accounted for (included or skipped with a reason).
- Legacy datasets reproduce the current `n`/`d` and arrays exactly.
- No missing values, duplicates, ID-leakage columns, or out-of-range `d`/`n`.
- User approves the final list.

## Tests

- Bucket boundaries: `3,6 → [3,7)`; `7,10 → [7,11)`; `11,14 → [11,15)`.
- Deterministic manifest (same inputs and seed → identical CSV).
- Dedup by content hash and dataset identity.
- Rejection of NaN-containing datasets.
- Registry round-trip: cached CSV reloads to the same `n`/`d`/hash.
- Legacy equivalence gate.

## Notes carried into Phase 1

- Test cap of 2,000 rows, uniform across old and new datasets.
- Exact SHAP at `d = 14` with 2,000 test rows is the main cost risk; the Phase 1
  plan will include sharding and a runtime cutoff.
- Multiclass binarization and numeric-only input are inherited from the current
  protocol and will be stated as limitations.

## Relevant files

- `scripts/results/base_model_families.py` — current loader/protocol to mirror.
- `scripts/_common.py` — paths and helpers.
- `notebooks/paper_results/pickles/results_across_datasets_{mlp,random_forest}.pkl`
  — legacy validation targets.
- `~/mlresearch_data/*.csv` — legacy processed datasets.
- `data/expanded/` — new processed-data cache (git-ignored).
- `registry.py` — frozen manifest (committed).

## Execution outcome (supersedes the pool-size and legacy counts above)

Deviations from the approved plan, each decided with the user during execution:

- **Target is 50 datasets total**, not ~38-42. The OpenML top-up was added to reach it.
- **Missing values allowed up to 20%** (fraction of missing cells in the retained numeric
  features, median-imputed) instead of "no missing". Target missing values reject a dataset.
- **Top-up source**: seeded, bucket-stratified draw from curated OpenML suites
  (OpenML-CC18 = 99, tabular numerical classification = 337, AutoML benchmark = 271,
  TabArena-v0.1 = 457), then a second stage from the open pool for `[11,15)` only, with a
  frozen manual denylist (`OPEN_POOL_DENY` in `registry.py`).
- **Legacy eligible under `n >= 200` = 8**, not 9: Flags (n=194) is also out, alongside
  Echocardiogram and Hepatitis. Legacy datasets always take the mlresearch cache (source
  `legacy-cache`); the OpenML copies of the same UCI datasets are skipped as overlaps.
- Every UCI mapping is verified against the UCI API (`n` within +-1, `d <= d_uci`);
  Wine Quality maps to the white-wine mirror (n=4898 matches the UCI page).
- Targets with more than 20 distinct values are rejected as regression (Energy Efficiency,
  Auction Verification, NHANES). Heart Failure uses `death_event` as target (override).

Result (frozen in `explainerpfn/datasets/registry.py`): **46 included**,
`[3,7)` = 16, `[7,11)` = 17, `[11,15)` = 13 (target 17; the open pool is exhausted).
Reproducibility: two builds give a byte-identical registry and cache. Legacy gate: 8/8
reproduce the stored `X`/label partition (older pickles store raw `{1,2}` labels for Heart
and German Credit; compared after the `{0,1}` remap).

Files: library code in `explainerpfn/datasets/` (`processing.py` = loading/processing
helpers, `_datasets.py` = `ExpandedDatasets`, a subclass of the mlresearch `Datasets` base
class, and the frozen `registry.py`); curation, reports and validation in
`scripts/datasets/` (`build_registry.py` + `candidates.py` = pool curation, `report.py`,
`legacy_gate.py`, generated `datasets_summary.tex` and `mapping_report.md`); tests in
`tests/test_dataset_registry.py`. Usage: `from explainerpfn.datasets import ExpandedDatasets`.
