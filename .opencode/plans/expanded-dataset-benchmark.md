# Expanded multi-family, bucket-aggregated benchmark

Status: **agreed plan, not executed.** Each phase below will be expanded into its
own detailed sub-plan and executed separately, after approval.

## Objective

Replace the main-body per-dataset results (currently 11 UCI datasets, per-dataset
tables) with an aggregate over a larger, criteria-defined dataset pool, grouped by
dimensionality. This addresses the (good) risk that the much-improved v5 results
look cherry-picked, and frees main-body space to report several base-model
families instead of only RF.

## Decisions (locked)

| Question | Decision |
|---|---|
| Dataset pool | Criteria-based, ~50 total, OpenML top-up |
| Families in main table | 5 core: RF, XGBoost, MLP, LogReg, SVM |
| Cell content | `EPFN mean / best few-shot mean`, winner bold |
| Per-dataset transparency | Full per-dataset appendix table for all N |

### Refinements agreed in discussion

- **Stratified random sample, not a curated top-up.** Define one eligible
  universe (all four mlresearch suites + OpenML), then sample 16/17/17 datasets
  per dimensionality bucket uniformly at random with a fixed seed; take all if a
  bucket has fewer. Prevents the artifact of adding only high-dimensional
  datasets to fill `[11,15)`.
- **Freeze the registry before running.** Commit `explainerpfn/datasets/registry.py`
  (id, source, name, n, d, bucket, sha256) before any cluster run. Raw data stays
  git-ignored.
- **"Best few-shot"** = per dataset, max over the 15 surrogate configs
  (2/4/6/8/10 shots × TabPFN/RF/MLP), then averaged over datasets. Deliberately
  conservative: it is an upper bound on the baseline. Also report the best fixed
  config and EPFN's mean rank.
- **Aggregation unit = dataset.** Bucket mean ± SD, 95% bootstrap over datasets
  (not instances), Fisher-z for Pearson (raw mean in main, z-mean in appendix).
  Paired Wilcoxon vs each surrogate, Holm-corrected; top-1/top-2 win rate.
- **Keep the original 11 as a sensitivity subset**, reported next to the full
  pool aggregate.

## Verified starting facts

- Current evaluation datasets (11) have `d ∈ {4,5,6,7,8,10}`, so the proposed
  `[11,15)` bucket is currently empty; new datasets are mandatory.
- A 7-family harness already exists and ran on the same 11 datasets:
  `scripts/results/base_model_families.py`,
  `notebooks/paper_results/csv/base_model_families_per_base_model.csv` (77 rows),
  `notebooks/paper_results/tables/base_model_families_cross_family_table.tex`.
  None of it is in the paper yet.
- The 7×11 family run took 45:50 on one A30 (`epfn_refresh_c`). 7×50 is roughly
  4–8 h wall, dominated by exact SHAP growing as `2^d`; shard on corsa.
- OpenML API is reachable from corsa; the `openml` package is absent, but
  `sklearn.datasets.fetch_openml` (sklearn 1.6.1) works without it.
- The existing mlresearch suites (Binary, ImbalancedBinary,
  ContinuousCategorical, MultiClass) contain roughly ~30 unique datasets in
  `3 ≤ d ≤ 14` (to be confirmed in Phase 0).
- Hardcoded old setup in the paper: `experiments.tex:5` ("11 tabular datasets"),
  `:61` ("5 of 11"), `appendix.tex:227-236` (gap uses 11), `:293`
  (characterization over n=11).

## Evaluation criteria (pre-registered)

- Real tabular classification; `3 ≤ d ≤ 14` numeric features **after**
  preprocessing.
- `200 ≤ n ≤ 20,000`.
- At most 20% missing values (numerics median-imputed).
- Numeric-only, dropping categorical columns, to match the current protocol.
- Multiclass targets binarised majority-vs-rest, matching the current protocol.
- Non-degenerate: base-model AUC ≥ 0.55 and `shap.std() > 1e-8`.
- Deduplicated by name and by content hash (e.g. Heart vs Cleveland).
- Dataset must not appear in training (training is synthetic, so all real
  datasets qualify).

## Phases

### Phase 0 — Registry and data prep

Build and freeze the dataset pool.

- New `scripts/datasets/build_registry.py` (curation) and `explainerpfn/datasets/processing.py` (loading/processing):
  - enumerate candidates from the four mlresearch suites and OpenML;
  - preprocess (drop `cat_`, median-impute numerics, binarise targets,
    stratified subsample to `n ≤ 20k`);
  - apply the criteria above and skip rules;
  - compute sha256, write `explainerpfn/datasets/registry.py`, cache
    `data/expanded/<id>.csv.gz` (git-ignored);
  - stratified random sample 16/17/17 per bucket with a fixed seed.
- OpenML listing/filtering via the REST API with `urllib`; pin dataset IDs and
  versions.
- Emit a generated `datasets_summary.tex` for all N.

Acceptance: registry reproducible from scratch; re-running the sampler with the
same seed reproduces the manifest byte-for-byte; **stop for user review of the
sampled dataset list before any benchmark run.**

### Phase 1 — Expanded benchmark

- Refactor `scripts/results/base_model_families.py` into
  `scripts/results/expanded_benchmark.py`:
  - `--registry`, `--families {rf,xgboost,mlp,logreg,svm}`, `--shard i/N`,
    `--resume`;
  - exact same protocol: 70/30 split, seed 42, exact SHAP on test, EPFN float32
    joint head, `apply_correction(["statistical","additive"])`, few-shot 2–10;
  - one pickle per `(dataset, family)` under
    `notebooks/paper_results/pickles/expanded/` (git-ignored), plus one appended
    row per run in `logs/benchmarks/expanded_runs.csv`.
- SLURM array on corsa, sharded by dataset.
- **Numerical-equivalence gate**: re-run the 11 existing datasets through the new
  script and assert Pearson values match
  `base_model_families_per_base_model.csv` to ~1e-9 before launching the full
  pool.

Acceptance: equivalence gate passes; all `(dataset, family)` pickles present;
runtime logged.

### Phase 2 — Aggregation, tables, figures

- New `scripts/results/aggregate_buckets.py`:
  - main full-width `table*`: rows = `[3,7)`, `[7,11)`, `[11,15)`, `Overall`;
    columns = RF / XGB / MLP / LogReg / SVM; cell = `EPFN mean / best-surrogate
    mean`, winner bold; final column = mean rank (1–16) and win rate;
  - appendix aggregates for Spearman and Jaccard, with bucket counts, SD,
    dataset-level CIs, Wilcoxon p-values;
  - sensitivity table: original 11 vs full pool;
  - runtime table per family: SHAP vs EPFN per 1000 cells, speedup;
  - figures: EPFN vs best-surrogate by bucket × family with dataset-level CIs.
- Recompute characterization (`d` vs performance) and synthetic-to-real gap over
  the full pool.

Acceptance: every table and figure is generated, no hand-edited numbers;
bucket counts and CIs are reported honestly.

### Phase 3 — Paper

- `sections/experiments.tex`: replace per-dataset main tables with the bucket
  table + figure; rewrite datasets/protocol paragraphs; replace "5 of 11" with
  win-rate/rank/CI statements; reframe around the dimensionality regime and
  speed.
- `sections/appendix.tex`: full per-dataset tables for all N, one longtable per
  family (rows grouped by bucket, columns = EPFN + 15 surrogate configs,
  scriptsize/landscape); update dataset summary, gap, characterization, and all
  "11" references.
- Keep the original 11 as a sensitivity subset in appendix/supplement.
- Add OpenML citation; update captions and cross-references.

Acceptance: `latexmk -pdf -f` clean; no stale numbers or references.

### Phase 4 — Verification

- Unit tests: bucket boundaries (3,6 → `[3,7)`; 7,10 → `[7,11)`; 11,14 →
  `[11,15)`), dedup, deterministic stratified sample, dataset-level bootstrap,
  Fisher-z, rank/win-rate, numerical-equivalence check.
- Two-dataset smoke test, full refresh, `pytest -q`, `latexmk`.

## Risks and notes

- **Expected result**: the `[11,15)` bucket will likely pull the aggregate down.
  The claim should be "competitive across model families, with a clear
  low-/high-dimensional regime and a large speed advantage", not "best on most
  datasets".
- **Appendix size**: 5 families × 16 methods × 50 datasets is ~5 longtables. If
  page pressure bites, collapse to EPFN + TabPFN@2/@10 + RF@10 per family and put
  the full grid in the supplementary CSV; decide before drafting.
- **Categorical features**: dropping them is inherited from the current protocol;
  state it as a limitation.
- **OpenML flakiness**: pin IDs+versions and checksums; the registry build must be
  restartable and fail loudly on checksum mismatch.
- **Exact SHAP at `d = 14`** is the runtime risk; sharding handles it.

## Relevant files

- `scripts/results/base_model_families.py` — protocol to reuse/refactor.
- `scripts/results/make_tables.py`, `scripts/results/bootstrap_cis.py` —
  per-dataset table/CI pipeline; hardcoded to 11 datasets and 16 methods.
- `scripts/_common.py` — paths and helpers (`resolve_model_path`, Jaccard).
- `notebooks/paper_results/{csv,pickles,tables,figures}/` — git-ignored outputs.
- `../paper-ijcai/sections/experiments.tex`,
  `../paper-ijcai/sections/appendix.tex`, `../paper-ijcai/tables/`.
- `logs/benchmarks/`, `logs/refresh/c1_base_model_families.log` — existing
  family results and timing.
