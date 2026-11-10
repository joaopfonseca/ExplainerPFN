# Phase 1 — Expanded benchmark run

Status: **Stage A implemented and smoke-tested** (`compute_ground_truth.py`); timing probe,
Stage B adaptation and the full run are still pending.

Parent plan: `.opencode/plans/expanded-dataset-benchmark.md`. Input: the frozen
registry from Phase 0 (`explainerpfn.datasets.ExpandedDatasets`, 46 datasets).

## Goal

Run the frozen ExplainerPFN (EPFN) and the 15 few-shot surrogates against exact SHAP
for every `(dataset, base-model family)` pair of the registry, with exactly the
protocol of the current paper results, and store one pickle per pair. No
aggregation or paper edits (Phase 2/3).

## Protocol (unchanged from `base_model_families.py` / notebook 8)

- 70/30 split, `random_state=42`, labels remapped to {0, 1}
  (`ExpandedDatasets.train_test_split` already does this).
- Base model fitted on the full training split.
- Ground truth: `shap.explainers.Exact` on `predict_proba[:, 1]`, background = scaled
  `X_train`.
- EPFN: float32, `fit(X_train ++ X_test, scores)`, `predict(X_test, scores)`, keys
  `explainerpfn`, `explainerpfn_statistical+additive`, `explainerpfn_additive`.
- Few-shot surrogates: TabPFN / MLP / RFR at n = 2, 4, 6, 8, 10 (15 keys + times).
- Result schema identical to the existing `results_across_datasets_<family>.pkl`
  entries, so `make_tables.py` / `bootstrap_cis.py` logic keeps working.

## Changes relative to the current harness

1. **Dataset source**: `ExpandedDatasets` instead of the mlresearch loader. The old
   `load_datasets` filters are no longer needed (the registry already applies them).
2. **Families (decided: five)**: `random_forest`, `xgboost`, `mlp`, `logreg`, `knn`,
   defined once in the new script so the RF/MLP configs are no longer notebook-only
   (RF: 500 trees `RandomForestClassifier`; MLP: `(12, 12)`, `max_iter=5000`).
   Deep MLP and Naive Bayes stay on the original datasets only (limitation to state).
3. **Split rule (decided)**: one rule for every dataset: the test set has
   `min(ceil(0.3 n), 2000)` rows and the base model trains on the rest
   (`random_state=42`). Float 0.3 is passed when `n <= 6666` (identical to the stored
   legacy splits, so the 8 legacy datasets and 38 pool datasets are unchanged) and the
   integer 2000 above it (8 datasets; train = `n - 2000` instead of 70%).
   Implemented once in `ExpandedDatasets.train_test_split` (class constant for the cap),
   used by Stage A and the tests. SHAP, EPFN (fit context = full dataset) and the
   few-shot surrogates all use this test set; no separate subsampling. State in the
   paper as "70/30 split, test set capped at 2,000 rows".
4. **Sharding / resume** (Stage A): one pickle per pair (see the revised structure);
   `--resume` skips existing pickles; `--shard i/N` takes every N-th dataset of the
   sorted registry; one row per run appended to `logs/benchmarks/expanded_runs.csv`
   (dataset, family, n, d, bucket, n_test, AUC, shap_time, epfn_time, status).
5. **Pre-registered quality fields** (recorded, not filtered here): base-model test
   AUC and `shap.std()`, so Phase 2 can apply "AUC ≥ 0.55 and shap.std > 1e-8"
   without reruns.
6. The metrics/tables/figures half of `base_model_families.py` stays where it is; the
   new script only produces pickles and the run log.

## Revised structure: two stages (supersedes the single-runner design)

The expensive, checkpoint-independent part is separated from the evaluation, because
`scripts/eval/benchmark_checkpoints.py` and `benchmark_surrogates.py` already evaluate
EPFN checkpoints and the few-shot surrogates against a stored results pickle
(`X_train/X_test/y_*`, fitted `base_model`, `y_*_scores`, `shap`). They never train base
models or compute the SHAP ground truth.

**Stage A — ground-truth artifacts** (new `scripts/results/compute_ground_truth.py`,
sharded, resumable, run once per dataset × family):
split → fit base model → SHAP on the test rows (capped at 2,000) → one pickle per pair at
`notebooks/paper_results/pickles/ground_truth/<family>/<dataset>.pkl` (git-ignored) with
the existing schema minus the EPFN/few-shot keys, plus `shap_time`, `shap_method`,
`n_train`, `n_test`, base-model test AUC and `shap.std()`. Appends a row to
`logs/benchmarks/ground_truth_runs.csv`. Real datasets have no SCM, so the artifacts are
the fitted models, their scores and the attributions; synthetic SCMs stay in
`explainerpfn/train`.

**Stage B — evaluation** (adapt `scripts/eval/benchmark_*.py`, no new runner):
- read a directory of Stage-A artifacts as well as a single pickle (loader in `_common`);
- also report Spearman, per-method runtime (EPFN fit/predict, few-shot fit), and save the
  predictions (`--save-predictions`) so Phase 2 can bootstrap the CIs;
- keep the EPFN variants (`explainerpfn`, `explainerpfn_statistical+additive`,
  `explainerpfn_additive`) and the surrogate protocol that reproduces the 15 notebook-8
  keys (`original`);
- shard by dataset and `--resume` for the SLURM array.

Benefits: new checkpoints and surrogate settings never recompute SHAP; the equivalence
gate splits cleanly into Stage A against the stored `X`/`shap`/scores and Stage B
against the stored Pearson values; the approximation choice lives in one place.

New or changed files: `scripts/results/compute_ground_truth.py`,
`scripts/results/ground_truth_slurm.sh`, edits to `scripts/_common.py` and
`scripts/eval/benchmark_{checkpoints,surrogates}.py`, `tests/test_ground_truth.py`.

## Numerical-equivalence gate (before the full run)

Re-run the 8 legacy datasets × the 5 families through the new script and compare with
`notebooks/paper_results/pickles/results_across_datasets_*.pkl`:

- split arrays, base-model scores and `shap`: exact or `allclose` (same library
  versions, CPU);
- EPFN / few-shot predictions and the Pearson values in
  `csv/base_model_families_per_base_model.csv`: start at 1e-9, relax only if GPU
  float32 inference proves non-reproducible, and record the tolerance used.

The full pool is launched only if the gate passes.

## Cost control (decided: uniform 2,000 cap, approximate SHAP as fallback)

Exact SHAP cost grows as `2^d × background × test rows × model cost`; at `d = 14` with
2,000 test rows and a 500-tree forest or a k-NN scorer over ~18k training rows this is the dominant risk.
The test-set cap is 2,000 rows for every pair. When exact SHAP is too slow, the ground truth
for that pair falls back to an approximation:

Fallback for every family: `PermutationExplainer` (same 100-row Independent background
as `Exact`, `seed=42`, `max_evals=1000`), approximate and recorded as such. Interventional
`TreeExplainer` on the probability output was tried and **rejected**: it differs from
`Exact` by up to 0.01 (RF) and 0.11 (XGBoost) on Heart, and its additivity against
`predict_proba` is off, whereas `Exact` is additive to 1e-15. Measured permutation
accuracy on Heart vs `Exact`: Pearson 0.9999 (RF) and 0.9990 (MLP). Permutation only
pays off when `2^d` exceeds `max_evals` (it is slower than `Exact` at `d = 6`), i.e. for
the high-`d` cells the probe identifies.

Rules, so that results stay reproducible:
1. **Timing probe first** (one dataset per bucket, `d = 14`, every family, 100 rows; also time the
   base-model fit, k-NN prediction cost grows with the training set) to
   measure exact-SHAP seconds per row and decide, per `(family, d)`, whether exact is
   affordable under a stated per-pair budget. This table is frozen before the full run.
2. The method used is chosen from that frozen table, not from wall-clock behaviour during
   the run. A wall-clock limit remains only as a safety net: a pair that exceeds it is
   rerun with the fallback and flagged `shap_method_fallback=timeout`.
3. Every pickle and run-log row records `shap_method` (`exact`, `permutation`). Phase 2 reports the main results on all pairs and a sensitivity check
   restricted to exact pairs; the paper states the approximate cells.
4. Permutation agreement with `Exact` is part of the tests (small `d`).

## Execution

- SLURM array on corsa, one task per dataset (all families inside a task, so the
  EPFN model and TabPFN load once), `--shard $SLURM_ARRAY_TASK_ID/46`, GPU for EPFN
  and TabPFN, CPU for SHAP.
- Smoke test first: two datasets, one family, locally (`.venv`).

## Acceptance

- Equivalence gate passes on the 8 legacy datasets for all families.
- 46 × 5 pickles present, or a logged and explained status for each missing pair.
- Run log complete with runtimes, AUC and the `n_test` actually used.
- Tests pass; no change to Phase 0 files.

## Decisions (recorded)

1. Families: five (RF, XGBoost, MLP, LogReg, k-NN).
2. SHAP cost: uniform 2,000-row cap; approximate SHAP (permutation; tree SHAP was
   rejected, see Cost control) when exact is too slow, chosen by a frozen timing-probe table.
3. Equivalence gate: try 1e-9; relax to the tightest tolerance that holds and document it.
