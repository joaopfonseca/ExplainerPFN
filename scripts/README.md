# Experiment scripts

All experiments run from the repository root. Generated artifacts go to
`notebooks/paper_results/` (git-ignored); benchmark CSVs go to `logs/`.

## Checkpoint resolution

Every script that needs a trained model resolves it in this order:

1. `--model-path PATH`
2. `$EPFN_MODEL_PATH`
3. the newest `checkpoints*/final_model.ckp` in the repository, falling back to
   `best_model.ckp` (both `checkpoints/v5/...` and `checkpoints_v5/...` layouts
   are recognised)

`epfn` is the reference model used by the paper.

## Layout

```
scripts/
├── _common.py            shared paths, model resolution, metric helpers
├── run_notebook.py       execute a notebook without Jupyter (cluster nodes)
├── start_explainerpfn.sh canonical training entry point (wraps train/)
├── train/
│   ├── train_explainerpfn.py
│   └── validate_training_data.py
├── eval/
│   ├── benchmark_checkpoints.py
│   ├── benchmark_surrogates.py
│   ├── check_synthetic_shap.py
│   └── stress_scalability.py
└── results/
    ├── bootstrap_cis.py
    ├── runtime_cis.py
    ├── make_tables.py
    ├── dataset_characterization.py
    ├── synthetic_to_real_gap.py
    ├── acs_case_study.py
    ├── dag_reconstruction.py
    ├── base_model_families.py
    ├── black_box_case_study.py
    └── efficiency_residuals.py
```

## Running the notebooks

`run_notebook.py` executes a notebook's code cells in order without
`nbconvert` (not installed on the cluster), leaving the `.ipynb` untouched:

    python scripts/run_notebook.py notebooks/8-analysis-across-datasets.ipynb

Set `$EPFN_MODEL_PATH` when the checkpoint is not under the default
`checkpoints*/final_model.ckp` location (e.g. `checkpoints_v5/` on the cluster).

## Training (`train/`)

| Script | Purpose |
| --- | --- |
| `train_explainerpfn.py` | Trains the foundation model on online exact-SHAP synthetic data. Checkpoints (`checkpoint_*.pt`, `final_model.pt/.ckp`) and logs go to `--save-dir` / `--log-dir`. Use it directly for a fresh run, or with `--resume` / `--auto-resume`. |
| `validate_training_data.py` | Sanity-checks the synthetic data stream (feature counts, sizes, SHAP tiers, efficiency residuals) without training. |

`start_explainerpfn.sh` is the recommended training entry point: it resolves the
repository root, sets cluster-friendly defaults and forwards every extra flag to
`train_explainerpfn.py`.

## Evaluation (`eval/`)

These benchmark a model or a baseline; they write CSV summaries under `logs/`
and never modify the paper artifacts.

| Script | Purpose |
| --- | --- |
| `benchmark_checkpoints.py` | Scores one or more checkpoints against the SHAP references stored in a results pickle, using the paper protocol (fit on train+test, `statistical+additive` corrections, Pearson + top-k Jaccard). `--mode joint` uses the checkpoint's attribution head; `--mode per-feature` the legacy per-feature path. |
| `benchmark_surrogates.py` | Trains the few-shot surrogate baselines (TabPFN / MLP / RF) on 2-10 reference explanations and scores them with the same metrics. `--protocols aligned original` contrasts the scores-conditioned and label-conditioned variants. |
| `check_synthetic_shap.py` | End-to-end smoke test on a synthetic linear problem with analytic Shapley values. |
| `stress_scalability.py` | Zero-shot performance as a function of the number of features on synthetic data (appendix figure). |

## Paper artifacts (`results/`)

These produce the tables and figures of the paper under
`notebooks/paper_results/{tables,figures,pickles,csv,caches}`.

| Script | Notebook it replaces | Output |
| --- | --- | --- |
| `bootstrap_cis.py` | - | `csv/bootstrap_cis_all_methods.csv`, `pickles/bootstrap_cis_all_methods.pkl` |
| `runtime_cis.py` | notebook 8 (time cell) | `csv/runtime_bootstrap_cis.csv`, `tables/results_time_per_1000_cells.tex` |
| `make_tables.py` | - | compact and CI Pearson/Spearman tables for both base predictors |
| `dataset_characterization.py` | - | `figures/dataset_characterization.pdf`, `tables/dataset_characterization_summary.tex` |
| `synthetic_to_real_gap.py` | - | `figures/synthetic_to_real_gap_histograms.pdf`, `tables/synthetic_to_real_gap_summary.tex` |
| `acs_case_study.py` | `notebooks/9-case-study.ipynb` | ACS waterfall and agreement figures |
| `dag_reconstruction.py` | notebook 8 (DAG cells) | DAG ground-truth/predicted and graph-edit-distance figures |
| `base_model_families.py` | `notebooks/11-*.ipynb` (removed) | results pickles/tables/figures for the additional base-model families |
| `black_box_case_study.py` | - | third-party black-box case study figure set |
| `efficiency_residuals.py` | - | Shapley efficiency residuals on the cached synthetic ablations |

The main cross-dataset result pickles (`results_across_datasets_{mlp,random_forest}.pkl`)
are produced by `notebooks/8-analysis-across-datasets.ipynb`; the notebook also
writes the `results_{corr,jaccard}_*.tex` tables and the best/worst-case figures.
The DAG appendix figures are produced by `scripts/results/dag_reconstruction.py`
(parallel graph-edit-distance search). `notebooks/10-ablations.ipynb` produces the correction-ablation figures
consumed by `efficiency_residuals.py`.

## Canonical order

1. Train a checkpoint: `scripts/start_explainerpfn.sh` (cluster).
2. Collect the cross-dataset results: `notebooks/8-analysis-across-datasets.ipynb`
   (with `EPFN_MODEL_PATH` pointing at the checkpoint).
3. Confidence intervals and tables:
   `python scripts/results/bootstrap_cis.py` then
   `python scripts/results/make_tables.py`.
4. Remaining figures: `notebooks/9-case-study.ipynb`, `notebooks/10-ablations.ipynb`,
   and the `results/` scripts listed above.
5. Publish into the LaTeX source:
   `python scripts/results/make_tables.py --paper-tables-dir ../arxiv-v2/tables`
   and copy `notebooks/paper_results/figures/*.pdf` into `../arxiv-v2/figures/`.
