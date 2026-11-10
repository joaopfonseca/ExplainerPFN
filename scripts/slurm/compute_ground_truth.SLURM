#!/bin/bash
# Stage A of the expanded benchmark: base models + SHAP ground truth for every dataset.
# One array task per dataset (46 in the frozen registry). Each task runs the five base-model
# families in parallel, one process per family, so every experiment uses the same setup.
# Finished pairs are skipped (--resume), so resubmitting the whole array after a failure or
# timeout is safe. CPU only.
#
# One-time setup on the cluster (compute nodes have no internet access): sync the code
# (explainerpfn/, scripts/, tests/) and the dataset cache (data/expanded/) from the local
# repo, and create ~/slurm_outputs. Then:
#   sbatch scripts/slurm/compute_ground_truth.SLURM
# Results: notebooks/paper_results/pickles/ground_truth/<family>/<dataset>.pkl,
#   logs/benchmarks/ground_truth_runs.csv and logs/ground_truth/ (per-process logs).
# Monitor: python scripts/results/ground_truth_status.py [--pending]; squeue -u $USER;
#   tail -c 300 logs/ground_truth/shard<i>_<family>.log (SHAP progress of one process).
#SBATCH --job-name=epfn_gt
#SBATCH --output=/mnt/storage/admindi/home/jfonseca/slurm_outputs/%x-%A_%a.out
#SBATCH --error=/mnt/storage/admindi/home/jfonseca/slurm_outputs/%x-%A_%a.err
#SBATCH --array=0-45
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --partition=compute
#SBATCH --nodelist=corsa

set -uo pipefail
cd /mnt/storage/admindi/home/jfonseca/ExplainerPFN
source .venv/bin/activate

# Single-threaded BLAS/OpenMP in every process: numerics identical to a local run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# tqdm bars go to log files: refresh at most once a minute to keep them readable.
export TQDM_MININTERVAL=60

FAMILIES="random_forest xgboost mlp logreg knn"
N_SHARDS=46
mkdir -p logs/ground_truth

echo "task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname), families: ${FAMILIES}"
pids=()
for family in ${FAMILIES}; do
  python -u scripts/results/compute_ground_truth.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" --families "${family}" \
    --resume --offline \
    > "logs/ground_truth/shard${SLURM_ARRAY_TASK_ID}_${family}.log" 2>&1 &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
echo "task ${SLURM_ARRAY_TASK_ID} finished with status ${status}"
exit ${status}
