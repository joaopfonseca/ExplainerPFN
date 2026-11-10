#!/bin/bash
# Stage A of the expanded benchmark: base models + SHAP ground truth for every dataset.
# One array task per dataset (46 in the frozen registry); each task runs all families for
# its dataset and skips pairs that already have an artifact, so the whole array can be
# resubmitted safely after a failure or timeout. CPU only.
#
# One-time setup on the cluster (compute nodes have no internet access):
#   git pull                                  # in ~/ExplainerPFN, needs the Phase 0/1 commits
#   rsync -a data/expanded/ fcul-cluster:ExplainerPFN/data/expanded/   # from the local repo
#   mkdir -p ~/slurm_outputs
# Submit:
#   sbatch scripts/slurm/compute_ground_truth.SLURM
#   sbatch --array=7,12 scripts/slurm/compute_ground_truth.SLURM       # selected shards only
#   FAMILIES="logreg knn" sbatch scripts/slurm/compute_ground_truth.SLURM
# Results: notebooks/paper_results/pickles/ground_truth/<family>/<dataset>.pkl and
#   logs/benchmarks/ground_truth_runs.csv (rsync both back).
#SBATCH --job-name=epfn_gt
#SBATCH --output=/mnt/storage/admindi/home/jfonseca/slurm_outputs/%x-%A_%a.out
#SBATCH --error=/mnt/storage/admindi/home/jfonseca/slurm_outputs/%x-%A_%a.err
#SBATCH --array=0-45%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --partition=compute
#SBATCH --nodelist=corsa

set -euo pipefail
cd /mnt/storage/admindi/home/jfonseca/ExplainerPFN
source .venv/bin/activate

# Single-threaded BLAS/OpenMP: keeps the numerics identical to a local run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

FAMILIES=${FAMILIES:-"random_forest xgboost mlp logreg knn"}
N_SHARDS=${N_SHARDS:-46}

echo "task ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} on $(hostname), families: ${FAMILIES}"
python -u scripts/results/compute_ground_truth.py \
  --shard "${SLURM_ARRAY_TASK_ID}/${N_SHARDS}" \
  --families ${FAMILIES} \
  --resume --offline ${EXTRA_ARGS:-}
