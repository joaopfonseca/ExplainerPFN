#!/usr/bin/env bash
# Launch ExplainerPFN training on a cluster (or locally).
#
# Thin wrapper around scripts/train_explainerpfn.py that:
#   1. resolves the repo root and a Python interpreter with torch installed;
#   2. pins sane, cluster-friendly defaults (checkpoints/logs under the repo);
#   3. reduces CUDA allocator fragmentation across variable-size datasets;
#   4. forwards every extra CLI flag to the Python trainer, so any knob can be
#      overridden without editing this file.
#
# -- Fresh run ------------------------------------------------------------
#     ./scripts/start_explainerpfn.sh
#     ./scripts/start_explainerpfn.sh --device cpu --num-epochs 100
#     ./scripts/start_explainerpfn.sh --model-path notebooks/tabpfn-v2-regressor.ckpt
#
# -- Resume after a crash / preemption ------------------------------------
#     ./scripts/start_explainerpfn.sh --resume checkpoints/checkpoint_5000.pt
#     ./scripts/start_explainerpfn.sh --auto-resume
#
# ``--resume <path>`` takes precedence over ``--auto-resume`` when both are
# given. ``--auto-resume`` finds the latest ``checkpoint_*.pt`` under
# ``--save-dir`` (default ``checkpoints``); if none exists it starts fresh.
#

set -e

TRAIN_ARGS=(
  --model-path "auto"
  --device "auto"
  --save-dir "checkpoints"
  --log-dir "logs"
  --num-epochs "100000"
  --num-batches "1"
  --num-samples "1024"
  --lr "-1e-5"
  --weight-decay "1e-7"
  --max-grad-norm "1.0"
  --save-freq "1000"
  --save-interval "32"
  --log-interval "1000"
  --seed "42"
)

# Reduce allocator fragmentation across variable-size datasets.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/train_explainerpfn.py \
    "${TRAIN_ARGS[@]}" \
    "$@"
