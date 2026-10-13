"""Train ExplainerPFN on online exact-do-Shapley synthetic data.

Python port of ``notebooks/6-train-explainerpfn.ipynb``.
It keeps the notebook's training contract unchanged:

  * data is generated online by :class:`TrainingBatchIterator` (no pickle
    files), each batch carrying ``X, y, shap, executor_configs,
    feature_exp_std``;
  * for every feature we call ``xai.forward(...)`` against the borrowed
    executor and minimize ``xai.bardist_(logits.T, shap / feature_exp_std)``;
  * one optimizer step per epoch (a "batch" is one generated dataset).

On top of the notebook it adds:

  * an argument parser for all model/data/training knobs;
  * periodic, atomic checkpoints (``checkpoint_<epoch>.pt``) containing the
    model, optimizer, LR scheduler, epoch, best loss, and the exact dataset
    index / RNG state needed to resume the online data stream;
  * ``--resume <path>`` and ``--auto-resume`` in the style of
    DiffusionExplainerPFN, plus SIGTERM/SIGINT handling that flushes a
    checkpoint before exiting (so preemption restarts lose at most the
    current epoch);
  * a human-readable log file and an optional JSONL loss history.

Usage::

    # Fresh run on one GPU
    python scripts/train_explainerpfn.py --device cuda --num-epochs 100000

    # Resume from a specific checkpoint
    python scripts/train_explainerpfn.py --resume checkpoints/checkpoint_5000.pt

    # Resume from the latest checkpoint in --save-dir (or start fresh)
    python scripts/train_explainerpfn.py --auto-resume

The launched shell wrapper ``scripts/start_explainerpfn.sh`` sets up the
interpreter and sane cluster defaults; it is the recommended entry point.

NOTE: retraining invalidates previously trained checkpoints.
"""

import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explainerpfn.base import ExplainerPFN  # noqa: E402
from explainerpfn.model.attribution_head import (  # noqa: E402
    bucket_means,
    efficiency_penalty,
)
from explainerpfn.train.synthetic_data import (  # noqa: E402
    SyntheticDataGenerator,
    TrainingBatchIterator,
)
from explainerpfn.utils import _retry_io, find_latest_checkpoint  # noqa: E402

try:  # tqdm is optional on minimal cluster images
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


# ── Logging ─────────────────────────────────────────────────────────────


class _Logger:
    """Print to stdout and, if configured, append to a log file."""

    def __init__(self, log_path=None):
        self.log_path = log_path
        self._fh = open(log_path, "a", buffering=1) if log_path else None

    def __call__(self, msg):
        print(msg, flush=True)
        if self._fh is not None:
            self._fh.write(msg + "\n")

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ── Checkpoint I/O ──────────────────────────────────────────────────────


@_retry_io(max_retries=4, base_delay=1.0, backoff=2.0)
def _atomic_torch_save(path, payload):
    """``torch.save`` to ``path`` atomically (tmp + ``os.replace``).

    Wrapped in :func:`_retry_io` so transient NFS errors during a checkpoint
    write do not kill the run, and atomicity guarantees a preempted write can
    never leave a corrupt checkpoint under the final name.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise


def save_checkpoint(
    path,
    xai,
    optimizer,
    scheduler,
    epoch,
    best_avg_loss,
    losses,
    data_start_index,
    rng,
    args,
    logger,
):
    """Write a full training checkpoint (model + optim + scheduler + state)."""
    # The trainset KV cache is an inference-only optimisation and can hold
    # large per-dataset tensors; drop it so checkpoints stay model-sized.
    xai.model_.empty_trainset_representation_cache()
    payload = {
        "model_state_dict": xai.model_.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler else None),
        "epoch": int(epoch),
        "global_step": int(epoch),
        "best_avg_loss": float(best_avg_loss),
        "losses": list(losses),
        "data_start_index": int(data_start_index),
        "rng_state": rng.bit_generator.state,
        "config": vars(args),
    }
    if getattr(xai, "head_", None) is not None:
        payload["head_state_dict"] = xai.head_state_dict()
        payload["head_config"] = xai.head_config_
    _atomic_torch_save(path, payload)
    logger(f"Saved checkpoint: {path} (epoch {epoch}, data index {data_start_index})")


def load_checkpoint(path, xai, optimizer, device, logger):
    """Load model/optimizer/epoch state. Returns a state dict.

    Returns:
        dict with keys ``epoch``, ``best_avg_loss``, ``losses``,
        ``data_start_index``, ``rng_state``, ``config`` (any of which may be
        missing in older checkpoints).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    xai.model_.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    else:
        logger("WARNING: checkpoint has no optimizer state; optimizer starts fresh.")

    if checkpoint.get("head_state_dict") is not None:
        if getattr(xai, "head_", None) is None:
            logger(
                "WARNING: checkpoint contains an attribution head but this run "
                "was started without --joint-head; head state ignored."
            )
        else:
            xai.head_.load_state_dict(checkpoint["head_state_dict"])
            xai.head_.to(device)
            logger("Loaded attribution head state from checkpoint.")

    # The LR scheduler state is deliberately NOT restored: its ``T_max`` was
    # the *old* target, and a resume may legitimately extend training. The
    # caller instead sets ``scheduler.last_epoch`` from the resumed epoch so
    # the cosine curve anneals to the new ``--num-epochs`` endpoint.

    state = {
        "epoch": checkpoint.get("epoch", checkpoint.get("global_step", 0)),
        "best_avg_loss": checkpoint.get("best_avg_loss", float("inf")),
        "losses": checkpoint.get("losses", []),
        "data_start_index": checkpoint.get("data_start_index", 0),
        "rng_state": checkpoint.get("rng_state", None),
        "config": checkpoint.get("config", {}),
    }
    logger(
        f"Loaded checkpoint {path}: epoch={state['epoch']}, "
        f"best_avg_loss={state['best_avg_loss']:.4f}, "
        f"data_start_index={state['data_start_index']}"
    )
    return state


# ── Argument parsing ────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train ExplainerPFN on online exact-do-Shapley synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument(
        "--model-path",
        type=str,
        default="auto",
        help="Base checkpoint to start from ('auto' downloads the stock TabPFN "
        "v2 regressor). Ignored for weights when --resume is given, but still "
        "used for architecture/bardistribution.",
    )
    p.add_argument("--device", type=str, default="auto", help="cuda | cpu | auto")
    p.add_argument("--seed", type=int, default=42)

    # Training
    p.add_argument("--num-epochs", type=int, default=100000)
    p.add_argument("--num-batches", type=int, default=1, help="Datasets per epoch.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Rows kept per dataset (subsample cap; smaller datasets are kept whole).",
    )
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-7)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "none"],
        help="LR schedule over num_epochs.",
    )
    p.add_argument("--log-interval", type=int, default=1000, help="Print cadence (epochs).")
    p.add_argument(
        "--save-interval",
        type=int,
        default=32,
        help="Window (epochs) for the best-average-loss 'best' save.",
    )

    # Checkpointing / resume
    p.add_argument("--save-dir", type=str, default="checkpoints")
    p.add_argument(
        "--save-freq",
        type=int,
        default=1000,
        help="Write a periodic checkpoint every N epochs (0 disables).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from. Takes precedence over "
        "--auto-resume.",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest checkpoint_*.pt in --save-dir, or start "
        "fresh if none exists.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for train.log and losses.jsonl (default: --save-dir).",
    )

    # Data distribution (mirrors SyntheticDataGenerator / DAGGenerator)
    g = p.add_argument_group("data distribution")
    g.add_argument("--n-features-range", type=int, nargs=2, default=[3, 15])
    g.add_argument(
        "--n-samples-range",
        type=int,
        nargs=2,
        default=[250, 5000],
        help="Rows per generated dataset, inclusive high.",
    )
    g.add_argument("--n-dags-range", type=int, nargs=2, default=[1, 4])
    g.add_argument("--nodes-per-dag-range", type=int, nargs=2, default=[3, 8])
    g.add_argument("--edge-prob-range", type=float, nargs=2, default=[0.2, 0.4])
    g.add_argument(
        "--sink-sigmoid-steepness-range", type=float, nargs=2, default=[0.5, 3.0]
    )
    g.add_argument("--linear-dag-prob", type=float, default=0.05)
    g.add_argument("--mlp-prob", type=float, default=0.15)
    g.add_argument("--mlp-sink-prob", type=float, default=0.25)
    g.add_argument("--mlp-hidden-units-range", type=int, nargs=2, default=[8, 16])
    g.add_argument("--noise-std", type=float, default=0.1)
    g.add_argument("--feature-correlation", type=float, default=0.3)
    g.add_argument("--beta-marginals", action="store_true", default=False)
    g.add_argument("--mc-samples", type=int, default=100)
    g.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Rejection-sampling retries per dataset (d must stay in range).",
    )
    g.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Skip datasets whose n_samples * n_features reaches this bound "
        "(disabled by default).",
    )

    # Joint attribution head
    j = p.add_argument_group("joint attribution head")
    j.add_argument(
        "--joint-head",
        action="store_true",
        default=False,
        help="Train the joint multi-attribution head alongside the backbone.",
    )
    j.add_argument(
        "--efficiency-weight",
        type=float,
        default=0.1,
        help="Weight of the efficiency penalty "
        "(sum_j phi_j - (f(x) - base))^2.",
    )
    j.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="Learning rate for the head. Defaults to --lr.",
    )
    j.add_argument("--head-n-buckets", type=int, default=256)
    j.add_argument(
        "--head-hidden-dim",
        type=int,
        default=None,
        help="Internal width of the slot mixer (default: backbone embedding size).",
    )
    j.add_argument("--head-n-layers", type=int, default=2)
    j.add_argument("--head-nhead", type=int, default=6)
    j.add_argument(
        "--head-max-features",
        type=int,
        default=None,
        help="Largest number of feature slots (default: n_features_range[1]).",
    )

    args = p.parse_args(argv)

    # Normalise tuple-like args and record the exact resume-relevant data config.
    args.n_features_range = tuple(args.n_features_range)
    args.n_samples_range = tuple(args.n_samples_range)
    args.n_dags_range = tuple(args.n_dags_range)
    args.nodes_per_dag_range = tuple(args.nodes_per_dag_range)
    args.edge_prob_range = tuple(args.edge_prob_range)
    args.sink_sigmoid_steepness_range = tuple(args.sink_sigmoid_steepness_range)
    args.mlp_hidden_units_range = tuple(args.mlp_hidden_units_range)
    return args


# ── Data iterator ───────────────────────────────────────────────────────


def make_data_iterator(args, start_index=0):
    """Build a streaming iterator matching the notebook's ``make_data_iterator``."""
    generator = SyntheticDataGenerator(
        n_features_range=args.n_features_range,
        n_samples_range=args.n_samples_range,
        max_attempts=args.max_attempts,
        random_state=args.seed,
        verbose=False,
        n_dags_range=args.n_dags_range,
        nodes_per_dag_range=args.nodes_per_dag_range,
        edge_prob_range=args.edge_prob_range,
        sink_sigmoid_steepness_range=args.sink_sigmoid_steepness_range,
        linear_dag_prob=args.linear_dag_prob,
        mlp_prob=args.mlp_prob,
        mlp_sink_prob=args.mlp_sink_prob,
        mlp_hidden_units_range=args.mlp_hidden_units_range,
        noise_std=args.noise_std,
        feature_correlation=args.feature_correlation,
        beta_marginals=args.beta_marginals,
        mc_samples=args.mc_samples,
    )
    return TrainingBatchIterator(
        generator=generator,
        max_datasets=None,  # stream indefinitely: the data is online
        max_cells=args.max_cells,
        start_index=start_index,
        prefetch=1,
        device="cpu",
        random_state=args.seed,
        verbose=False,
    )


# ── Training loop ───────────────────────────────────────────────────────


def train(args, logger):
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    logger(f"Device: {device}")
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        logger(
            f"GPU: {torch.cuda.get_device_name(0)} | "
            f"free {free / 1024**2:.1f}MB / total {total / 1024**2:.1f}MB"
        )

    # Model + optimizer + scheduler (same construction as notebook 6).
    head_kwargs = {
        "n_buckets": args.head_n_buckets,
        "hidden_dim": args.head_hidden_dim,
        "n_layers": args.head_n_layers,
        "nhead": args.head_nhead,
        "max_features": args.head_max_features or args.n_features_range[1],
    }
    xai = ExplainerPFN(
        fit_mode="low_memory",
        model_path=args.model_path,
        random_state=args.seed,
        device=device,
        joint_head=args.joint_head,
        head_kwargs=head_kwargs,
    )
    xai._initialize_model_variables()
    xai.model_.train()

    head_lr = args.head_lr if args.head_lr is not None else args.lr
    param_groups = [
        {
            "params": list(xai.model_.parameters()),
            "lr": args.lr,
            "initial_lr": args.lr,
            "name": "backbone",
        }
    ]
    trainable_params = list(xai.model_.parameters())
    if xai.head_ is not None:
        xai.head_.train()
        n_head_params = sum(p.numel() for p in xai.head_.parameters())
        logger(f"Joint attribution head: {n_head_params} params, lr={head_lr}")
        param_groups.append(
            {
                "params": list(xai.head_.parameters()),
                "lr": head_lr,
                "initial_lr": head_lr,
                "name": "head",
            }
        )
        trainable_params += list(xai.head_.parameters())

    optimizer = optim.Adam(
        param_groups, lr=args.lr, weight_decay=args.weight_decay
    )

    rng = np.random.default_rng(args.seed)

    # Resolve resume source: --resume wins over --auto-resume.
    resume_state = None
    ckpt_path = None
    if args.resume is not None:
        ckpt_path = args.resume
        logger(f"[resume] Explicit checkpoint: {ckpt_path}")
    elif args.auto_resume:
        ckpt_path = find_latest_checkpoint(args.save_dir)
        if ckpt_path is None:
            logger(f"[auto-resume] No checkpoint in {args.save_dir}; starting fresh.")
        else:
            logger(f"[auto-resume] Latest checkpoint: {ckpt_path}")

    start_epoch = 0
    best_avg_loss = float("inf")
    losses = []
    data_start_index = 0
    if ckpt_path is not None:
        resume_state = load_checkpoint(
            ckpt_path, xai, optimizer, device, logger
        )
        # Checkpoints record the epoch just *completed* and the next dataset
        # index, so training resumes at the following epoch.
        start_epoch = int(resume_state["epoch"]) + 1
        best_avg_loss = float(resume_state["best_avg_loss"])
        losses = list(resume_state["losses"])
        data_start_index = int(resume_state["data_start_index"])
        if resume_state["rng_state"] is not None:
            rng.bit_generator.state = resume_state["rng_state"]
            logger("Restored RNG state from checkpoint.")
        # ``optimizer.load_state_dict`` restores the checkpoint's *old* LR into
        # the param groups, which would desync the scheduler. Reset each group
        # to its own base LR so the freshly-built scheduler anneals correctly.
        for group in optimizer.param_groups:
            base_lr = group.get("initial_lr", args.lr)
            group["lr"] = base_lr
            group["initial_lr"] = base_lr

    # Scheduler is built AFTER resume so its base LR is ``--lr`` and its T_max
    # is the (possibly extended) ``--num-epochs`` target.
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        if ckpt_path is not None:
            # Replaying ``start_epoch`` steps positions the schedule exactly
            # where a contiguous run would be at the start of that epoch.
            for _ in range(start_epoch):
                scheduler.step()

    data = make_data_iterator(args, start_index=data_start_index)

    # Graceful preemption: finish the current epoch, flush a checkpoint, exit.
    stop = {"requested": False}

    def _handle_signal(signum, _frame):
        stop["requested"] = True
        logger(f"Received signal {signum}; will checkpoint and exit after this epoch.")

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    epoch_iter = range(start_epoch, args.num_epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", initial=start_epoch,
                          total=args.num_epochs)

    best_path = os.path.join(args.save_dir, "best_model.ckp")
    last_epoch = start_epoch - 1
    try:
        for epoch in epoch_iter:
            optimizer.zero_grad()
            epoch_losses = []
            epoch_head_losses = []
            epoch_eff_losses = []
            last_dataset_index = data_start_index

            for _ in range(args.num_batches):
                batch = data.next_batch()
                last_dataset_index = batch["dataset_index"] + 1
                indices = rng.permutation(len(batch["X"]))

                X_batch = batch["X"][indices][-args.num_samples:]
                y_batch = batch["y"][indices][-args.num_samples:]
                shap_batch = batch["shap"][indices][-args.num_samples:]
                executor_configs = batch["executor_configs"]

                batch_loss = 0.0
                feature_embeddings = []
                for feature_idx in range(X_batch.shape[1]):
                    xai.executor_ = executor_configs
                    xai.executor_[feature_idx].model = xai.model_
                    xai.executor_[feature_idx].use_torch_inference_mode(False)

                    with torch.enable_grad():
                        if xai.head_ is not None:
                            feature_logits, emb = xai.forward_with_embeddings(
                                X_batch, y_batch, feature_idx
                            )
                            feature_embeddings.append(emb)
                        else:
                            feature_logits, outputs, borders = xai.forward(
                                X_batch,
                                y_batch,
                                feature_idx=feature_idx,
                                only_return_standard_out=True,
                            )
                            del outputs, borders  # borders is a numpy array

                    target_shap = torch.tensor(
                        shap_batch[:, feature_idx] / batch["feature_exp_std"],
                        dtype=torch.float32,
                        device=xai.device_,
                    )
                    # bardist lives on CPU (no fit() call moved it to device).
                    feature_loss = xai.bardist_(
                        feature_logits.cpu().T, target_shap.cpu()
                    ).mean()
                    batch_loss = batch_loss + feature_loss

                batch_loss = batch_loss / X_batch.shape[1]

                if xai.head_ is not None:
                    embeddings = torch.stack(feature_embeddings, dim=1)
                    features_t = torch.tensor(
                        X_batch, dtype=torch.float32, device=xai.device_
                    )
                    score_t = torch.tensor(
                        y_batch, dtype=torch.float32, device=xai.device_
                    )
                    head_logits = xai.head_(embeddings, features_t, score_t)
                    n_s, n_f, n_b = head_logits.shape
                    flat = head_logits.reshape(n_s * n_f, n_b)
                    target_flat = torch.tensor(
                        shap_batch / batch["feature_exp_std"],
                        dtype=torch.float32,
                        device=xai.device_,
                    ).reshape(-1)
                    head_loss = xai.head_criterion_(
                        flat.cpu(), target_flat.cpu()
                    ).mean()
                    means = bucket_means(
                        flat, xai.head_centers_.to(flat.device)
                    ).reshape(n_s, n_f)
                    # The exact do-Shapley labels satisfy sum_j phi_j = f(x) -
                    # base_value, so their row sums are the efficiency target.
                    target_sum = torch.tensor(
                        shap_batch.sum(axis=1) / batch["feature_exp_std"],
                        dtype=torch.float32,
                        device=xai.device_,
                    )
                    eff_loss = efficiency_penalty(means, target_sum)
                    batch_loss = (
                        batch_loss + head_loss + args.efficiency_weight * eff_loss
                    )
                    epoch_head_losses.append(float(head_loss.item()))
                    epoch_eff_losses.append(float(eff_loss.item()))

                if torch.isfinite(batch_loss):
                    epoch_losses.append(batch_loss)
                else:
                    logger(
                        f"Epoch {epoch}: non-finite batch loss; skipping batch."
                    )

            step_ok = len(epoch_losses) > 0
            if step_ok:
                total_loss = torch.stack(epoch_losses).mean()
                step_ok = bool(torch.isfinite(total_loss))
            else:
                total_loss = torch.tensor(float("nan"), device=xai.device_)
            if step_ok:
                total_loss.backward()
            else:
                # Never call backward/step on a non-finite loss: NaN gradients
                # would corrupt every weight and crash the next forward pass.
                logger(
                    f"Epoch {epoch}: non-finite loss ({total_loss.item():.4f}); "
                    "skipping optimizer step."
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

            total_grad_norm = 0.0
            for prm in trainable_params:
                if prm.grad is not None:
                    total_grad_norm += prm.grad.norm().item() ** 2
            total_grad_norm = total_grad_norm**0.5

            if step_ok:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, args.max_grad_norm
                )
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if step_ok:
                losses.append(float(total_loss.item()))
            data_start_index = last_dataset_index
            last_epoch = epoch

            if epoch % args.log_interval == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"Epoch {epoch}, Loss: {total_loss.item():.3f}, "
                    f"Grad Norm: {total_grad_norm:.3f}, LR: {current_lr:.2e}"
                )
                if epoch_head_losses:
                    msg += (
                        f", Head Loss: {np.mean(epoch_head_losses):.3f}"
                        f", Eff: {np.mean(epoch_eff_losses):.4f}"
                    )
                if device.type == "cuda":
                    free, total = torch.cuda.mem_get_info()
                    msg += f", Free Mem: {free / 1024**2:.1f}MB"
                logger(msg)
                if tqdm is not None:
                    epoch_iter.set_postfix(loss=f"{total_loss.item():.3f}")

            # Keep the best model by average loss over the trailing window.
            last_avg_loss = float(np.mean(losses[-args.save_interval:]))
            if epoch >= args.save_interval and best_avg_loss > last_avg_loss:
                best_avg_loss = last_avg_loss
                xai.save_foundation_model(best_path)
                logger(f"Saved best model (avg loss {best_avg_loss:.4f}): {best_path}")

            # Periodic checkpoint.
            if args.save_freq and epoch > 0 and epoch % args.save_freq == 0:
                save_checkpoint(
                    os.path.join(args.save_dir, f"checkpoint_{epoch}.pt"),
                    xai, optimizer, scheduler, epoch, best_avg_loss, losses,
                    data_start_index, rng, args, logger,
                )

            if stop["requested"]:
                save_checkpoint(
                    os.path.join(args.save_dir, f"checkpoint_{epoch}.pt"),
                    xai, optimizer, scheduler, epoch, best_avg_loss, losses,
                    data_start_index, rng, args, logger,
                )
                logger("Preemption requested; checkpoint written. Exiting.")
                break
    finally:
        data.close()

    # Final artifacts: a full training checkpoint and a loadable foundation model.
    save_checkpoint(
        os.path.join(args.save_dir, "final_model.pt"),
        xai, optimizer, scheduler, last_epoch,
        best_avg_loss, losses, data_start_index, rng, args, logger,
    )
    xai.save_foundation_model(os.path.join(args.save_dir, "final_model.ckp"))
    logger(f"Training complete. Wrote final_model.pt and final_model.ckp to {args.save_dir}")
    return losses


# ── Entry point ─────────────────────────────────────────────────────────


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.save_dir, exist_ok=True)
    log_dir = args.log_dir or args.save_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = _Logger(os.path.join(log_dir, "train.log"))

    logger("=" * 70)
    logger(f"ExplainerPFN training | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger("=" * 70)
    logger("Arguments: " + json.dumps(vars(args), default=str, sort_keys=True))

    try:
        losses = train(args, logger)
        # Rewrite (not append): on resume ``losses`` already contains the
        # restored history, so appending would duplicate earlier epochs.
        with open(os.path.join(log_dir, "losses.jsonl"), "w") as fh:
            for i, loss in enumerate(losses):
                fh.write(json.dumps({"epoch": i, "loss": loss}) + "\n")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
