import warnings
import logging

# import contextlib
import torch

# import urllib.request
# from dataclasses import dataclass
# from enum import Enum
from pathlib import Path

# from urllib.error import URLError
from typing import Any, Literal, cast
from typing_extensions import TypeAlias
from torch import nn
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.loading import (
    get_n_out,
    get_loss_criterion,
    resolve_model_path,
    download_model,
)
from explainerpfn.model import parse_config, get_architecture

logger = logging.getLogger(__name__)

# TODO: Define ArchitectureConfig
ArchitectureConfig: TypeAlias = (
    Any  # Placeholder for the actual architecture config type
)


def _reconstitute_foundation_checkpoint(
    trainer_checkpoint: dict,
    reference_checkpoint: dict,
    *,
    source_path: str | Path,
    reference_path: str | Path,
) -> dict:
    """Build a foundation-format payload from a trainer checkpoint.

    Trainer checkpoints (``scripts/train_explainerpfn.py``) store the
    fine-tuned weights under ``model_state_dict`` but neither the
    bar-distribution (``criterion.*``) state nor the architecture config.
    Both are training-invariant -- the trainer's optimizer only ever held
    ``model_.parameters()`` -- so they are sourced from
    ``reference_checkpoint``, the foundation checkpoint the run was
    fine-tuned from.
    """
    model_state = trainer_checkpoint["model_state_dict"]
    reference_state = reference_checkpoint["state_dict"]

    unknown = set(model_state) - set(reference_state)
    if unknown:
        raise ValueError(
            f"Trainer checkpoint {source_path} has tensor keys that are "
            f"absent from the reference foundation checkpoint "
            f"{reference_path}: {sorted(unknown)[:5]}. The reference is "
            "probably not the model this checkpoint was fine-tuned from."
        )
    for key, tensor in model_state.items():
        reference_tensor = reference_state[key]
        if (
            torch.is_tensor(tensor)
            and torch.is_tensor(reference_tensor)
            and tensor.shape != reference_tensor.shape
        ):
            raise ValueError(
                f"Shape mismatch for '{key}' between trainer checkpoint "
                f"{source_path} ({tuple(tensor.shape)}) and reference "
                f"checkpoint {reference_path} "
                f"({tuple(reference_tensor.shape)})."
            )

    logger.info(
        f"Trainer checkpoint {source_path} (epoch "
        f"{trainer_checkpoint.get('epoch', '?')}): using its model weights, "
        f"with criterion and architecture config from {reference_path}. "
        "Optimizer, scheduler and RNG state are ignored (to resume training, "
        "use scripts/train_explainerpfn.py --resume)."
    )
    return {
        "state_dict": {**reference_state, **model_state},
        "config": reference_checkpoint["config"],
    }


def _resolve_foundation_reference(
    trainer_checkpoint: dict,
    *,
    which: Literal["regressor", "classifier"],
    version: str,
    download: bool,
) -> Path:
    """Find a foundation-format reference for a trainer checkpoint.

    Preference order: the foundation model the run was fine-tuned from, as
    recorded in the trainer checkpoint's saved arguments
    (``config["model_path"]``), if it exists on this machine; otherwise the
    default stock TabPFN checkpoint (downloaded once if necessary).
    """
    config = trainer_checkpoint.get("config")
    if isinstance(config, dict):
        recorded = config.get("model_path")
        if isinstance(recorded, (str, Path)) and Path(recorded).exists():
            return Path(recorded)

    model_path, model_dir, model_name, _ = resolve_model_path(
        None, which, version
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        if not download:
            raise ValueError(
                "A foundation reference is required to load a trainer "
                "checkpoint, the default model path does not exist, and "
                f"downloading is disabled\nmodel path: {model_path}"
            )
        logger.info(f"Downloading foundation reference model to {model_path}.")
        res = download_model(
            model_path,
            version=version,
            which=cast("Literal['classifier', 'regressor']", which),
            model_name=model_name,
        )
        if res != "ok":
            raise RuntimeError(
                f"Failed to download the foundation reference model to "
                f"{model_path}! Trainer checkpoints need the stock model "
                "the run was fine-tuned from. Place a foundation-format "
                "checkpoint (a .ckp file) at the trainer checkpoint's "
                "recorded 'model_path', or at the default model path."
            ) from res[0]
    return model_path


def load_model(
    *,
    path: Path,
    which: Literal["regressor", "classifier"] = "regressor",
    version: str = "v2",
    download: bool = True,
) -> tuple[
    nn.Module,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ArchitectureConfig,
]:
    """Loads a model from a given path. Only for inference.

    Both checkpoint formats written by this project are accepted: foundation
    checkpoints (``{"state_dict", "config"}`` -- the stock TabPFN models or
    ``*.ckp`` files written by ``save_foundation_model``) and trainer
    checkpoints from ``scripts/train_explainerpfn.py`` (recognized by their
    ``model_state_dict`` key). For trainer checkpoints only the model weights
    are used; the bar distribution and architecture config are taken from the
    foundation checkpoint the run was fine-tuned from.

    Args:
        path: Path to the checkpoint
        which: Which stock model to fetch as the foundation reference when
            ``path`` is a trainer checkpoint whose recorded reference is not
            available on this machine.
        version: The version of the stock reference model.
        download: Whether the stock reference model may be downloaded if
            missing.
    """
    # TODO: See below.
    # Catch the `FutureWarning` that torch raises. This should be dealt with!
    # The warning is raised due to `torch.load`, which advises against ckpt
    # files that contain non-tensor data.  This `weights_only=None` is the
    # default value. In the future this will default to `True`, dissallowing
    # loading of arbitrary objects.
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore", category=FutureWarning)
    #     checkpoint = torch.load(path, map_location="cpu", weights_only=None)
    checkpoint = torch.load(path, map_location="cpu", weights_only=None)

    if (
        isinstance(checkpoint, dict)
        and "state_dict" not in checkpoint
        and "model_state_dict" in checkpoint
    ):
        reference_path = _resolve_foundation_reference(
            checkpoint, which=which, version=version, download=download
        )
        reference = torch.load(
            reference_path, map_location="cpu", weights_only=None
        )
        checkpoint = _reconstitute_foundation_checkpoint(
            checkpoint,
            reference,
            source_path=path,
            reference_path=reference_path,
        )

    if (
        not isinstance(checkpoint, dict)
        or "state_dict" not in checkpoint
        or "config" not in checkpoint
    ):
        found = (
            sorted(checkpoint.keys())
            if isinstance(checkpoint, dict)
            else type(checkpoint).__name__
        )
        raise ValueError(
            f"Checkpoint {path} matches neither supported format: expected a "
            "foundation checkpoint with keys 'state_dict' and 'config', or a "
            "trainer checkpoint from scripts/train_explainerpfn.py with key "
            f"'model_state_dict'. Found: {found}."
        )

    state_dict = checkpoint["state_dict"]
    config, unused_config = parse_config(checkpoint["config"])
    logger.debug(
        "Keys in config that were not parsed by architecture config: "
        f"{', '.join(unused_config.keys())}"
    )

    criterion_state_keys = [k for k in state_dict if "criterion." in k]
    loss_criterion = get_loss_criterion(config)
    if isinstance(loss_criterion, FullSupportBarDistribution):
        # Remove from state dict
        criterion_state = {
            k.replace("criterion.", ""): state_dict.pop(k) for k in criterion_state_keys
        }
        loss_criterion.load_state_dict(criterion_state)
    else:
        assert len(criterion_state_keys) == 0, criterion_state_keys

    model = get_architecture(
        config,
        n_out=get_n_out(config, loss_criterion),
        cache_trainset_representation=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, loss_criterion, config


def load_model_criterion_config(
    model_path: None | str | Path,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
    download: bool,
) -> tuple[
    nn.Module,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ArchitectureConfig,
]:
    """Load the model, criterion, and config from the given path.

    Args:
        model_path: The path to the model.
        check_bar_distribution_criterion:
            Whether to check if the criterion
            is a FullSupportBarDistribution, which is the expected criterion
            for models trained for regression.
        cache_trainset_representation:
            Whether the model should know to cache the trainset representation.
        which: Whether the model is a regressor or classifier.
        version: The version of the model.
        download: Whether to download the model if it doesn't exist.

    Returns:
        The model, criterion, and config.
    """
    (model_path, model_dir, model_name, which) = resolve_model_path(
        model_path, which, version
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        if not download:
            raise ValueError(
                f"Model path does not exist and downloading is disabled"
                f"\nmodel path: {model_path}",
            )

        logger.info(f"Downloading model to {model_path}.")
        res = download_model(
            model_path,
            version=version,
            which=cast("Literal['classifier', 'regressor']", which),
            model_name=model_name,
        )
        if res != "ok":
            repo_type = "clf" if which == "classifier" else "reg"
            raise RuntimeError(
                f"Failed to download model to {model_path}!\n\n"
                f"For offline usage, please download the model manually from:\n"
                f"https://huggingface.co/Prior-Labs/TabPFN-v2-{repo_type}/resolve/main/{model_name}\n\n"
                f"Then place it at: {model_path}",
            ) from res[0]

    loaded_model, criterion, config = load_model(
        path=model_path, which=which, version=version, download=download
    )
    loaded_model.cache_trainset_representation = cache_trainset_representation
    if check_bar_distribution_criterion and not isinstance(
        criterion,
        FullSupportBarDistribution,
    ):
        raise ValueError(
            f"The model loaded, '{model_path}', was expected to have a"
            " FullSupportBarDistribution criterion, but instead "
            f" had a {type(criterion).__name__} criterion.",
        )
    return loaded_model, criterion, config
