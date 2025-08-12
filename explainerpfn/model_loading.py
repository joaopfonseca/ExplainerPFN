import warnings
import logging

# import contextlib
import torch

# import urllib.request
# from dataclasses import dataclass
# from enum import Enum
from pathlib import Path

# from urllib.error import URLError
from typing import Any, Literal
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


def load_model(
    *,
    path: Path,
) -> tuple[
    nn.Module,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    ArchitectureConfig,
]:
    """Loads a model from a given path. Only for inference.

    Args:
        path: Path to the checkpoint
    """
    # TODO: See below.
    # Catch the `FutureWarning` that torch raises. This should be dealt with!
    # The warning is raised due to `torch.load`, which advises against ckpt
    # files that contain non-tensor data.  This `weights_only=None` is the
    # default value. In the future this will default to `True`, dissallowing
    # loading of arbitrary objects.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        checkpoint = torch.load(path, map_location="cpu", weights_only=None)

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

    loaded_model, criterion, config = load_model(path=model_path)
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
