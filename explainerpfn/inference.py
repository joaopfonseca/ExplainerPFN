"""
TODO: Redesign the self.executor_ object (inference can't be done in the same
way it's done in TabPFN)
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal
from typing_extensions import override
import contextlib
import numpy as np
import torch

from tabpfn.model.memory import MemoryUsageEstimator
from tabpfn.inference import InferenceEngine
from explainerpfn.preprocessing import (
    SequentialFeatureTransformer,
    EnsembleConfig,
    fit_preprocessing,
)
from tabpfn.utils import get_autocast_context

from explainerpfn.utils import prepare_explanation_dataset


@dataclass
class InferenceEngineCachePreprocessing(InferenceEngine):
    """Inference engine that caches the preprocessing for feeding as model
    context on predict.

    This will fit the preprocessors on the training data, as well as cache the
    transformed training data on RAM (not GPU RAM).

    This saves some time on each predict call, at the cost of increasing the
    amount of memory in RAM. The main functionality performed at `predict()`
    time is to forward pass through the model which is currently done
    sequentially.

    NOTE: This overrides the original `iter_outputs` method
    """

    X_trains: Sequence[np.ndarray | torch.Tensor]
    y_trains: Sequence[np.ndarray | torch.Tensor]
    model: torch.nn.Module
    cat_ixs: Sequence[list[int]]
    ensemble_configs: Sequence[EnsembleConfig]
    preprocessors: Sequence[SequentialFeatureTransformer]
    force_inference_dtype: torch.dtype | None
    inference_mode: bool
    no_preprocessing: bool = False

    @classmethod
    def prepare(  # noqa: PLR0913
        cls,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        *,
        cat_ix: list[int],
        model,
        ensemble_configs: Sequence[EnsembleConfig],
        n_workers: int,
        rng: np.random.Generator,
        dtype_byte_size: int,
        force_inference_dtype: torch.dtype | None,
        save_peak_mem: bool | Literal["auto"] | float | int,
        inference_mode: bool,
        no_preprocessing: bool = False,
    ) -> "InferenceEngineCachePreprocessing":
        """Prepare the inference engine.

        Args:
            X_train: The training data.
            y_train: The training target.
            cat_ix: The categorical indices.
            model: The model to use.
            ensemble_configs: The ensemble configurations to use.
            n_workers: The number of workers to use.
            rng: The random number generator.
            dtype_byte_size: The byte size of the dtype.
            force_inference_dtype: The dtype to force inference to.
            save_peak_mem: Whether to save peak memory usage.
            inference_mode: Whether to use torch.inference mode
                (this is quicker but disables backpropagation)
            no_preprocessing: If turned of, the preprocessing on the test
                tensors is tuned off. Used for differentiablity.

        Returns:
            The prepared inference engine.
        """
        itr = fit_preprocessing(
            configs=ensemble_configs,
            X_train=X_train,
            y_train=y_train,
            random_state=rng,
            cat_ix=cat_ix,
            n_workers=n_workers,
            parallel_mode="block",
        )
        configs, preprocessors, X_trains, y_trains, cat_ixs = list(zip(*itr))
        return InferenceEngineCachePreprocessing(
            X_trains=X_trains,
            y_trains=y_trains,
            model=model,
            cat_ixs=cat_ixs,
            ensemble_configs=configs,
            preprocessors=preprocessors,
            dtype_byte_size=dtype_byte_size,
            force_inference_dtype=force_inference_dtype,
            save_peak_mem=save_peak_mem,
            inference_mode=inference_mode,
            no_preprocessing=no_preprocessing,
        )

    def prepare_inference_data(
        self,
        preprocessor: SequentialFeatureTransformer,
        X_train: np.ndarray | torch.Tensor,
        y_train: np.ndarray | torch.Tensor,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        feature_idx: int,
        config: EnsembleConfig,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the inference data for by transforming the training and test data.
        """
        X_train, y_train = prepare_explanation_dataset(
            X=X_train,
            y=y_train,
            feature_idx=feature_idx,
        )
        X, y = prepare_explanation_dataset(
            X=X,
            y=y,
            feature_idx=feature_idx,
        )

        if not isinstance(X_train, torch.Tensor):
            X_train = torch.as_tensor(X_train, dtype=torch.float32)  # noqa: PLW2901
        X_train = X_train.to(device)  # noqa: PLW2901

        X_test = preprocessor.transform(X).X if not self.no_preprocessing else X
        if not isinstance(X_test, torch.Tensor):
            X_test = torch.as_tensor(X_test, dtype=torch.float32)
        X_test = X_test.to(device)
        X_full = torch.cat([X_train, X_test], dim=0).unsqueeze(1)

        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(y_train, dtype=torch.float32)  # noqa: PLW2901
        y_train = y_train.to(device)  # noqa: PLW2901

        y_test = (
            config.target_transform.transform(
                y.reshape(-1, 1),
            ).ravel()
            if config.target_transform is not None
            else y
        )
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.as_tensor(y_test, dtype=torch.float32)  # noqa: PLW2901
        y_test = y_test.to(device)  # noqa: PLW2901
        y_full = torch.cat([y_train, y_test], dim=0).unsqueeze(1)

        # batched_cat_ix = [cat_ix]

        # Handle type casting
        with contextlib.suppress(Exception):  # Avoid overflow error
            X_full = X_full.float()
        if self.force_inference_dtype is not None:
            X_full = X_full.type(self.force_inference_dtype)
            y_full = y_train.type(self.force_inference_dtype)  # type: ignore # noqa: PLW2901

        return X_full, y_full

    @override
    def iter_outputs(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        feature_idx: int,
        *,
        device: torch.device,
        autocast: bool,
        only_return_standard_out: bool = True,
    ) -> Iterator[tuple[torch.Tensor | dict, EnsembleConfig]]:
        self.model = self.model.to(device)
        if self.force_inference_dtype is not None:
            self.model = self.model.type(self.force_inference_dtype)
        for preprocessor, X_train, y_train, config, cat_ix in zip(
            self.preprocessors,
            self.X_trains,
            self.y_trains,
            self.ensemble_configs,
            self.cat_ixs,
        ):

            # NOTE: This chunk was moved to its own method
            # to facilitate model training
            X_full, y_full = self.prepare_inference_data(
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X=X,
                y=y,
                feature_idx=feature_idx,
                config=config,
                device=device,
            )

            if self.inference_mode:
                MemoryUsageEstimator.reset_peak_memory_if_required(
                    save_peak_mem=self.save_peak_mem,
                    model=self.model,
                    X=X_full,
                    cache_kv=False,
                    device=device,
                    dtype_byte_size=self.dtype_byte_size,
                    safety_factor=1.2,  # TODO(Arjun): make customizable
                )

            with (
                get_autocast_context(device, enabled=autocast),
                torch.inference_mode(self.inference_mode),
            ):
                output = self.model(
                    X_full,
                    y_full,
                    only_return_standard_out=only_return_standard_out,
                    categorical_inds=[cat_ix],
                    single_eval_pos=len(
                        y_train
                    ),  # len(y_full),  # TODO: check if this is correct
                )

            output = output if isinstance(output, dict) else output.squeeze(1)

            yield output, config
        if self.inference_mode:  # if inference
            self.model = self.model.cpu()

    @override
    def use_torch_inference_mode(self, use_inference: bool):
        self.inference_mode = use_inference
