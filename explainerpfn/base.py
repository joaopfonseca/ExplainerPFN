import typing
from typing import Any, Literal
from typing_extensions import TypeAlias
from collections.abc import Sequence

import numpy as np
import torch
from torch.types import _dtype
from sklearn import config_context
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

# from tabpfn import TabPFNRegressor
from tabpfn.base import initialize_tabpfn_model, check_cpu_warning, determine_precision
from tabpfn.preprocessing import (
    RegressorEnsembleConfig,
    ReshapeFeatureDistributionsStep,
    EnsembleConfig,
    PreprocessorConfig,
    default_regressor_preprocessor_configs
)
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.utils import (
    validate_Xy_fit,
    infer_categorical_features,
    update_encoder_params,
    _fix_dtypes,
    _get_ordinal_encoder,
    _process_text_na_dataframe,
)
from tabpfn.inference import InferenceEngineCachePreprocessing
from tabpfn.config import ModelInterfaceConfig

XType: TypeAlias = Any
SampleWeightType: TypeAlias = Any
YType: TypeAlias = Any


class ExplainerPFN:  # (TabPFNRegressor):
    """
    A class to handle the SHAP values for a PFN model.

    NOTE: This is intended to be a proof of concept. It's meant to handle only
    preprocessed data with standardized features and no missing values.
    """
    interface_config_: ModelInterfaceConfig

    def __init__(
        self,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature=0.9,
        model_path="auto",
        device=None,
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state=None,
        n_jobs: int = -1,
    ):
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.model_path = model_path
        self.device = device
        self.inference_precision = inference_precision
        self.memory_saving_mode = memory_saving_mode
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _initialize_dataset_preprocessing(
        self, X: XType, y: YType, rng: np.random.Generator
    ) -> tuple[list[RegressorEnsembleConfig], XType, YType, FullSupportBarDistribution]:
        """
        NOTE: This will need to be entirely rewritten.
        """

        X, y, feature_names_in, n_features_in = validate_Xy_fit(
            X,
            y,
            estimator=self,
            ensure_y_numeric=False,
            max_num_samples=self.interface_config_.MAX_NUMBER_OF_SAMPLES,
            max_num_features=self.interface_config_.MAX_NUMBER_OF_FEATURES,
            ignore_pretraining_limits=True,
        )

        assert isinstance(X, np.ndarray)
        check_cpu_warning(
            self.device, X, allow_cpu_override=True
        )

        if feature_names_in is not None:
            self.feature_names_in_ = feature_names_in
        self.n_features_in_ = n_features_in

        self.inferred_categorical_indices_ = infer_categorical_features(
            X=X,
            provided=self.categorical_features_indices,
            min_samples_for_inference=self.interface_config_.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,  # noqa: E501
            max_unique_for_category=self.interface_config_.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
            min_unique_for_numerical=self.interface_config_.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
        )

        # Will convert inferred categorical indices to category dtype,
        # to be picked up by the ord_encoder, as well
        # as handle `np.object` arrays or otherwise `object` dtype pandas columns.
        X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
        # Ensure categories are ordinally encoded
        ord_encoder = _get_ordinal_encoder()
        X = _process_text_na_dataframe(
            X,
            ord_encoder=ord_encoder,
            fit_encoder=True,  # type: ignore
        )
        self.preprocessor_ = ord_encoder

        possible_target_transforms = (
            ReshapeFeatureDistributionsStep.get_all_preprocessors(
                num_examples=y.shape[0],  # Use length of validated y
                random_state=rng,  # Use the provided rng
            )
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for (
            y_target_preprocessor
        ) in self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS:
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None
            target_preprocessors.append(preprocessor)
        preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS

        # TODO: NEEDS TO BE MODIFIED TO PRESERVE THE ORDER OF CERTAIN FEATURES
        ensemble_configs = EnsembleConfig.generate_for_regression(
            n=n_features_in,  # refers to the number of estimators
            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=typing.cast(
                "Sequence[PreprocessorConfig]",
                preprocess_transforms
                if preprocess_transforms is not None
                else default_regressor_preprocessor_configs(),
            ),
            target_transforms=target_preprocessors,
            random_state=rng,
        )

        self.bardist_ = self.bardist_.to(self.device_)

        assert len(ensemble_configs) == self.n_features_in_

        return ensemble_configs, X, y, self.bardist_

    def _initialize_model_variables(self) -> tuple[int, np.random.Generator]:
        """
        TODO: refactor to use the new model.
        """
        rng = np.random.default_rng(self.random_state)
        static_seed = (
            int(self.random_state)
            if isinstance(self.random_state, int)
            else rng.integers(0, 2**31 - 1)
        )

        self.model_, self.config_, self.bardist_ = initialize_tabpfn_model(
            model_path=self.model_path,
            which="regressor",
            fit_mode="batched",
        )

        # Get the device type and ensure it's a valid torch device
        if (self.device is None) or (isinstance(self.device, str) and self.device == "auto"):
            device_type_ = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            self.device_ = torch.device(device_type_)
        elif isinstance(self.device, str):
            self.device_ = torch.device(self.device)
        elif isinstance(self.device, torch.device):
            self.device_ = self.device
        else:
            raise ValueError(f"Invalid device: {self.device}")

        (
            self.use_autocast_,
            self.forced_inference_dtype_,
            byte_size,
        ) = determine_precision(
            self.inference_precision, self.device_
        )
        self.model_.to(self.device_)

        # Build the interface_config
        _config = ModelInterfaceConfig.from_user_input(
            inference_config=None,  # self.inference_config,
        )  # shorter alias

        self.interface_config_ = _config
        outlier_removal_std = _config._REGRESSION_DEFAULT_OUTLIER_REMOVAL_STD

        update_encoder_params(  # Use the renamed function if available, or original one
            model=self.model_,
            remove_outliers_std=outlier_removal_std,
            seed=static_seed,
            inplace=True,
            differentiable_input=False,
        )
        return byte_size, rng

    @config_context(transform_output="default")  # type: ignore
    def fit(self, X, y):
        """
        Feed background data to ExplainerPFN.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Background data for the explainer.
        y : array-like, shape (n_samples,)
            Target values for the background data. It may refer to labels,
            scores or ground truth values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        byte_size, rng = self._initialize_model_variables()
        ensemble_configs, X, y, self.bardist_ = self._initialize_dataset_preprocessing(
            X, y, rng
        )

        assert len(ensemble_configs) == self.n_features_in_

        self.X_ = X
        self.y_ = y

        # TODO: handle constant targets

        mean, std = np.mean(y), np.std(y)
        self.y_train_std_ = std.item() + 1e-20
        self.y_train_mean_ = mean.item()
        y = (y - self.y_train_mean_) / self.y_train_std_
        self.normalized_bardist_ = FullSupportBarDistribution(
            self.bardist_.borders  # * self.y_train_std_ + self.y_train_mean_,
        ).float()

        # Create the inference engine
        self.executor_ = InferenceEngineCachePreprocessing.prepare(
            X_train=X,
            y_train=y,
            cat_ix=self.inferred_categorical_indices_,
            ensemble_configs=ensemble_configs,
            n_workers=self.n_jobs,
            model=self.model_,
            rng=rng,
            dtype_byte_size=byte_size,
            force_inference_dtype=self.forced_inference_dtype_,
            save_peak_mem=self.memory_saving_mode,
            inference_mode=True,  # set to False if backprop is needed
        )

        return self

    def finetune(self, X, y, contributions):
        pass

    def _get_contributions_embeddings(
        self, X, y, feature_idx, only_return_standard_out=True
    ):
        """
        Estimate feature importance embeddings.

        NOTE: atm I'm assuming that the target is either a score or binary.
        NOTE #2: Rename to _forward?
        """

        # Prepare background and input data
        X_full = []
        y_full = []
        for X_, y_ in ([self.X_, self.y_], [X, y]):
            feature_values = X_[:, feature_idx]
            X_part = X_[:, np.arange(X_.shape[1]) != feature_idx]
            X_part = np.concatenate([feature_values.reshape(-1, 1), X_part], axis=1)
            y_full.append(y_.reshape(-1, 1))
            X_full.append(X_part)

        # Concatenate all data
        X_full = torch.tensor(
            np.expand_dims(np.concatenate(X_full, axis=0), 1), dtype=torch.float32
        )
        y_full = torch.tensor(np.concatenate(y_full, axis=0), dtype=torch.float32)

        with (
            torch.autocast("cpu", enabled=True),
            torch.inference_mode(True),
        ):
            embeddings = self.model_[0](
                *(X_full, y_full),
                only_return_standard_out=only_return_standard_out,
                categorical_inds=None,
                single_eval_pos=len(self.y_)
            )

        return embeddings

    def _get_feature_contributions(self, X, y, feature_idx):
        """
        Get the contributions of a specific feature.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,)
            Target values.
        feature_idx : int
            Index of the feature for which to get contributions.

        Returns
        -------
        contributions : array-like, shape (n_samples,)
            Contributions of the specified feature.
        """
        output = self._get_contributions_embeddings(
            X, y, feature_idx, only_return_standard_out=True
        )

        if self.softmax_temperature != 1:
            output = output.float() / self.softmax_temperature  # noqa: PLW2901

        return output
