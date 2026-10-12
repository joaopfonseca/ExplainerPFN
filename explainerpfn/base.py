import logging
import typing
from typing import Any, List, Literal, Union
from typing_extensions import TypeAlias, TypedDict
from collections.abc import Sequence
from pathlib import Path
from functools import partial
from dataclasses import asdict

import numpy as np
import torch
from torch.types import _dtype
from sklearn import config_context
from sklearn.base import TransformerMixin, check_is_fitted
from sklearn.pipeline import Pipeline


# from tabpfn import TabPFNRegressor
from tabpfn.base import check_cpu_warning, determine_precision
from explainerpfn.preprocessing import (
    ExplainerEnsembleConfig,
    ReshapeFeatureDistributionsStep,
    EnsembleConfig,
    PreprocessorConfig,
    default_explanation_preprocessor_configs,
)
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.utils import (
    validate_Xy_fit,
    # validate_X_predict,
    infer_categorical_features,
    update_encoder_params,
    _fix_dtypes,
    _get_ordinal_encoder,
    _process_text_na_dataframe,
    _transform_borders_one,
)
from explainerpfn.config import ModelInterfaceConfig

from explainerpfn.inference import InferenceEngineCachePreprocessing
from explainerpfn.model.attribution_head import (
    AttributionHead,
    bucket_means,
    downsample_borders,
)
from explainerpfn.model_loading import load_model_criterion_config
from explainerpfn.utils import (
    linear_correction,
    additive_correction,
    multiplicative_correction,
    statistical_correction,
)


logger = logging.getLogger(__name__)


XType: TypeAlias = Any
SampleWeightType: TypeAlias = Any
YType: TypeAlias = Any
# TODO: Define ArchitectureConfig
ArchitectureConfig: TypeAlias = (
    Any  # Placeholder for the actual architecture config type
)


class MainOutputDict(TypedDict):
    """Dictionary containing the main output types from the TabPFN regressor."""

    mean: np.ndarray
    median: np.ndarray
    mode: np.ndarray
    quantiles: list[np.ndarray]


class FullOutputDict(MainOutputDict):
    """Dictionary containing all outputs from the TabPFN regressor."""

    criterion: FullSupportBarDistribution
    logits: torch.Tensor


class BaseModelSpecs:
    """Base class for model specifications."""

    def __init__(self, model: torch.nn.Module, config: ArchitectureConfig):
        self.model = model
        self.config = config


class RegressorModelSpecs(BaseModelSpecs):
    """Model specs for regressors."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: ArchitectureConfig,
        norm_criterion: FullSupportBarDistribution,
    ):
        super().__init__(model, config)
        self.norm_criterion = norm_criterion


def initialize_explainerpfn_model(
    model_path: Union[str, Path, Literal["auto"], RegressorModelSpecs],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
) -> RegressorModelSpecs:
    """Initializes a TabPFN model based on the provided configuration.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model
            from. Trainer checkpoints written by
            ``scripts/train_explainerpfn.py`` (``checkpoint_*.pt``) are also
            accepted: only the model weights are used, with the bar
            distribution and architecture config taken from the foundation
            model the run was fine-tuned from. If the checkpoint carries a
            joint attribution head (top-level ``head_state_dict``/
            ``head_config``), it is loaded automatically during ``fit`` and
            ``predict`` then defaults to ``mode="joint"``.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.

    Returns:
        model: The loaded TabPFN model.
        config: The configuration object associated with the loaded model.
        bar_distribution: The BarDistribution for regression (`None` if classifier).
    """
    model, config, norm_criterion = None, None, None
    if isinstance(model_path, RegressorModelSpecs):
        model = model_path.model
        config = model_path.config
        norm_criterion = model_path.norm_criterion
    elif model_path is None or isinstance(model_path, (str, Path)):
        # (after processing 'auto')
        download = True
        if isinstance(model_path, str) and model_path == "auto":
            model_path = None  # type: ignore

        # Load model with potential caching
        # The regressor's bar distribution is required
        model, bardist, config = load_model_criterion_config(
            model_path=model_path,  # type: ignore
            check_bar_distribution_criterion=True,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="regressor",
            version="v2",
            download=download,
        )
        norm_criterion = bardist

    return model, config, norm_criterion


def _logits_to_output(
    *,
    output_type: str,
    logits: torch.Tensor,
    criterion: FullSupportBarDistribution,
    quantiles: list[float],
) -> np.ndarray | list[np.ndarray]:
    """Convert the logits to the specified output type.

    Args:
        output_type: The output type to convert the logits to.
        logits: The logits to convert.
        criterion: The criterion to use for the conversion.
        quantiles: The quantiles to use for the conversion.

    Returns:
        The converted logits or list of converted logits.
    """
    if output_type == "quantiles":
        return [criterion.icdf(logits, q).cpu().detach().numpy() for q in quantiles]

    if output_type == "mean":
        output = criterion.mean(logits)
    elif output_type == "median":
        output = criterion.median(logits)
    elif output_type == "mode":
        output = criterion.mode(logits)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    return output.cpu().detach().numpy()  # type: ignore


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat(logits.shape[:-1] + (1,))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    """Translate the probabilities across the borders.

    Args:
        logits: The logits defining the distribution to translate.
        frm: The borders to translate from.
        to: The borders to translate to.

    Returns:
        The translated probabilities.
    """
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0

    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


class ExplainerPFN:  # (TabPFNRegressor):
    """
    A class to handle the SHAP values for a PFN model.

    NOTE: This is intended to be a proof of concept. It's meant to handle only
    preprocessed data with standardized features and no missing values.
    """

    interface_config_: ModelInterfaceConfig

    def __init__(
        self,
        n_estimators: int = 1,
        categorical_features_indices: Sequence[int] | None = None,
        softmax_temperature=0.9,
        model_path="auto",
        device=None,
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
            "batched",
        ] = "fit_with_cache",
        inference_precision: _dtype | Literal["autocast", "auto"] = "auto",
        memory_saving_mode: bool | Literal["auto"] | float | int = "auto",
        random_state=None,
        n_jobs: int = -1,
        joint_head: bool = False,
        head_kwargs: dict | None = None,
    ):
        self.n_estimators = n_estimators
        self.categorical_features_indices = categorical_features_indices
        self.softmax_temperature = softmax_temperature
        self.model_path = model_path
        self.device = device
        self.fit_mode = fit_mode
        self.inference_precision = inference_precision
        self.memory_saving_mode = memory_saving_mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.joint_head = joint_head
        self.head_kwargs = dict(head_kwargs or {})
        # Populated by ``initialize_attribution_head`` / ``_initialize_model_variables``.
        self.head_ = None
        self.head_config_ = None
        self.head_borders_ = None
        self.head_centers_ = None
        self.head_criterion_ = None

    def _initialize_dataset_preprocessing(
        self, X: XType, y: YType, rng: np.random.Generator
    ) -> tuple[list[ExplainerEnsembleConfig], XType, YType, FullSupportBarDistribution]:
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
        check_cpu_warning(self.device, X, allow_cpu_override=True)

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
            # {"none": FunctionTransformer()}
        )
        target_preprocessors: list[TransformerMixin | Pipeline | None] = []
        for y_target_preprocessor in (
            # "none"
            # self.interface_config_.REGRESSION_Y_PREPROCESS_TRANSFORMS
            self.interface_config_.EXPLANATION_Y_PREPROCESS_TRANSFORMS
        ):
            if y_target_preprocessor is not None:
                preprocessor = possible_target_transforms[y_target_preprocessor]
            else:
                preprocessor = None
            target_preprocessors.append(preprocessor)

        # preprocess_transforms = self.interface_config_.PREPROCESS_TRANSFORMS

        ensemble_configs = EnsembleConfig.generate_for_explainability(
            n=self.n_estimators,  # refers to the number of estimators
            subsample_size=self.interface_config_.SUBSAMPLE_SAMPLES,
            add_fingerprint_feature=self.interface_config_.FINGERPRINT_FEATURE,
            feature_shift_decoder=self.interface_config_.FEATURE_SHIFT_METHOD,
            polynomial_features=self.interface_config_.POLYNOMIAL_FEATURES,
            max_index=len(X),
            preprocessor_configs=typing.cast(
                "Sequence[PreprocessorConfig]",
                default_explanation_preprocessor_configs(),
            ),
            target_transforms=target_preprocessors,
            random_state=rng,
        )

        self.bardist_ = self.bardist_.to(self.device_)

        assert len(ensemble_configs) == self.n_estimators

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

        self.model_, self.config_, self.bardist_ = initialize_explainerpfn_model(
            model_path=self.model_path,
            fit_mode=self.fit_mode,
        )

        # Get the device type and ensure it's a valid torch device
        if (self.device is None) or (
            isinstance(self.device, str) and self.device == "auto"
        ):
            device_type_ = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
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
        ) = determine_precision(self.inference_precision, self.device_)
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
        if self.head_ is None:
            loaded_head = False
            if isinstance(self.model_path, (str, Path)) and Path(
                self.model_path
            ).is_file():
                # A checkpoint may carry a joint attribution head (trainer
                # ``.pt`` files and ``*.ckp`` files saved with an embedded head
                # both store top-level ``head_state_dict``/``head_config``).
                # Load it from its saved config/weights; otherwise fall back to
                # a freshly initialized head when ``joint_head=True``.
                loaded_head = self.load_attribution_head(self.model_path)
                if loaded_head:
                    logger.info(
                        "Loaded attribution head from %s", self.model_path
                    )
            if not loaded_head and self.joint_head:
                self.initialize_attribution_head()
        return byte_size, rng

    # ------------------------------------------------------------------
    # Joint attribution head
    # ------------------------------------------------------------------

    def initialize_attribution_head(self, head_config: dict | None = None):
        """Construct the joint :class:`AttributionHead` (and its criterion).

        Args:
            head_config: Optional constructor kwargs; merged over
                ``self.head_kwargs``. ``emb_dim`` defaults to the backbone
                embedding size.
        """
        kwargs = dict(self.head_kwargs)
        if head_config is not None:
            kwargs = {**kwargs, **head_config}
        kwargs.setdefault("emb_dim", self.config_.emsize)
        kwargs.setdefault("n_buckets", 256)
        kwargs.setdefault("max_features", 15)
        self.head_ = AttributionHead(**kwargs)
        self.head_config_ = self.head_.config()

        borders = downsample_borders(
            self.bardist_.borders, self.head_.n_buckets
        ).to(torch.float32)
        self.head_borders_ = borders
        self.head_centers_ = (borders[:-1] + borders[1:]) / 2.0
        self.head_criterion_ = FullSupportBarDistribution(borders)
        self.head_.to(self.device_)
        return self.head_

    def load_attribution_head(self, checkpoint_path) -> bool:
        """Load the head from a trainer checkpoint (``head_state_dict``).

        Returns:
            True if a head was found and loaded, False otherwise.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, dict) or "head_state_dict" not in checkpoint:
            return False
        self.initialize_attribution_head(checkpoint.get("head_config"))
        self.head_.load_state_dict(checkpoint["head_state_dict"])
        self.head_.to(self.device_)
        return True

    def head_state_dict(self):
        """Backbone-independent head state for checkpointing."""
        if self.head_ is None:
            return None
        return self.head_.state_dict()

    def _collect_feature_embeddings(self, X, y) -> torch.Tensor:
        """Per-feature backbone ``test_embeddings`` as ``(n, d, emb)``."""
        embeddings = []
        for feature_idx in range(self.n_features_in_):
            output = self.forward(
                X, y, feature_idx=feature_idx, only_return_standard_out=False
            )
            test_embeddings = output["test_embeddings"]
            if test_embeddings.ndim == 3:
                test_embeddings = test_embeddings[:, 0, :]
            embeddings.append(test_embeddings)
        return torch.stack(embeddings, dim=1)

    def _predict_joint(
        self,
        X,
        y,
        output_type: str = "mean",
        quantiles: list[float] | None = None,
    ):
        """All-features attribution predictions from the joint head."""
        if self.head_ is None:
            raise ValueError(
                "predict(mode='joint') requires an AttributionHead; load a "
                "checkpoint containing 'head_state_dict' (via model_path) or "
                "construct the estimator with joint_head=True."
            )
        if self.n_estimators != 1:
            raise ValueError(
                "joint mode supports a single estimator (n_estimators=1); got "
                f"{self.n_estimators}."
            )
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        embeddings = self._collect_feature_embeddings(X, y)
        features = np.asarray(
            self._prepare_predict_input(X), dtype=np.float32
        )
        score = np.asarray(y, dtype=np.float32).reshape(-1)
        features_t = torch.as_tensor(
            features, dtype=torch.float32, device=self.device_
        )
        score_t = torch.as_tensor(
            score, dtype=torch.float32, device=self.device_
        )
        with torch.inference_mode():
            logits = self.head_(embeddings, features_t, score_t)  # (n, d, K)
        n_samples, n_features, n_buckets = logits.shape

        scale = self.feature_exp_std_
        centers = self.head_centers_.to(logits.device)
        flat = logits.reshape(n_samples * n_features, n_buckets)

        if output_type == "quantiles":
            return [
                (
                    self.head_criterion_.icdf(flat.cpu(), q)
                    .reshape(n_samples, n_features)
                    .numpy()
                    * scale
                )
                for q in quantiles
            ]

        normalized = bucket_means(flat, centers).reshape(
            n_samples, n_features
        )
        raw = normalized.cpu().numpy() * scale
        return raw

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

        self.X_ = X
        self.y_ = y

        # TODO: handle constant targets

        # TODO: Check if there's a way to improve this
        self.y_train_std_ = np.std(y) + 1e-20
        self.feature_exp_std_ = self.y_train_std_  # / np.sqrt(X.shape[1])
        self.base_value_ = np.mean(y).item()
        # mean, std = np.mean(y), np.std(y)
        # self.y_train_std_ = std.item() + 1e-20
        # self.y_train_mean_ = mean.item()
        # y = (y - self.y_train_mean_) / self.y_train_std_
        self.normalized_bardist_ = FullSupportBarDistribution(
            self.bardist_.borders * self.feature_exp_std_  # + self.y_train_mean_,
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

    def forward(self, X, y, feature_idx, only_return_standard_out=True):
        """
        Estimate feature importance embeddings. Used internally or for model tuning.

        NOTE: atm I'm assuming that the target is either a score or binary.
        """
        check_is_fitted(self)

        std_borders = self.bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        borders: list[np.ndarray] = []

        # Apply the same dtype/text handling used during ``fit``. This is the
        # single funnel for ``predict``, ``predict_feature`` and ``finetune``.
        X = self._prepare_predict_input(X)

        # Iterate over estimators
        for output, config in self.executor_[feature_idx].iter_outputs(
            X,
            y,
            #  base_value=self.base_value_,
            device=self.device_,
            autocast=self.use_autocast_,
            only_return_standard_out=only_return_standard_out,
        ):
            if not only_return_standard_out:
                return output

            if self.softmax_temperature != 1:
                output = output.float() / self.softmax_temperature

            # BSz.= 1 Scenario, the same as normal predict() function
            # Handled by first if-statement
            config_for_ensemble = config
            if isinstance(config, list) and len(config) == 1:
                single_config = config[0]
                config_for_ensemble = single_config

            if isinstance(config_for_ensemble, ExplainerEnsembleConfig):
                borders_t: np.ndarray
                logit_cancel_mask: np.ndarray | None
                descending_borders: bool

                if config_for_ensemble.target_transform is None:
                    borders_t = std_borders.copy()
                    logit_cancel_mask = None
                    descending_borders = False
                else:
                    logit_cancel_mask, descending_borders, borders_t = (
                        _transform_borders_one(
                            std_borders,
                            target_transform=config_for_ensemble.target_transform,
                            repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                        )
                    )
                    if descending_borders:
                        borders_t = borders_t.flip(-1)  # type: ignore

                borders.append(borders_t)

                if logit_cancel_mask is not None:
                    output = output.clone()  # noqa: PLW2901
                    output[..., logit_cancel_mask] = float("-inf")

            else:
                raise ValueError(
                    "Unexpected config format "
                    "and Batch prediction is not supported yet!"
                )

            outputs.append(output)  # type: ignore

        averaged_logits = None
        all_logits = None

        if outputs:
            all_logits = torch.stack(outputs, dim=0)  # [N_est, N_sampls, N_bord]
            averaged_logits_over_ensemble = torch.mean(
                all_logits, dim=0
            )  # [N_sampls, N_bord]
            averaged_logits = averaged_logits_over_ensemble.transpose(0, 1)

        # TODO: Modify borders definition
        return averaged_logits, outputs, borders

    def forward_with_embeddings(self, X, y, feature_idx):
        """Like :meth:`forward`, but also return the per-query embeddings.

        Returns the same averaged logits as ``forward`` (so the per-feature
        training objective is unchanged) *plus* the backbone
        ``test_embeddings`` for the query rows, which feed the joint head.

        Returns:
            ``(averaged_logits, embeddings)`` where ``averaged_logits`` has the
            same shape convention as :meth:`forward` and ``embeddings`` is
            ``(n_samples, emb_dim)``.
        """
        check_is_fitted(self)

        std_borders = self.bardist_.borders.cpu().numpy()
        outputs: list[torch.Tensor] = []
        embeddings: list[torch.Tensor] = []

        X = self._prepare_predict_input(X)

        for output, config in self.executor_[feature_idx].iter_outputs(
            X,
            y,
            device=self.device_,
            autocast=self.use_autocast_,
            only_return_standard_out=False,
        ):
            test_embeddings = output["test_embeddings"]
            if test_embeddings.ndim == 3:
                test_embeddings = test_embeddings[:, 0, :]
            embeddings.append(test_embeddings)

            logits = output["standard"]
            if logits.ndim == 3:
                logits = logits.squeeze(1)

            if self.softmax_temperature != 1:
                logits = logits.float() / self.softmax_temperature

            config_for_ensemble = config
            if isinstance(config, list) and len(config) == 1:
                config_for_ensemble = config[0]

            if not isinstance(config_for_ensemble, ExplainerEnsembleConfig):
                raise ValueError(
                    "Unexpected config format "
                    "and Batch prediction is not supported yet!"
                )

            if config_for_ensemble.target_transform is not None:
                logit_cancel_mask, _, _ = _transform_borders_one(
                    std_borders,
                    target_transform=config_for_ensemble.target_transform,
                    repair_nan_borders_after_transform=self.interface_config_.FIX_NAN_BORDERS_AFTER_TARGET_TRANSFORM,
                )
                if logit_cancel_mask is not None:
                    logits = logits.clone()
                    logits[..., logit_cancel_mask] = float("-inf")

            outputs.append(logits)

        all_logits = torch.stack(outputs, dim=0)  # [N_est, N_samples, N_bord]
        averaged_logits = torch.mean(all_logits, dim=0).transpose(0, 1)
        stacked_embeddings = torch.stack(embeddings, dim=0).mean(dim=0)
        return averaged_logits, stacked_embeddings

    def predict_feature(
        self,
        X,
        y,
        feature_idx: int,
        output_type: Literal[
            "mean",
            "median",
            "mode",
            "quantiles",
            "full",
            "main",
        ] = "mean",
        quantiles: list[float] | None = None,
    ):
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
        check_is_fitted(self)

        # TODO: Check what quantiles are for
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Forward pass
        (
            _,  # averaged logits (not used here)
            outputs,  # list of tensors [N_est, N_samples, N_borders] (after forward)
            borders,  # list of numpy arrays containing borders for each estimator
        ) = self.forward(X, y, feature_idx=feature_idx, only_return_standard_out=True)

        # Process outputs into probabilities, expected value and get final logits
        transformed_logits = [
            translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t, device=self.device_),
                to=self.normalized_bardist_.borders.to(self.device_),
            )
            for logits, borders_t in zip(outputs, borders)
        ]
        stacked_logits = torch.stack(transformed_logits, dim=0)
        logits = stacked_logits.mean(dim=0)

        # Post-process the logits
        logits = logits.log()
        if logits.dtype == torch.float16:
            logits = logits.float()
        logits = logits.cpu()

        # Determine and return intended output type
        logit_to_output = partial(
            _logits_to_output,
            logits=logits,
            criterion=self.normalized_bardist_,
            quantiles=quantiles,
        )
        if output_type in ["full", "main"]:
            # Create a dictionary of outputs with proper typing via TypedDict
            # Get individual outputs with proper typing
            mean_out = typing.cast("np.ndarray", logit_to_output(output_type="mean"))
            median_out = typing.cast(
                "np.ndarray", logit_to_output(output_type="median")
            )
            mode_out = typing.cast("np.ndarray", logit_to_output(output_type="mode"))
            quantiles_out = typing.cast(
                "list[np.ndarray]",
                logit_to_output(output_type="quantiles"),
            )

            # Create our typed dictionary
            main_outputs = MainOutputDict(
                mean=mean_out,
                median=median_out,
                mode=mode_out,
                quantiles=quantiles_out,
            )

            if output_type == "full":
                # Return full output with criterion and logits
                return FullOutputDict(
                    **main_outputs,
                    criterion=self.normalized_bardist_,
                    logits=logits,
                )

            return main_outputs

        return logit_to_output(output_type=output_type)

    def _prepare_predict_input(self, X: XType) -> np.ndarray:
        """Apply the same dtype and text/NA handling used during ``fit``.

        Fitting ordinal-encodes text/categorical columns. The same encoding
        must be applied to the data passed to ``predict``/``get_embeddings``/
        ``finetune``, otherwise the model receives raw strings or mismatched
        dtypes.

        When this instance was not fitted itself but borrows pre-fitted
        ``executor_`` objects (the notebook 6 training-loop pattern, where the
        executors come from a throwaway ``ExplainerPFN.fit`` call), the
        fit-time encoding attributes are absent. The executors were built from
        already-encoded data, so the numeric query features only need a dtype
        coercion.
        """
        if not hasattr(self, "inferred_categorical_indices_"):
            return np.asarray(X, dtype=np.float64)
        X = _fix_dtypes(X, cat_indices=self.inferred_categorical_indices_)
        X = _process_text_na_dataframe(X, ord_encoder=self.preprocessor_)
        return X

    def predict(
        self,
        X: XType,
        y: YType,
        output_type: Literal[
            "mean",
            "median",
            "mode",
            "quantiles",
            "full",
            "main",
        ] = "mean",
        quantiles: list[float] | None = None,
        mode: Literal["per-feature", "joint"] | None = None,
    ):
        check_is_fitted(self)

        if mode is None:
            # A loaded/constructed joint head makes joint the natural default;
            # estimators without a head keep the historical per-feature output.
            mode = "joint" if self.head_ is not None else "per-feature"

        if mode == "joint":
            return self._predict_joint(
                X, y, output_type=output_type, quantiles=quantiles
            )
        if mode != "per-feature":
            raise ValueError(f"Invalid predict mode: {mode}")

        # Check whether model is in inference mode
        for executor in self.executor_:
            if not executor.inference_mode:
                executor.use_torch_inference_mode(True)
                executor.model.eval()

        self.model_.eval()

        if output_type in ["mean", "median", "mode"]:
            explanations = np.zeros(X.shape)
            for feature_idx in range(X.shape[1]):
                explanations[:, feature_idx] = self.predict_feature(
                    X,
                    y,
                    feature_idx=feature_idx,
                    output_type=output_type,
                    quantiles=quantiles,
                )
        elif output_type == "quantiles":
            # ``predict_feature`` returns a list with one array per quantile.
            # Stack across features so each quantile is an
            # ``(n_samples, n_features)`` array.
            per_feature = [
                self.predict_feature(
                    X,
                    y,
                    feature_idx=feature_idx,
                    output_type=output_type,
                    quantiles=quantiles,
                )
                for feature_idx in range(X.shape[1])
            ]
            n_quantiles = len(per_feature[0])
            explanations = [
                np.column_stack([per_feature[f][q] for f in range(X.shape[1])])
                for q in range(n_quantiles)
            ]
        elif output_type in ["full", "main"]:
            explanations = []
            for feature_idx in range(X.shape[1]):
                explanations.append(
                    self.predict_feature(
                        X,
                        y,
                        feature_idx=feature_idx,
                        output_type=output_type,
                        quantiles=quantiles,
                    )
                )
        else:
            raise ValueError(f"Invalid output type: {output_type}")

        return explanations

    def get_embeddings(self, X, y, feature_idx: int):
        check_is_fitted(self)
        X = self._prepare_predict_input(X)

        # Check whether model is in inference mode
        if not self.executor_[feature_idx].inference_mode:
            self.executor_[feature_idx].use_torch_inference_mode(True)
            self.model_.eval()

        embeddings = []
        for output, _config in self.executor_[feature_idx].iter_outputs(
            X,
            y,
            device=self.device_,
            autocast=self.use_autocast_,
            only_return_standard_out=False,
        ):
            embeddings.append(output["test_embeddings"])
        # ``[n_estimators, n_samples, embedding_dim]``
        return torch.stack(embeddings, dim=0)

    def save_foundation_model(self, path: Union[str, Path]):
        """
        Save the underlying model to the specified path.

        This writes the base pre-trained weights and configuration. It does
        **not** store a fitted :class:`TabPFNRegressor`/``Classifier`` instance.
        The resulting file is merely a checkpoint consumed by
        :func:`load_model_criterion_config` to build a new estimator.

        When a joint attribution head is present on this estimator, its weights
        and config are embedded as additive ``head_state_dict``/``head_config``
        keys. These are ignored by older loaders, and are picked up
        automatically by :meth:`load_attribution_head` (and hence by
        ``ExplainerPFN(model_path=...)``) so the file is self-contained.

        Args:
            model:
                The internal model object of a ``TabPFN`` estimator.
            save_path:
                Path to save the checkpoint to.
        """
        # Clear trainset representation cache if it exists
        self.model_.empty_trainset_representation_cache()

        # Get model state dict
        model_state = self.model_.state_dict()

        # Get bardist state dict and prefix with 'criterion.'
        if hasattr(self.model_, "bardist_") and self.model_.bardist_ is not None:
            bardist_state = {
                f"criterion.{k}": v
                for k, v in self.model_.bardist_.state_dict().items()
            }
            # Combine model and bardist states
            state_dict = {**model_state, **bardist_state}
        elif hasattr(self, "bardist_") and self.bardist_ is not None:
            bardist_state = {
                f"criterion.{k}": v for k, v in self.bardist_.state_dict().items()
            }
            # Combine model and bardist states
            state_dict = {**model_state, **bardist_state}
        else:
            state_dict = model_state

        # Create checkpoint with correct structure
        checkpoint = {"state_dict": state_dict, "config": asdict(self.config_)}

        # Optionally embed the joint attribution head. These keys are additive:
        # ``load_model`` only requires ``state_dict``/``config`` and ignores
        # unknown keys, so foundation checkpoints stay backward-compatible.
        if self.head_ is not None:
            checkpoint["head_state_dict"] = self.head_state_dict()
            checkpoint["head_config"] = self.head_config_

        # Save the checkpoint
        torch.save(checkpoint, path)

        return self

    def finetune(self, optimizer, X, y, shapley_values):
        check_is_fitted(self)

        for executor in self.executor_:
            if executor.inference_mode:
                # Set model to training mode and disable inference mode to allow backprop
                executor.use_torch_inference_mode(False)
                executor.model.train()

        self.model_.train()

        # A previous inference pass may have enabled memory-saving chunking,
        # which is incompatible with backprop. Reset it before training.
        self.model_.reset_save_peak_mem_factor(None)
        # The trainset KV cache is an inference-only optimisation; disable it
        # so gradients can flow through the transformer during fine-tuning.
        self.model_.cache_trainset_representation = False

        # Zero gradients
        optimizer.zero_grad()

        # Get logits
        with torch.enable_grad():
            # self.forward returns a tuple (first element are averaged logits across estimators)
            all_features_logits = [
                self.forward(
                    X, y, feature_idx=feature_idx, only_return_standard_out=True
                )[0]
                for feature_idx in range(X.shape[1])
            ]

        logits = torch.concat(all_features_logits, axis=1)

        # Convert SHAP values to tensor
        shap_tensor = torch.tensor(shapley_values, dtype=torch.float32).to(self.device_)
        shap_tensor = torch.concat(
            [shap_tensor[:, i] for i in range(shap_tensor.shape[1])], dim=0
        )

        # Compute the loss
        loss_per_sample = self.bardist_(logits.T, shap_tensor)

        # Compute mean loss across all test samples
        loss = loss_per_sample.mean()

        # Backpropagate and update weights
        loss.backward()
        optimizer.step()

        return loss.item()

    def apply_correction(
        self,
        y: np.ndarray,
        explanations: np.ndarray,
        kind: Union[List[str], str] = "statistical",
    ) -> np.ndarray:
        """
        Apply mean/std correction to the explanations.

        Parameters
        ----------
        explanations : array-like, shape (n_samples, n_features)
            Raw explanations to be corrected.

        kind : str or list of str, optional
            Type of correction to apply. Options are:
            - "statistical": Adjust explanations to match the expected mean
              and standard deviation of standard Shapley values.
            - "multiplicative": Scale explanations multiplicatively based on
              the target variable.
            - "additive": Adjust explanations additively based on the target
              variable.
            - "linear": Apply a linear correction based on the target variable.
            Can also be a list of these strings to apply multiple corrections
            sequentially. Default is "statistical".

        Returns
        -------
        corrected_explanations : array-like, shape (n_samples, n_features)
            Corrected explanations.
        """
        explanations = explanations.copy()
        if isinstance(kind, list):
            for k in kind:
                explanations = self.apply_correction(
                    y,
                    explanations,
                    kind=k,
                )
            return explanations

        if kind == "statistical":
            corrected_explanations = statistical_correction(
                explanations,
                y,
            )
        elif kind == "multiplicative":
            corrected_explanations = multiplicative_correction(
                explanations,
                y,
                base_value=self.base_value_,
            )
        elif kind == "additive":
            corrected_explanations = additive_correction(
                explanations,
                y,
                base_value=self.base_value_,
            )
        elif kind == "linear":
            corrected_explanations = linear_correction(
                explanations,
                y,
                base_value=self.base_value_,
            )
        else:
            raise NotImplementedError(
                f"Explanation correction of kind '{kind}' is not implemented "
                "yet."
            )

        return corrected_explanations
