"""Registry of models and imbalance strategies, and the pipeline builder.

Every experiment is an ``imblearn`` pipeline::

    features -> scaling (if needed) -> sampler (if any) -> model

Scaling and resampling are fitted inside the pipeline, on whatever data ``fit`` receives.
During ``predict`` the sampler is skipped by imblearn, so validation and test data are never
resampled, and inside cross-validation each fold is resampled on its own training part only.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fraud.config import RANDOM_SEED
from fraud.features import FraudFeatures

# ``sampling_strategy`` as imblearn understands it: a minority:majority ratio after
# resampling, or "auto" for full balancing (ratio 1.0).
SamplingStrategy = float | str


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model and how it expresses class weighting."""

    factory: Callable[[int], BaseEstimator]
    needs_scaling: bool
    class_weight_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategySpec:
    """How an imbalance strategy changes the pipeline."""

    sampler_factory: Callable[[SamplingStrategy, int], BaseEstimator] | None = None
    uses_class_weight: bool = False
    # Samplers that interpolate between nearest neighbours need features on a common scale.
    distance_based: bool = False


def _logreg(seed: int) -> BaseEstimator:
    return LogisticRegression(max_iter=1000, random_state=seed)


def _random_forest(seed: int) -> BaseEstimator:
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)


def _lightgbm(seed: int) -> BaseEstimator:
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )


# class_weight="balanced" weights each class by n_samples / (2 * n_class), computed from the
# data passed to fit. For LightGBM this is the same weighting as scale_pos_weight = neg / pos,
# but derived inside each fit, so it never uses class counts from outside the training fold.
MODELS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        _logreg, needs_scaling=True, class_weight_params={"class_weight": "balanced"}
    ),
    "random_forest": ModelSpec(
        _random_forest,
        needs_scaling=False,
        class_weight_params={"class_weight": "balanced_subsample"},
    ),
    "lightgbm": ModelSpec(
        _lightgbm, needs_scaling=False, class_weight_params={"class_weight": "balanced"}
    ),
}


def _undersampler(ratio: SamplingStrategy, seed: int) -> BaseEstimator:
    return RandomUnderSampler(sampling_strategy=ratio, random_state=seed)


def _smote(ratio: SamplingStrategy, seed: int) -> BaseEstimator:
    return SMOTE(sampling_strategy=ratio, random_state=seed)


def _borderline_smote(ratio: SamplingStrategy, seed: int) -> BaseEstimator:
    return BorderlineSMOTE(sampling_strategy=ratio, random_state=seed)


def _adasyn(ratio: SamplingStrategy, seed: int) -> BaseEstimator:
    return ADASYN(sampling_strategy=ratio, random_state=seed)


def _smote_tomek(ratio: SamplingStrategy, seed: int) -> BaseEstimator:
    # Tomek-link removal searches neighbours over the whole resampled set; use all cores.
    return SMOTETomek(
        sampling_strategy=ratio,
        smote=_smote(ratio, seed),
        tomek=TomekLinks(n_jobs=-1),
        random_state=seed,
    )


STRATEGIES: dict[str, StrategySpec] = {
    "none": StrategySpec(),
    "class_weight": StrategySpec(uses_class_weight=True),
    "undersample": StrategySpec(_undersampler),
    "smote": StrategySpec(_smote, distance_based=True),
    "borderline_smote": StrategySpec(_borderline_smote, distance_based=True),
    "adasyn": StrategySpec(_adasyn, distance_based=True),
    "smote_tomek": StrategySpec(_smote_tomek, distance_based=True),
}


def _lookup(registry: dict[str, Any], name: str, kind: str) -> Any:
    if name not in registry:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {sorted(registry)}")
    return registry[name]


def build_pipeline(
    model: str,
    strategy: str = "none",
    sampling_ratio: float | None = None,
    model_params: dict[str, Any] | None = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build the pipeline for one model / imbalance strategy combination.

    ``sampling_ratio`` is the minority:majority ratio after resampling (e.g. 0.1 or 0.5);
    ``None`` means full balancing. It is only valid for strategies that resample.

    Scaling is added when the model needs it (logistic regression) or when the sampler is
    distance-based (the SMOTE family picks nearest neighbours, so unscaled features would
    let the widest ones dominate). Tree models are unaffected by the extra scaling.
    """
    model_spec: ModelSpec = _lookup(MODELS, model, "model")
    strategy_spec: StrategySpec = _lookup(STRATEGIES, strategy, "strategy")
    if strategy_spec.sampler_factory is None and sampling_ratio is not None:
        raise ValueError(f"Strategy '{strategy}' does not resample; sampling_ratio must be None.")
    if sampling_ratio is not None and not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio must be in (0, 1].")

    estimator = model_spec.factory(random_state)
    if strategy_spec.uses_class_weight:
        estimator.set_params(**model_spec.class_weight_params)
    if model_params:
        estimator.set_params(**model_params)

    steps: list[tuple[str, BaseEstimator]] = [("features", FraudFeatures())]
    if model_spec.needs_scaling or strategy_spec.distance_based:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    if strategy_spec.sampler_factory is not None:
        ratio: SamplingStrategy = "auto" if sampling_ratio is None else sampling_ratio
        steps.append(("sampler", strategy_spec.sampler_factory(ratio, random_state)))
    steps.append(("model", estimator))
    return Pipeline(steps)
