"""Registry of classifiers and imbalance strategies, and the pipeline builder.

Every experiment is an ``imblearn`` pipeline::

    features -> scaling (if needed) -> sampler (if any) -> model

Scaling and resampling are fitted inside the pipeline, on whatever data ``fit`` receives.
During ``predict`` the sampler is skipped by imblearn, so validation and test data are never
resampled, and inside cross-validation each fold is resampled on its own training part only.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraud.config import RANDOM_SEED
from fraud.features import FraudFeatures
from fraud.samplers import PolynomFitSMOTE, SafeLevelSMOTE

# Neighbours used by every neighbour-based oversampler (PLAN.md section 6).
K_NEIGHBORS: int = 5


class AutoWeightXGBClassifier(XGBClassifier):
    """XGBoost with ``scale_pos_weight = negatives / positives`` computed in every ``fit``.

    XGBoost has no ``class_weight="balanced"``. Computing the weight from the data passed to
    ``fit`` gives the same behaviour: inside cross-validation it only ever uses the class
    counts of the fold's training part.
    """

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "AutoWeightXGBClassifier":
        target = np.asarray(y)
        positives = max(int((target == 1).sum()), 1)
        self.set_params(scale_pos_weight=float((target == 0).sum() / positives))
        return super().fit(X, y, **kwargs)


@dataclass(frozen=True)
class ModelSpec:
    """How to build a classifier and how it expresses class weighting."""

    factory: Callable[[int], BaseEstimator]
    needs_scaling: bool
    # Parameters that switch on class weighting; ``None`` if the classifier cannot weight.
    class_weight_params: dict[str, Any] | None = None
    # A different class used for the class-weight strategy (XGBoost).
    weighted_factory: Callable[[int], BaseEstimator] | None = None

    @property
    def supports_class_weight(self) -> bool:
        return self.class_weight_params is not None or self.weighted_factory is not None


@dataclass(frozen=True)
class StrategySpec:
    """How an imbalance strategy changes the pipeline."""

    sampler_factory: Callable[[float, int], BaseEstimator] | None = None
    uses_class_weight: bool = False


def _xgboost_params(seed: int) -> dict[str, Any]:
    return {"random_state": seed, "n_jobs": -1, "tree_method": "hist"}


# Library defaults everywhere, except: fixed seeds, all cores, quiet output, enough
# iterations for logistic regression to converge, and early stopping for the MLP.
MODELS: dict[str, ModelSpec] = {
    "logreg": ModelSpec(
        lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
        needs_scaling=True,
        class_weight_params={"class_weight": "balanced"},
    ),
    "decision_tree": ModelSpec(
        lambda seed: DecisionTreeClassifier(random_state=seed),
        needs_scaling=False,
        class_weight_params={"class_weight": "balanced"},
    ),
    "gaussian_nb": ModelSpec(lambda seed: GaussianNB(), needs_scaling=False),
    "knn": ModelSpec(lambda seed: KNeighborsClassifier(n_jobs=-1), needs_scaling=True),
    "random_forest": ModelSpec(
        lambda seed: RandomForestClassifier(n_jobs=-1, random_state=seed),
        needs_scaling=False,
        class_weight_params={"class_weight": "balanced_subsample"},
    ),
    "extra_trees": ModelSpec(
        lambda seed: ExtraTreesClassifier(n_jobs=-1, random_state=seed),
        needs_scaling=False,
        class_weight_params={"class_weight": "balanced"},
    ),
    "xgboost": ModelSpec(
        lambda seed: XGBClassifier(**_xgboost_params(seed)),
        needs_scaling=False,
        weighted_factory=lambda seed: AutoWeightXGBClassifier(**_xgboost_params(seed)),
    ),
    # class_weight="balanced" is LightGBM's equivalent of scale_pos_weight = neg / pos,
    # computed from the data passed to fit.
    "lightgbm": ModelSpec(
        lambda seed: LGBMClassifier(
            random_state=seed, deterministic=True, force_row_wise=True, verbose=-1
        ),
        needs_scaling=False,
        class_weight_params={"class_weight": "balanced"},
    ),
    "mlp": ModelSpec(
        lambda seed: MLPClassifier(early_stopping=True, random_state=seed),
        needs_scaling=True,
    ),
}


def _smote(ratio: float, seed: int) -> SMOTE:
    return SMOTE(sampling_strategy=ratio, k_neighbors=K_NEIGHBORS, random_state=seed)


def _smote_tomek(ratio: float, seed: int) -> BaseEstimator:
    # Both rows of every Tomek link are removed (the original SMOTE-Tomek definition).
    # The search runs over the whole resampled set, so it uses all cores.
    return SMOTETomek(
        sampling_strategy=ratio,
        smote=_smote(ratio, seed),
        tomek=TomekLinks(sampling_strategy="all", n_jobs=-1),
        random_state=seed,
    )


def _smote_enn(ratio: float, seed: int) -> BaseEstimator:
    # Standard edited-nearest-neighbours rule: 3 neighbours, applied to both classes.
    return SMOTEENN(
        sampling_strategy=ratio,
        smote=_smote(ratio, seed),
        enn=EditedNearestNeighbours(sampling_strategy="all", n_neighbors=3, n_jobs=-1),
        random_state=seed,
    )


def _safe_level_smote(ratio: float, seed: int) -> BaseEstimator:
    return SafeLevelSMOTE(sampling_ratio=ratio, k_neighbors=K_NEIGHBORS, random_state=seed)


def _polynom_fit_smote(ratio: float, seed: int) -> BaseEstimator:
    return PolynomFitSMOTE(sampling_ratio=ratio, random_state=seed)


STRATEGIES: dict[str, StrategySpec] = {
    "none": StrategySpec(),
    "class_weight": StrategySpec(uses_class_weight=True),
    "smote": StrategySpec(_smote),
    "smote_tomek": StrategySpec(_smote_tomek),
    "smote_enn": StrategySpec(_smote_enn),
    "safe_level_smote": StrategySpec(_safe_level_smote),
    "polynom_fit_smote": StrategySpec(_polynom_fit_smote),
}


def _lookup(registry: dict[str, Any], name: str, kind: str) -> Any:
    if name not in registry:
        raise ValueError(f"Unknown {kind} '{name}'. Available: {sorted(registry)}")
    return registry[name]


def is_supported(model: str, strategy: str) -> bool:
    """False only for class weights on a classifier that cannot weight classes."""
    model_spec: ModelSpec = _lookup(MODELS, model, "model")
    strategy_spec: StrategySpec = _lookup(STRATEGIES, strategy, "strategy")
    return model_spec.supports_class_weight or not strategy_spec.uses_class_weight


def build_pipeline(
    model: str,
    strategy: str = "none",
    sampling_ratio: float | None = None,
    model_params: dict[str, Any] | None = None,
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Build the pipeline for one classifier / imbalance strategy combination.

    ``sampling_ratio`` is the minority:majority ratio after resampling (e.g. 0.1 or 0.5);
    ``None`` means full balancing (1.0). It is only valid for strategies that resample.

    Scaling is added when the classifier needs it or when the pipeline resamples: the
    samplers interpolate between rows and most of them search nearest neighbours, so the
    features must be on a common scale. Tree models are unaffected by the extra scaling, and
    it means every classifier is trained on the same resampled data.
    """
    model_spec: ModelSpec = _lookup(MODELS, model, "model")
    strategy_spec: StrategySpec = _lookup(STRATEGIES, strategy, "strategy")
    if not is_supported(model, strategy):
        raise ValueError(f"Model '{model}' does not support class weights.")
    if strategy_spec.sampler_factory is None and sampling_ratio is not None:
        raise ValueError(f"Strategy '{strategy}' does not resample; sampling_ratio must be None.")
    if sampling_ratio is not None and not 0 < sampling_ratio <= 1:
        raise ValueError("sampling_ratio must be in (0, 1].")

    weighted = strategy_spec.uses_class_weight
    factory = model_spec.weighted_factory if weighted and model_spec.weighted_factory else None
    estimator = (factory or model_spec.factory)(random_state)
    if weighted and model_spec.class_weight_params:
        estimator.set_params(**model_spec.class_weight_params)
    if model_params:
        estimator.set_params(**model_params)

    steps: list[tuple[str, BaseEstimator]] = [("features", FraudFeatures())]
    if model_spec.needs_scaling or strategy_spec.sampler_factory is not None:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    if strategy_spec.sampler_factory is not None:
        ratio = 1.0 if sampling_ratio is None else sampling_ratio
        steps.append(("sampler", strategy_spec.sampler_factory(ratio, random_state)))
    steps.append(("model", estimator))
    return Pipeline(steps)
