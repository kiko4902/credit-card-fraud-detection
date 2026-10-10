"""Fit the preprocessing steps of a pipeline, with an optional cache for resampled data.

Resampling does not depend on the classifier: for a given training part, sampler, ratio and
seed, every classifier is trained on the same resampled rows. Computing that once and
reusing it saves most of the runtime of the SMOTE-family experiments.

The cache key is a hash of the preprocessing steps (class and all parameters, including the
seed) **and of the training data itself**. A fold can therefore only ever get back the
result that was computed from exactly its own training rows, and the scaler in front of the
sampler was fitted on those rows only. Nothing about the validation part enters the key or
the cached value.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone

Steps = list[tuple[str, BaseEstimator]]
# Part of every cache key. Increase it when the implementation of a sampler changes without
# its parameters changing, so that results of the old implementation are never reused.
# 2: Safe-Level-SMOTE and Polynom-Fit SMOTE implemented in fraud.samplers.
CACHE_VERSION: int = 2


@dataclass(frozen=True)
class Prepared:
    """Fitted preprocessing steps and the data the classifier is trained on."""

    steps: Steps
    X: pd.DataFrame
    y: pd.Series
    seconds: float


def is_sampler(step: BaseEstimator) -> bool:
    return hasattr(step, "fit_resample")


def fit_steps(steps: Steps, X: pd.DataFrame, y: pd.Series) -> Prepared:
    """Fit fresh clones of ``steps`` in order, as ``imblearn.pipeline.Pipeline.fit`` does."""
    start = time.perf_counter()
    fitted: Steps = []
    for name, step in steps:
        step = clone(step)
        if is_sampler(step):
            X, y = step.fit_resample(X, y)
        else:
            X = step.fit_transform(X, y)
        fitted.append((name, step))
    return Prepared(fitted, X, y, time.perf_counter() - start)


class ResampleCache:
    """Disk cache of fitted preprocessing steps and resampled training data."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(steps: Steps, X: pd.DataFrame, y: pd.Series) -> str:
        """Hash of the steps (with all parameters) and of the training data."""
        description: list[Any] = [
            (name, type(step).__module__, type(step).__qualname__, repr(step.get_params()))
            for name, step in steps
        ]
        data = [list(X.columns), X.to_numpy(), X.index.to_numpy(), y.to_numpy()]
        return joblib.hash([CACHE_VERSION, description, data])

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.joblib"

    def fit(self, steps: Steps, X: pd.DataFrame, y: pd.Series) -> Prepared:
        """Return the cached result for these steps and data, computing it on a miss."""
        path = self._path(self.key(steps, X, y))
        if path.exists():
            self.hits += 1
            return joblib.load(path)
        self.misses += 1
        prepared = fit_steps(steps, X, y)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        joblib.dump(prepared, temporary)
        temporary.replace(path)  # atomic: an interrupted run never leaves a partial file
        return prepared


def fit_preprocessing(
    steps: Steps, X: pd.DataFrame, y: pd.Series, cache: ResampleCache | None = None
) -> Prepared:
    """Fit the preprocessing steps; use ``cache`` only when they include a sampler."""
    if cache is not None and any(is_sampler(step) for _, step in steps):
        return cache.fit(steps, X, y)
    return fit_steps(steps, X, y)
