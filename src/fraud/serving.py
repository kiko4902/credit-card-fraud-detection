"""Serving core: load the saved pipeline once, score transactions, explain the score.

The API (``app/api.py``) and the Streamlit app (``app/streamlit_app.py``) both go through
:class:`FraudModel`, so the model logic exists in one place. This module only needs the
runtime dependencies (no plotting, no ``shap`` package): the contributions come from
XGBoost's own TreeSHAP implementation and equal the values of ``shap.TreeExplainer``.

The model output is a **fraud score** between 0 and 1. It ranks transactions and is
compared with a threshold; it is not a calibrated probability, because the model was
trained on oversampled data.
"""

import json
import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost
from imblearn.pipeline import Pipeline

from fraud.config import (
    AMOUNT_COLUMN,
    FINAL_CONFIG_NAME,
    FINAL_MODEL_NAME,
    MODELS_DIR,
    PCA_COLUMNS,
    TIME_COLUMN,
)
from fraud.features import FEATURE_LABELS, MODEL_FEATURES, SECONDS_PER_DAY

# Name of the time input towards clients: the second of the day, 0 to 86,399.
SECONDS_OF_DAY: str = "seconds_of_day"
# Columns a client sends, and the columns the pipeline expects (``Time`` first).
INPUT_COLUMNS: list[str] = [*PCA_COLUMNS, AMOUNT_COLUMN, SECONDS_OF_DAY]
PIPELINE_COLUMNS: list[str] = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
# Overrides the directory of the model files (used by the Docker image and by tests).
MODELS_DIR_ENV: str = "FRAUD_MODELS_DIR"
DEFAULT_TOP_FEATURES: int = 5
INCREASES: str = "increases"
DECREASES: str = "decreases"


def model_inputs(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """The rows as the classifier sees them: every transformer applied, samplers skipped."""
    for _, step in pipeline.steps[:-1]:
        if hasattr(step, "fit_resample"):
            continue
        features = step.transform(features)
    return features


@dataclass(frozen=True)
class Contribution:
    """How much one feature moved the score of one transaction, in log-odds."""

    feature: str
    label: str
    value: float
    contribution: float
    direction: str


@dataclass(frozen=True)
class Prediction:
    """Score and decision for one transaction."""

    fraud_score: float
    flagged: bool
    threshold: float
    top_features: list[Contribution]
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FraudModel:
    """The saved pipeline with its threshold and metadata."""

    def __init__(self, pipeline: Pipeline, config: dict[str, Any]) -> None:
        if config.get("threshold") is None:
            raise ValueError("The model config has no threshold; run the final evaluation first.")
        self.pipeline = pipeline
        self.config = config
        self.threshold: float = float(config["threshold"])
        self.version: str = str(config["model_version"])
        self._booster = pipeline.steps[-1][1].get_booster()

    @classmethod
    def load(cls, models_dir: Path | None = None) -> "FraudModel":
        """Read the pipeline and its config from ``models_dir`` (default: ``models/``)."""
        directory = Path(models_dir or os.environ.get(MODELS_DIR_ENV) or MODELS_DIR)
        config = json.loads((directory / FINAL_CONFIG_NAME).read_text(encoding="utf-8"))
        return cls(joblib.load(directory / FINAL_MODEL_NAME), config)

    @staticmethod
    def to_frame(transactions: pd.DataFrame | list[dict[str, float]]) -> pd.DataFrame:
        """Bring transactions into the pipeline's input layout.

        Accepts ``seconds_of_day`` or ``Time`` for the time of day. Only the second within
        the day matters to the model, so either gives the same features.
        """
        frame = pd.DataFrame(transactions)
        if SECONDS_OF_DAY in frame.columns:
            frame = frame.rename(columns={SECONDS_OF_DAY: TIME_COLUMN})
        missing = [column for column in PIPELINE_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing input columns: {missing}")
        frame = frame[PIPELINE_COLUMNS].astype(float)
        frame[TIME_COLUMN] = frame[TIME_COLUMN] % SECONDS_PER_DAY
        return frame.reset_index(drop=True)

    def _prepare(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, xgboost.DMatrix]:
        """Readable features (before scaling) and the matrix the booster scores.

        The feature step runs once and the booster gets a plain array: for single
        transactions most of the time would otherwise go into DataFrame handling.
        """
        readable = self.pipeline.steps[0][1].transform(frame)
        inputs = readable
        for _, step in self.pipeline.steps[1:-1]:
            if not hasattr(step, "fit_resample"):
                inputs = step.transform(inputs)
        matrix = xgboost.DMatrix(np.asarray(inputs, dtype=np.float32), feature_names=MODEL_FEATURES)
        return readable[MODEL_FEATURES], matrix

    def scores(self, transactions: pd.DataFrame | list[dict[str, float]]) -> np.ndarray:
        """Fraud score of every transaction."""
        _, matrix = self._prepare(self.to_frame(transactions))
        return self._booster.predict(matrix)

    def flag(self, scores: np.ndarray) -> np.ndarray:
        """The decision rule: flag when the score reaches the threshold."""
        return np.asarray(scores) >= self.threshold

    def contributions(
        self, transactions: pd.DataFrame | list[dict[str, float]]
    ) -> tuple[np.ndarray, float]:
        """TreeSHAP values (rows x features, log-odds) and the base value."""
        _, matrix = self._prepare(self.to_frame(transactions))
        values = self._booster.predict(matrix, pred_contribs=True)
        return values[:, :-1], float(values[0, -1])

    def predict(
        self,
        transactions: pd.DataFrame | list[dict[str, float]],
        top_n: int = DEFAULT_TOP_FEATURES,
    ) -> list[Prediction]:
        """Score, decision and the ``top_n`` strongest contributions of every transaction."""
        frame = self.to_frame(transactions)
        if frame.empty:
            return []
        readable_frame, matrix = self._prepare(frame)
        readable = readable_frame.to_numpy()
        scores = self._booster.predict(matrix)
        flagged = self.flag(scores)
        values = self._booster.predict(matrix, pred_contribs=True)[:, :-1]
        strongest = np.argsort(-np.abs(values), axis=1)[:, :top_n]
        predictions = []
        for row, columns in enumerate(strongest):
            top = [
                Contribution(
                    feature=MODEL_FEATURES[column],
                    label=FEATURE_LABELS[MODEL_FEATURES[column]],
                    value=float(readable[row, column]),
                    contribution=float(values[row, column]),
                    direction=INCREASES if values[row, column] > 0 else DECREASES,
                )
                for column in columns
            ]
            predictions.append(
                Prediction(
                    float(scores[row]), bool(flagged[row]), self.threshold, top, self.version
                )
            )
        return predictions

    def info(self) -> dict[str, Any]:
        """Version, decision rule, cost model and the headline metrics of the saved model."""
        config = self.config
        return {
            "model_version": self.version,
            "config_id": config["config_id"],
            "model": config["model"],
            "strategy": config["strategy"],
            "sampling_ratio": config["sampling_ratio"],
            "threshold": self.threshold,
            "decision": config["decision"],
            "cost_model": config["cost_model"],
            "metrics": config["metrics"],
            "output": config["output"],
            "input_features": INPUT_COLUMNS,
            "trained_on": config["trained_on"],
        }


@lru_cache(maxsize=1)
def get_model() -> FraudModel:
    """The model, loaded on first use and kept for the lifetime of the process."""
    return FraudModel.load()
