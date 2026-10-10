"""Feature derivation and univariate feature diagnostics.

Everything here is stateless, so it cannot leak information between splits. Fitted steps
(scaling, resampling) are added by ``fraud.models.build_pipeline``.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from fraud.config import AMOUNT_COLUMN, PCA_COLUMNS, TARGET_COLUMN, TIME_COLUMN

SECONDS_PER_HOUR: int = 3600
HOURS_PER_DAY: int = 24
SECONDS_PER_DAY: int = SECONDS_PER_HOUR * HOURS_PER_DAY

LOG_AMOUNT: str = "log_amount"
HOUR_SIN: str = "hour_sin"
HOUR_COS: str = "hour_cos"
RAW_FEATURES: list[str] = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN]
MODEL_FEATURES: list[str] = [*PCA_COLUMNS, LOG_AMOUNT, HOUR_SIN, HOUR_COS]
# Names shown to people (plots, API responses, the app).
FEATURE_LABELS: dict[str, str] = {
    **{column: column for column in PCA_COLUMNS},
    LOG_AMOUNT: "Amount (log)",
    HOUR_SIN: "Time of day (sine)",
    HOUR_COS: "Time of day (cosine)",
}


def elapsed_hour(time_seconds: pd.Series) -> pd.Series:
    """Whole hours since the first transaction (0, 1, 2, ... across both days)."""
    return (time_seconds // SECONDS_PER_HOUR).astype(int).rename("elapsed_hour")


def hour_of_day(time_seconds: pd.Series) -> pd.Series:
    """Hour within the day (0-23), assuming the recording starts at midnight.

    ``Time`` is seconds since the first transaction, not a clock time, so this is the hour
    relative to the start of the recording. The volume pattern suggests the start is close
    to midnight, but the dataset does not state it.
    """
    return (elapsed_hour(time_seconds) % HOURS_PER_DAY).rename("hour")


class FraudFeatures(TransformerMixin, BaseEstimator):
    """Turn raw transaction columns into the model's input features.

    Output columns (``MODEL_FEATURES``):

    - ``V1``-``V28`` unchanged.
    - ``log_amount = log1p(Amount)``: Amount is heavily right-skewed and contains zeros.
    - ``hour_sin``, ``hour_cos``: time of day on the unit circle.

    Raw ``Time`` is read to derive the time of day and then dropped: it only encodes the
    position in the recording and is out of the training range in validation and test.

    Why sine/cosine and not the hour as a number: hour 23 and hour 0 are neighbours, but as
    numbers they are the two most distant values, and a linear model can only fit a monotone
    effect to a single number while the fraud rate peaks at night and is flat otherwise. On
    the circle, midnight is continuous and a linear model can place one peak anywhere in the
    day. Tree models can also use the two columns (a time window becomes one or two splits),
    so one encoding serves every model and the comparison between models stays like for like.
    The angle uses the exact second of the day, not the rounded hour, to avoid artificial
    steps at full hours.

    The transformer is stateless (nothing is learned in ``fit``), so it cannot leak
    information between splits. It still lives inside the pipeline so that the features used
    for training and for prediction can never differ.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> "FraudFeatures":
        self._check_columns(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_columns(X)
        angle = 2 * np.pi * (X[TIME_COLUMN] % SECONDS_PER_DAY) / SECONDS_PER_DAY
        features = X[PCA_COLUMNS].copy()
        features[LOG_AMOUNT] = np.log1p(X[AMOUNT_COLUMN])
        features[HOUR_SIN] = np.sin(angle)
        features[HOUR_COS] = np.cos(angle)
        return features

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(MODEL_FEATURES, dtype=object)

    def __sklearn_is_fitted__(self) -> bool:
        return True

    @staticmethod
    def _check_columns(X: pd.DataFrame) -> None:
        missing = [column for column in RAW_FEATURES if column not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")


def class_separation(
    df: pd.DataFrame, features: Sequence[str], target_column: str = TARGET_COLUMN
) -> pd.DataFrame:
    """Rank features by how well each one alone separates fraud from legitimate rows.

    ``auc`` is the ROC-AUC of the raw feature value used as a score (equivalent to the
    Mann-Whitney U statistic), so it is rank-based and needs no model. ``separation`` is
    ``|2 * auc - 1|``: 0 = no separation, 1 = perfect, regardless of direction.
    ``median_diff_std`` is the fraud-minus-legit difference in medians, in units of the
    overall standard deviation of the feature.
    """
    is_fraud = df[target_column] == 1
    rows = {}
    for feature in features:
        values = df[feature]
        auc = roc_auc_score(is_fraud, values)
        median_legit = values[~is_fraud].median()
        median_fraud = values[is_fraud].median()
        rows[feature] = {
            "median_legit": median_legit,
            "median_fraud": median_fraud,
            "median_diff_std": (median_fraud - median_legit) / values.std(),
            "auc": auc,
            "separation": abs(2 * auc - 1),
        }
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("feature")
    return table.sort_values("separation", ascending=False)
