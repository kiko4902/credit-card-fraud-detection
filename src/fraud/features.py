"""Feature derivation and univariate feature diagnostics.

Fitted transforms (scaling etc.) are added in Phase 2 and live inside the training pipeline.
Everything here is stateless, so it cannot leak information between splits.
"""

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import roc_auc_score

from fraud.config import TARGET_COLUMN

SECONDS_PER_HOUR: int = 3600
HOURS_PER_DAY: int = 24


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
