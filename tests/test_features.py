import pandas as pd
import pytest

from fraud import features
from fraud.config import PCA_COLUMNS


def test_elapsed_hour() -> None:
    time = pd.Series([0.0, 3599.0, 3600.0, 86_400.0, 172_792.0])
    assert features.elapsed_hour(time).tolist() == [0, 0, 1, 24, 47]


def test_hour_of_day_wraps_after_24_hours() -> None:
    time = pd.Series([0.0, 3599.0, 3600.0, 86_399.0, 86_400.0, 172_792.0])
    hours = features.hour_of_day(time)
    assert hours.tolist() == [0, 0, 1, 23, 0, 23]
    assert hours.name == "hour"


def test_hour_of_day_range_and_index(transactions: pd.DataFrame) -> None:
    hours = features.hour_of_day(transactions["Time"])
    assert hours.index.equals(transactions.index)
    assert hours.between(0, 23).all()
    assert pd.api.types.is_integer_dtype(hours)


def test_class_separation_ranks_informative_features_first(transactions: pd.DataFrame) -> None:
    # The synthetic fixture shifts V1-V4 upwards for fraud rows and leaves the rest as noise.
    table = features.class_separation(transactions, PCA_COLUMNS)
    assert set(table.index[:4]) == {"V1", "V2", "V3", "V4"}
    assert table["separation"].is_monotonic_decreasing
    assert table["separation"].between(0, 1).all()
    assert (table.loc[["V1", "V2", "V3", "V4"], "auc"] > 0.8).all()
    assert (table.loc[["V1", "V2", "V3", "V4"], "median_diff_std"] > 1).all()
    assert table.loc["V20", "separation"] < 0.3


def test_class_separation_is_direction_agnostic(transactions: pd.DataFrame) -> None:
    flipped = transactions.assign(V1=lambda d: -d["V1"])
    original = features.class_separation(transactions, ["V1"])
    mirrored = features.class_separation(flipped, ["V1"])
    assert mirrored.loc["V1", "separation"] == pytest.approx(original.loc["V1", "separation"])
    assert mirrored.loc["V1", "auc"] == pytest.approx(1 - original.loc["V1", "auc"])
