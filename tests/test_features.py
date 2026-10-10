import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

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


def test_fraud_features_output_columns(transactions: pd.DataFrame) -> None:
    raw = transactions.drop(columns=["Class"])
    out = features.FraudFeatures().fit_transform(raw)
    assert list(out.columns) == features.MODEL_FEATURES
    assert list(out.columns) == [*PCA_COLUMNS, "log_amount", "hour_sin", "hour_cos"]
    assert "Time" not in out.columns
    assert "Amount" not in out.columns
    assert len(out) == len(raw)
    assert out.index.equals(raw.index)
    pd.testing.assert_frame_equal(out[PCA_COLUMNS], raw[PCA_COLUMNS])


def test_fraud_features_values() -> None:
    raw = pd.DataFrame(0.0, index=range(4), columns=features.RAW_FEATURES)
    raw["Time"] = [0.0, 6 * 3600.0, 12 * 3600.0, 24 * 3600.0 + 6 * 3600.0]
    raw["Amount"] = [0.0, np.e - 1, 99.0, 9.0]
    out = features.FraudFeatures().fit_transform(raw)
    assert out["log_amount"].tolist() == pytest.approx([0.0, 1.0, np.log(100.0), np.log(10.0)])
    assert out["hour_sin"].tolist() == pytest.approx([0.0, 1.0, 0.0, 1.0], abs=1e-12)
    assert out["hour_cos"].tolist() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-12)


def test_fraud_features_treat_midnight_as_continuous() -> None:
    raw = pd.DataFrame(0.0, index=range(2), columns=features.RAW_FEATURES)
    raw["Time"] = [86_400.0 - 1, 86_400.0 + 1]  # one second before and after midnight
    out = features.FraudFeatures().fit_transform(raw)
    assert abs(out["hour_sin"].diff().iloc[1]) < 1e-3
    assert abs(out["hour_cos"].diff().iloc[1]) < 1e-3


def test_fraud_features_ignore_extra_columns_and_do_not_mutate(transactions: pd.DataFrame) -> None:
    before = transactions.copy()
    out = features.FraudFeatures().fit_transform(transactions)  # includes Class
    assert "Class" not in out.columns
    pd.testing.assert_frame_equal(transactions, before)


def test_fraud_features_require_raw_columns(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Amount"):
        features.FraudFeatures().fit(transactions.drop(columns=["Amount"]))


def test_fraud_features_is_a_cloneable_sklearn_transformer(transactions: pd.DataFrame) -> None:
    transformer = clone(features.FraudFeatures()).fit(transactions)
    assert list(transformer.get_feature_names_out()) == features.MODEL_FEATURES
