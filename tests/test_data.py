from pathlib import Path

import pandas as pd
import pytest

from fraud import data
from fraud.config import EXPECTED_COLUMNS, TARGET_COLUMN
from tests.conftest import N_ROWS, make_synthetic_transactions


def test_fixture_matches_real_schema(transactions: pd.DataFrame) -> None:
    assert list(transactions.columns) == EXPECTED_COLUMNS
    assert len(transactions) == N_ROWS
    assert 0 < transactions[TARGET_COLUMN].sum() < N_ROWS * 0.1
    assert transactions["Time"].is_monotonic_increasing


def test_fixture_is_deterministic() -> None:
    pd.testing.assert_frame_equal(make_synthetic_transactions(), make_synthetic_transactions())


def test_validate_raw_accepts_expected_frame(transactions: pd.DataFrame) -> None:
    data.validate_raw(transactions, expected_rows=N_ROWS)


def test_validate_raw_rejects_wrong_row_count(transactions: pd.DataFrame) -> None:
    with pytest.raises(data.DataValidationError, match="rows"):
        data.validate_raw(transactions, expected_rows=N_ROWS + 1)


def test_validate_raw_rejects_missing_column(transactions: pd.DataFrame) -> None:
    with pytest.raises(data.DataValidationError, match="Class"):
        data.validate_raw(transactions.drop(columns=[TARGET_COLUMN]), expected_rows=N_ROWS)


def test_ensure_raw_data_skips_download_when_file_exists(
    transactions: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "creditcard.csv"
    transactions.to_csv(csv_path, index=False)

    def fail_download(dest_dir: Path) -> None:
        raise AssertionError("download_raw must not be called when the file exists")

    monkeypatch.setattr(data, "download_raw", fail_download)
    assert data.ensure_raw_data(csv_path, expected_rows=N_ROWS) == csv_path


def test_ensure_raw_data_downloads_when_file_missing(
    transactions: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "raw" / "creditcard.csv"

    def fake_download(dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        transactions.to_csv(dest_dir / "creditcard.csv", index=False)

    monkeypatch.setattr(data, "download_raw", fake_download)
    assert data.ensure_raw_data(csv_path, expected_rows=N_ROWS) == csv_path
    assert csv_path.exists()


def test_load_raw_reads_and_validates(transactions: pd.DataFrame, tmp_path: Path) -> None:
    csv_path = tmp_path / "creditcard.csv"
    transactions.to_csv(csv_path, index=False)
    loaded = data.load_raw(csv_path, expected_rows=N_ROWS)
    assert list(loaded.columns) == EXPECTED_COLUMNS
    assert len(loaded) == N_ROWS


def test_load_raw_missing_file_points_to_make_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        data.load_raw(tmp_path / "missing.csv")


@pytest.fixture
def with_duplicates(transactions: pd.DataFrame) -> pd.DataFrame:
    """Unique rows plus 3 exact copies and 2 copies that differ only in Time."""
    exact = transactions.iloc[[0, 1, 2]]
    shifted = transactions.iloc[[10, 11]].assign(Time=lambda d: d["Time"] + 1.0)
    return pd.concat([transactions, exact, shifted], ignore_index=True)


def test_synthetic_fixture_has_no_duplicates(transactions: pd.DataFrame) -> None:
    assert data.duplicate_mask(transactions, ignore=["Time"]).sum() == 0


def test_duplicate_mask_exact(with_duplicates: pd.DataFrame) -> None:
    mask = data.duplicate_mask(with_duplicates)
    assert mask.sum() == 3
    assert not mask.iloc[:N_ROWS].any()  # first occurrences are kept
    assert len(with_duplicates[~mask]) == N_ROWS + 2


def test_duplicate_mask_ignoring_time(with_duplicates: pd.DataFrame) -> None:
    assert data.duplicate_mask(with_duplicates, ignore=["Time"]).sum() == 5


def test_duplicate_group_id_groups_identical_rows(with_duplicates: pd.DataFrame) -> None:
    exact_groups = data.duplicate_group_id(with_duplicates)
    assert exact_groups.nunique() == N_ROWS + 2
    assert exact_groups.iloc[0] == exact_groups.iloc[N_ROWS]

    groups = data.duplicate_group_id(with_duplicates, ignore=["Time"])
    assert groups.nunique() == N_ROWS
    assert groups.iloc[10] == groups.iloc[N_ROWS + 3]
    assert groups.iloc[10] != groups.iloc[11]


def test_time_based_split_is_chronological_and_complete(transactions: pd.DataFrame) -> None:
    splits = data.time_based_split(transactions)
    assert list(splits) == ["train", "val", "test"]
    assert sum(len(part) for part in splits.values()) == N_ROWS
    assert splits["train"]["Time"].max() < splits["val"]["Time"].min()
    assert splits["val"]["Time"].max() < splits["test"]["Time"].min()
    assert len(splits["train"]) == pytest.approx(0.70 * N_ROWS, abs=0.01 * N_ROWS)
    assert len(splits["val"]) == pytest.approx(0.15 * N_ROWS, abs=0.01 * N_ROWS)


def test_time_based_split_ignores_row_order(transactions: pd.DataFrame) -> None:
    shuffled = transactions.sample(frac=1.0, random_state=0)
    expected = data.time_based_split(transactions)
    actual = data.time_based_split(shuffled)
    for name in expected:
        assert set(actual[name].index) == set(expected[name].index)


def test_time_based_split_keeps_tied_timestamps_together(transactions: pd.DataFrame) -> None:
    # Coarse timestamps create many ties, including at the split boundaries.
    coarse = transactions.assign(Time=lambda d: (d["Time"] // 20_000) * 20_000)
    splits = data.time_based_split(coarse)
    seen: set[float] = set()
    for part in splits.values():
        times = set(part["Time"].unique())
        assert not times & seen
        seen |= times


def test_time_based_split_rejects_bad_fractions(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="train_frac"):
        data.time_based_split(transactions, train_frac=0.9, val_frac=0.2)


def test_time_based_split_rejects_constant_time(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="distinct time"):
        data.time_based_split(transactions.assign(Time=0.0))


def test_split_summary_counts(transactions: pd.DataFrame) -> None:
    summary = data.split_summary(data.time_based_split(transactions))
    assert summary["rows"].sum() == N_ROWS
    assert summary["fraud"].sum() == transactions[TARGET_COLUMN].sum()
    assert summary["share"].sum() == pytest.approx(1.0)
    assert (summary["time_start"] <= summary["time_end"]).all()
