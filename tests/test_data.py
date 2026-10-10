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
