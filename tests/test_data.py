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


def test_drop_exact_duplicates_removes_exact_copies_only(with_duplicates: pd.DataFrame) -> None:
    deduplicated = data.drop_exact_duplicates(with_duplicates)
    # The 3 exact copies go; the 2 rows that repeat at a different Time stay.
    assert len(deduplicated) == N_ROWS + 2
    assert data.duplicate_mask(deduplicated).sum() == 0
    assert data.duplicate_mask(deduplicated, ignore=["Time"]).sum() == 2
    pd.testing.assert_frame_equal(deduplicated.iloc[:N_ROWS], with_duplicates.iloc[:N_ROWS])


@pytest.fixture
def raw_csv(with_duplicates: pd.DataFrame, tmp_path: Path) -> Path:
    """A stand-in for creditcard.csv: synthetic rows in shuffled order, with duplicates."""
    path = tmp_path / "creditcard.csv"
    with_duplicates.sample(frac=1.0, random_state=0).to_csv(path, index=False)
    return path


def test_load_train_val_is_deduplicated_ordered_and_disjoint(raw_csv: Path) -> None:
    train, val = data.load_train_val(raw_csv, expected_rows=None)
    assert train["Time"].is_monotonic_increasing
    assert val["Time"].is_monotonic_increasing
    assert train["Time"].max() < val["Time"].min()
    assert not set(train.index) & set(val.index)
    assert data.duplicate_mask(pd.concat([train, val])).sum() == 0


def test_load_train_val_returns_only_two_splits(raw_csv: Path) -> None:
    splits = data.load_train_val(raw_csv, expected_rows=None)
    assert len(splits) == 2
    train, val = splits
    # Together they hold about 85 % of the rows: the test part is not handed out.
    assert len(train) + len(val) == pytest.approx(0.85 * (N_ROWS + 2), abs=0.01 * N_ROWS)


def test_test_split_requires_explicit_permission(raw_csv: Path) -> None:
    with pytest.raises(data.HeldOutTestSetError, match="allow_test=True"):
        data.load_test(path=raw_csv, expected_rows=None)
    with pytest.raises(data.HeldOutTestSetError):
        data.load_test(allow_test=False, path=raw_csv, expected_rows=None)
    with pytest.raises(data.HeldOutTestSetError):
        data.load_test(allow_test="yes", path=raw_csv, expected_rows=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        data.load_test(True, path=raw_csv, expected_rows=None)  # type: ignore[misc]


def test_test_split_follows_validation_in_time(raw_csv: Path) -> None:
    _, val = data.load_train_val(raw_csv, expected_rows=None)
    test = data.load_test(allow_test=True, path=raw_csv, expected_rows=None)
    assert val["Time"].max() < test["Time"].min()
    assert not set(val.index) & set(test.index)


def test_split_xy_removes_only_the_target(transactions: pd.DataFrame) -> None:
    features, target = data.split_xy(transactions)
    assert list(features.columns) == EXPECTED_COLUMNS[:-1]
    assert target.name == TARGET_COLUMN
    assert len(features) == len(target) == N_ROWS


def test_time_ordered_folds_never_validate_on_the_past(transactions: pd.DataFrame) -> None:
    folds = data.time_ordered_folds(transactions, n_splits=3, min_fraud=1)
    assert len(folds) == 3
    previous_train_size = 0
    for train_idx, val_idx in folds:
        assert train_idx.max() < val_idx.min()
        train_time = transactions["Time"].iloc[train_idx]
        assert train_time.max() <= transactions["Time"].iloc[val_idx].min()
        assert len(train_idx) > previous_train_size  # expanding window
        previous_train_size = len(train_idx)


def test_time_ordered_folds_reject_unsorted_data(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="sorted"):
        data.time_ordered_folds(transactions.sample(frac=1.0, random_state=0), min_fraud=1)


def test_time_ordered_folds_require_enough_fraud(transactions: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="fraud cases"):
        data.time_ordered_folds(transactions, n_splits=3, min_fraud=N_ROWS)


def test_fold_summary_reports_fraud_per_fold(transactions: pd.DataFrame) -> None:
    folds = data.time_ordered_folds(transactions, n_splits=3, min_fraud=1)
    summary = data.fold_summary(transactions, folds)
    assert list(summary.index) == [1, 2, 3]
    for (train_idx, val_idx), (_, row) in zip(folds, summary.iterrows(), strict=True):
        assert row["train_rows"] == len(train_idx)
        assert row["val_fraud"] == transactions[TARGET_COLUMN].iloc[val_idx].sum()
    assert (summary["val_fraud"] >= 1).all()


@pytest.fixture
def csv_with_early_and_late_duplicates(transactions: pd.DataFrame, tmp_path: Path) -> Path:
    """Synthetic rows plus exact copies of two early rows and three late rows."""
    ordered = transactions.sort_values("Time")
    copies = pd.concat([ordered.iloc[[5, 9]], ordered.iloc[[-4, -3, -3]]])
    path = tmp_path / "creditcard.csv"
    pd.concat([transactions, copies]).sample(frac=1.0, random_state=1).to_csv(path, index=False)
    return path


def test_test_split_with_duplicates_requires_explicit_permission(raw_csv: Path) -> None:
    with pytest.raises(data.HeldOutTestSetError, match="allow_test=True"):
        data.load_test_with_duplicates(path=raw_csv, expected_rows=None)


def test_test_split_with_duplicates_returns_the_duplicates_of_its_period(
    csv_with_early_and_late_duplicates: Path,
) -> None:
    path = csv_with_early_and_late_duplicates
    test, duplicates = data.load_test_with_duplicates(
        allow_test=True, path=path, expected_rows=None
    )
    pd.testing.assert_frame_equal(
        test, data.load_test(allow_test=True, path=path, expected_rows=None)
    )
    assert len(duplicates) == 3  # the copies of the two early rows belong to training
    assert duplicates["Time"].min() >= test["Time"].min()
    assert data.duplicate_mask(pd.concat([test, duplicates])).sum() == 3
    assert len(data.dropped_duplicates(path, expected_rows=None)) == 5


def test_stratified_random_split_keeps_the_fraud_rate_and_every_row(
    transactions: pd.DataFrame,
) -> None:
    splits = data.stratified_random_split(transactions)
    assert sum(len(part) for part in splits.values()) == len(transactions)
    indices = [set(part.index) for part in splits.values()]
    assert not (indices[0] & indices[1] or indices[0] & indices[2] or indices[1] & indices[2])
    assert len(splits["train"]) == pytest.approx(0.70 * len(transactions), abs=2)
    assert len(splits["val"]) == pytest.approx(0.15 * len(transactions), abs=2)
    rate = transactions[TARGET_COLUMN].mean()
    for part in splits.values():
        assert part[TARGET_COLUMN].mean() == pytest.approx(rate, abs=0.004)
    # Unlike the primary split, it mixes early and late transactions.
    assert splits["test"]["Time"].min() < splits["train"]["Time"].max()
    again = data.stratified_random_split(transactions)
    pd.testing.assert_frame_equal(splits["test"], again["test"])
