"""Download, validate and load the raw dataset; deduplication; time-based split and folds.

Modelling code loads data only through :func:`load_train_val`. The test split is behind
:func:`load_test`, which requires ``allow_test=True``.

Run the download with ``python -m fraud.data`` (or ``make data``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from fraud.config import (
    CV_N_SPLITS,
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    KAGGLE_DATASET,
    MIN_FRAUD_PER_FOLD,
    RAW_CSV,
    TARGET_COLUMN,
    TIME_COLUMN,
    TRAIN_FRAC,
    VAL_FRAC,
)

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the raw dataset does not have the expected shape or columns."""


def validate_raw(df: pd.DataFrame, expected_rows: int | None = EXPECTED_ROWS) -> None:
    """Check columns (names and order) and, unless ``expected_rows`` is None, the row count."""
    columns = list(df.columns)
    if columns != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(columns))
        extra = sorted(set(columns) - set(EXPECTED_COLUMNS))
        raise DataValidationError(
            f"Unexpected columns. Missing: {missing or 'none'}; extra: {extra or 'none'}; "
            f"got {len(columns)} columns, expected {len(EXPECTED_COLUMNS)}."
        )
    if expected_rows is not None and len(df) != expected_rows:
        raise DataValidationError(f"Expected {expected_rows:,} rows, found {len(df):,}.")


def download_raw(dest_dir: Path) -> None:
    """Download and unzip the Kaggle dataset into ``dest_dir``.

    Credentials are resolved by the kaggle package itself (``KAGGLE_API_TOKEN`` or
    ``~/.kaggle/access_token``); this module never reads or logs them.
    """
    # Imported lazily: importing kaggle attempts authentication, which tests and CI must avoid.
    from kaggle.api.kaggle_api_extended import KaggleApi

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=True, quiet=False)


def load_raw(path: Path = RAW_CSV, expected_rows: int | None = EXPECTED_ROWS) -> pd.DataFrame:
    """Read the raw CSV and validate its columns and row count."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` first.")
    df = pd.read_csv(path)
    validate_raw(df, expected_rows=expected_rows)
    return df


def ensure_raw_data(path: Path = RAW_CSV, expected_rows: int | None = EXPECTED_ROWS) -> Path:
    """Download the raw CSV if it is not already present, then validate it."""
    if path.exists():
        logger.info("Found %s, skipping download.", path)
    else:
        logger.info("Downloading %s into %s ...", KAGGLE_DATASET, path.parent)
        download_raw(path.parent)
        if not path.exists():
            raise FileNotFoundError(f"Download finished but {path} was not created.")
    df = load_raw(path, expected_rows=expected_rows)
    logger.info("Validated %s: %s rows, %d columns.", path.name, f"{len(df):,}", df.shape[1])
    return path


def _columns_except(df: pd.DataFrame, ignore: Sequence[str]) -> list[str]:
    return [column for column in df.columns if column not in set(ignore)]


def duplicate_mask(df: pd.DataFrame, ignore: Sequence[str] = ()) -> pd.Series:
    """Flag rows that repeat an earlier row, comparing all columns except ``ignore``.

    The first occurrence of each group is not flagged, so ``df[~mask]`` is the deduplicated
    frame.
    """
    return df.duplicated(subset=_columns_except(df, ignore), keep="first")


def duplicate_group_id(df: pd.DataFrame, ignore: Sequence[str] = ()) -> pd.Series:
    """Label rows so that rows identical on all columns except ``ignore`` share an id."""
    groups = df.groupby(_columns_except(df, ignore), sort=False, dropna=False)
    return groups.ngroup().rename("duplicate_group")


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    time_column: str = TIME_COLUMN,
) -> dict[str, pd.DataFrame]:
    """Split chronologically into train / val / test without shuffling.

    Low-level helper that returns all three parts (the EDA uses it to describe the split).
    Modelling code must use :func:`load_train_val` and :func:`load_test` instead.

    Boundaries are cut on ``time_column`` values rather than row positions, so rows that
    share a timestamp always land in the same split and the splits never overlap in time.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to less than 1.")
    times = np.sort(df[time_column].to_numpy())
    val_start = times[int(len(times) * train_frac)]
    test_start = times[int(len(times) * (train_frac + val_frac))]
    if not times[0] < val_start < test_start:
        raise ValueError("Not enough distinct time values to form three non-empty splits.")
    time = df[time_column]
    return {
        "train": df[time < val_start],
        "val": df[(time >= val_start) & (time < test_start)],
        "test": df[time >= test_start],
    }


def split_summary(
    splits: dict[str, pd.DataFrame],
    target_column: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> pd.DataFrame:
    """Rows, fraud cases, fraud rate and time range of each split."""
    total = sum(len(part) for part in splits.values())
    rows = {
        name: {
            "rows": len(part),
            "share": len(part) / total,
            "fraud": int(part[target_column].sum()),
            "fraud_rate": part[target_column].mean(),
            "time_start": part[time_column].min(),
            "time_end": part[time_column].max(),
        }
        for name, part in splits.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("split")


class HeldOutTestSetError(RuntimeError):
    """Raised when the test split is requested without explicit permission."""


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical to an earlier row in every column (the first occurrence is kept).

    Rows that repeat only at a different ``Time`` are distinct transactions and are kept.
    """
    return df[~duplicate_mask(df)]


def _load_splits(
    path: Path, expected_rows: int | None, deduplicate: bool
) -> dict[str, pd.DataFrame]:
    df = load_raw(path, expected_rows=expected_rows)
    if deduplicate:
        df = drop_exact_duplicates(df)
    # Stable sort: time-ordered cross-validation relies on row order following Time.
    return time_based_split(df.sort_values(TIME_COLUMN, kind="stable"))


def load_train_val(
    path: Path = RAW_CSV, expected_rows: int | None = EXPECTED_ROWS, deduplicate: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, deduplicate and split the data, returning only the train and validation parts.

    This is the entry point for all modelling code. The test part is deliberately not
    returned; see :func:`load_test`.
    """
    splits = _load_splits(path, expected_rows, deduplicate)
    return splits["train"], splits["val"]


def load_test(
    *,
    allow_test: bool = False,
    path: Path = RAW_CSV,
    expected_rows: int | None = EXPECTED_ROWS,
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Return the held-out test split. Only for the final evaluation in Phase 4.

    Refuses to run unless ``allow_test=True`` is passed, so the test set cannot be used by
    accident during model development.
    """
    if allow_test is not True:
        raise HeldOutTestSetError(
            "The test split is reserved for the final evaluation (Phase 4). "
            "Pass allow_test=True only there."
        )
    return _load_splits(path, expected_rows, deduplicate)["test"]


def split_xy(
    df: pd.DataFrame, target_column: str = TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature columns from the target."""
    return df.drop(columns=[target_column]), df[target_column]


def time_ordered_folds(
    train: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
    min_fraud: int = MIN_FRAUD_PER_FOLD,
    target_column: str = TARGET_COLUMN,
    time_column: str = TIME_COLUMN,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds for cross-validation inside the training split.

    Each fold trains on an initial stretch of the recording and validates on the stretch that
    follows it, so a model is never validated on data older than what it was trained on.
    Returns positional indices. Raises if any validation fold has fewer than ``min_fraud``
    fraud cases, because metrics on such a fold are mostly noise.
    """
    if not train[time_column].is_monotonic_increasing:
        raise ValueError(f"Rows must be sorted by {time_column} for time-ordered folds.")
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(train))
    fraud_counts = [int(train[target_column].iloc[val_idx].sum()) for _, val_idx in folds]
    if min(fraud_counts) < min_fraud:
        raise ValueError(
            f"Validation folds hold {fraud_counts} fraud cases; at least {min_fraud} are "
            "required in each. Use fewer splits."
        )
    return folds


def fold_summary(
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Rows and fraud cases in the training and validation part of each fold."""
    target = train[target_column]
    rows = [
        {
            "fold": number,
            "train_rows": len(train_idx),
            "train_fraud": int(target.iloc[train_idx].sum()),
            "val_rows": len(val_idx),
            "val_fraud": int(target.iloc[val_idx].sum()),
        }
        for number, (train_idx, val_idx) in enumerate(folds, start=1)
    ]
    return pd.DataFrame(rows).set_index("fold")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_raw_data()


if __name__ == "__main__":
    main()
