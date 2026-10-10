"""Download, validate and load the raw dataset; duplicate detection; time-based split.

Run the download with ``python -m fraud.data`` (or ``make data``).
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    KAGGLE_DATASET,
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_raw_data()


if __name__ == "__main__":
    main()
