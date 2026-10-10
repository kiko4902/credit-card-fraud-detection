"""Download and validate the raw dataset.

Run with ``python -m fraud.data`` (or ``make data``). Loading helpers and the time-based
split are added in Phase 2.
"""

import logging
from pathlib import Path

import pandas as pd

from fraud.config import EXPECTED_COLUMNS, EXPECTED_ROWS, KAGGLE_DATASET, RAW_CSV

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


def ensure_raw_data(path: Path = RAW_CSV, expected_rows: int | None = EXPECTED_ROWS) -> Path:
    """Download the raw CSV if it is not already present, then validate it."""
    if path.exists():
        logger.info("Found %s, skipping download.", path)
    else:
        logger.info("Downloading %s into %s ...", KAGGLE_DATASET, path.parent)
        download_raw(path.parent)
        if not path.exists():
            raise FileNotFoundError(f"Download finished but {path} was not created.")
    df = pd.read_csv(path)
    validate_raw(df, expected_rows=expected_rows)
    logger.info("Validated %s: %s rows, %d columns.", path.name, f"{len(df):,}", df.shape[1])
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_raw_data()


if __name__ == "__main__":
    main()
