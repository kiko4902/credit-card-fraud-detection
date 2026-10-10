"""Shared fixtures.

Tests never touch the real Kaggle dataset (CI has no Kaggle access). They use a small
synthetic frame with the same columns, generated with a fixed seed.
"""

import numpy as np
import pandas as pd
import pytest

from fraud.config import (
    AMOUNT_COLUMN,
    PCA_COLUMNS,
    RANDOM_SEED,
    TARGET_COLUMN,
    TIME_COLUMN,
)

N_ROWS = 2_000
FRAUD_RATE = 0.02


def make_synthetic_transactions(
    n_rows: int = N_ROWS, fraud_rate: float = FRAUD_RATE, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """Build a small imbalanced dataset shaped like creditcard.csv."""
    rng = np.random.default_rng(seed)
    target = (rng.random(n_rows) < fraud_rate).astype(int)
    data: dict[str, np.ndarray] = {
        TIME_COLUMN: np.sort(rng.uniform(0, 172_800, n_rows)).round(),
    }
    for i, column in enumerate(PCA_COLUMNS):
        # Shift a few components for fraud rows so models have some signal to learn.
        shift = 2.0 * target if i < 4 else 0.0
        data[column] = rng.normal(0.0, 1.0, n_rows) + shift
    data[AMOUNT_COLUMN] = rng.lognormal(3.0, 1.5, n_rows).round(2)
    data[TARGET_COLUMN] = target
    return pd.DataFrame(data)


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_synthetic_transactions()
