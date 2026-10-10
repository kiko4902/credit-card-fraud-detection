"""Export the validation scores that the Streamlit app's threshold page needs.

    python -m fraud.app_data        (or ``make app-data``)

Writes one small compressed file with the fraud score, the label and the amount of every
validation transaction. Nothing else about the transactions is stored, and the test split
is not touched. Needs the raw dataset; the app itself only needs the committed file.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fraud.config import AMOUNT_COLUMN, APP_VALIDATION_SCORES, TARGET_COLUMN
from fraud.data import load_train_val
from fraud.serving import FraudModel

SCORE_COLUMN: str = "score"
# Significant digits kept for the scores: enough that no decision at the threshold changes.
SCORE_DIGITS: int = 6


def validation_scores(model: FraudModel, val: pd.DataFrame) -> pd.DataFrame:
    """Score, label and amount of every validation row, in random order."""
    table = pd.DataFrame(
        {
            SCORE_COLUMN: model.scores(val),
            TARGET_COLUMN: val[TARGET_COLUMN].to_numpy(),
            AMOUNT_COLUMN: val[AMOUNT_COLUMN].to_numpy(),
        }
    )
    return table.sample(frac=1, random_state=0).reset_index(drop=True)


def export(model: FraudModel, val: pd.DataFrame, path: Path = APP_VALIDATION_SCORES) -> Path:
    """Write the table and check that rounding the scores changed no decision."""
    table = validation_scores(model, val)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        path,
        index=False,
        float_format=f"%.{SCORE_DIGITS}g",
        lineterminator="\n",
        compression={"method": "gzip", "mtime": 0},
    )
    saved = pd.read_csv(path)
    if not np.array_equal(model.flag(saved[SCORE_COLUMN]), model.flag(table[SCORE_COLUMN])):
        raise ValueError("Rounding the scores changed a decision; keep more digits.")
    return path


def main() -> None:
    _, val = load_train_val()
    path = export(FraudModel.load(), val)
    print(f"{path}: {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
