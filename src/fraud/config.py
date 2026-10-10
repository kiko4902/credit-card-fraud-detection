"""Project-wide configuration: paths, random seed, dataset facts and cost assumptions."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
RESULTS_CSV: Path = REPORTS_DIR / "results.csv"

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED: int = 42

# --- Dataset -----------------------------------------------------------------
KAGGLE_DATASET: str = "mlg-ulb/creditcardfraud"
RAW_CSV_NAME: str = "creditcard.csv"
RAW_CSV: Path = RAW_DIR / RAW_CSV_NAME

TIME_COLUMN: str = "Time"
AMOUNT_COLUMN: str = "Amount"
TARGET_COLUMN: str = "Class"
PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS: list[str] = [TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]
EXPECTED_ROWS: int = 284_807

# --- Cost model (PLAN.md section 8) ------------------------------------------
# Missed fraud (FN) costs the transaction Amount.
# A false alarm (FP) costs a fixed manual review.
# A caught fraud (TP) also incurs the review cost.
# A correctly passed legitimate transaction (TN) costs nothing.
REVIEW_COST_EUR: float = 5.0
FN_COST_IS_AMOUNT: bool = True
