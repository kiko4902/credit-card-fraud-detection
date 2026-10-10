"""Project-wide configuration: paths, random seed, dataset facts and cost assumptions."""

from dataclasses import dataclass
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
EXPERIMENTS_SUMMARY: Path = REPORTS_DIR / "experiments_summary.md"
# Validation scores of every configuration, kept for paired comparisons (gitignored).
VAL_SCORES_DIR: Path = PROCESSED_DIR / "val_scores"
# Resampled training sets, shared between classifiers (gitignored, a few GB).
RESAMPLE_CACHE_DIR: Path = PROCESSED_DIR / "resample_cache"
FINAL_MODEL_NAME: str = "final_pipeline.joblib"
FINAL_CONFIG_NAME: str = "final_config.json"

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

# --- Time-based split (PLAN.md section 5) -------------------------------------
# Earliest 70 % of transactions for training, next 15 % validation, last 15 % test.
TRAIN_FRAC: float = 0.70
VAL_FRAC: float = 0.15

# --- Cross-validation inside the training split -------------------------------
# Time-ordered expanding-window folds. With 3 splits every validation fold of the real
# training data holds at least 56 fraud cases; 4 or more splits leave a fold with under 30.
CV_N_SPLITS: int = 3
MIN_FRAUD_PER_FOLD: int = 30

# --- Evaluation ---------------------------------------------------------------
DEFAULT_THRESHOLD: float = 0.5
N_BOOTSTRAP: int = 1000
CONFIDENCE_LEVEL: float = 0.95

# --- Experiment stages and model selection (PLAN.md sections 6 and 14) ----------
# Configurations within this distance of the best mean CV PR-AUC count as tied.
CV_TIE_TOLERANCE: float = 0.005
TOP_N_PAIRED: int = 3
# Classifiers that go from the screening stage to the SMOTE-family stage.
STAGE2_CLASSIFIERS: int = 4
# Classifiers that are tuned, each with its best strategy.
TUNING_FAMILIES: int = 2


@dataclass(frozen=True)
class ExperimentPaths:
    """Where the experiment stage reads and writes; tests point these at a temp directory."""

    results: Path = RESULTS_CSV
    scores_dir: Path = VAL_SCORES_DIR
    cache_dir: Path = RESAMPLE_CACHE_DIR
    figures_dir: Path = FIGURES_DIR
    summary: Path = EXPERIMENTS_SUMMARY
    models_dir: Path = MODELS_DIR
