"""Project-wide configuration: paths, random seed, dataset facts and cost assumptions."""

import os
from dataclasses import dataclass
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# The repository root. When the package is installed as a wheel (Docker images) the root
# cannot be derived from this file, so FRAUD_PROJECT_ROOT names it.
PROJECT_ROOT: Path = Path(
    os.environ.get("FRAUD_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
)

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
# Version of the saved model artifact; increase it whenever the model is retrained.
MODEL_VERSION: str = "1.0.0"
# Phase 4: final evaluation.
FINAL_EVALUATION_MD: Path = REPORTS_DIR / "final_evaluation.md"
FINAL_METRICS_JSON: Path = REPORTS_DIR / "final_metrics.json"
# Every load of the held-out test split is appended to this file.
TEST_ACCESS_LOG: Path = REPORTS_DIR / "test_set_access.log"
# Small sample of explained test transactions for the Streamlit app.
APP_SAMPLE_CSV: Path = PROJECT_ROOT / "app" / "data" / "explained_sample.csv"
# Validation scores (score, label, amount) for the threshold page of the app.
APP_VALIDATION_SCORES: Path = PROJECT_ROOT / "app" / "data" / "validation_scores.csv.gz"

# --- Serving (PLAN.md section 10) ---------------------------------------------
# Largest number of transactions accepted by one POST /predict/batch request.
MAX_BATCH_SIZE: int = 1000

# --- Colours (figures and the app) -------------------------------------------
# Blue / orange stay distinguishable under the common colour-vision deficiencies.
# Legitimate is always blue and fraud always orange, in every figure and in the app.
LEGIT_COLOR: str = "#2a78d6"
FRAUD_COLOR: str = "#eb6834"
NEUTRAL_COLOR: str = "#52514e"

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


@dataclass(frozen=True)
class EvaluationPaths:
    """Where the final evaluation reads and writes; a dry run points the outputs elsewhere."""

    results: Path = RESULTS_CSV
    scores_dir: Path = VAL_SCORES_DIR
    models_dir: Path = MODELS_DIR
    figures_dir: Path = FIGURES_DIR
    report: Path = FINAL_EVALUATION_MD
    metrics: Path = FINAL_METRICS_JSON
    access_log: Path = TEST_ACCESS_LOG
    sample: Path = APP_SAMPLE_CSV
    # The final config is read from ``models_dir`` and the updated one written here.
    config_out: Path = MODELS_DIR / FINAL_CONFIG_NAME
