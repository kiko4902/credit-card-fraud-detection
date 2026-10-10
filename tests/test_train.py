import json
from pathlib import Path

import pandas as pd
import pytest

from fraud import train as train_module
from fraud.config import TARGET_COLUMN
from fraud.data import time_based_split
from fraud.models import build_pipeline
from fraud.train import RESULT_COLUMNS, cross_validate, format_report, run_experiment, save_result
from tests.conftest import make_synthetic_transactions


@pytest.fixture(scope="module")
def train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    # More rows and fraud than the default fixture, so every CV fold has enough fraud cases.
    splits = time_based_split(make_synthetic_transactions(n_rows=6_000, fraud_rate=0.05))
    return splits["train"], splits["val"]


@pytest.fixture(scope="module")
def baseline(train_val: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    train, val = train_val
    return run_experiment("logreg", "none", None, train, val, n_boot=50)


def test_cross_validate_reports_fraud_and_pr_auc_per_fold(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = train_val
    table = cross_validate(build_pipeline("logreg"), train)
    assert list(table.columns) == ["train_rows", "train_fraud", "val_rows", "val_fraud", "pr_auc"]
    assert len(table) == 3
    assert (table["val_fraud"] >= 30).all()
    assert table["pr_auc"].between(0.5, 1.0).all()  # the synthetic signal is easy to learn


def test_run_experiment_row_matches_the_schema(
    baseline: tuple[dict, pd.DataFrame], train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    row, cv_table = baseline
    train, val = train_val
    assert list(row) == RESULT_COLUMNS
    assert (row["model"], row["strategy"], row["sampling_ratio"]) == ("logreg", "none", None)
    assert row["eval_split"] == "val"
    assert row["n_train"] == len(train)
    assert row["n_eval"] == len(val)
    assert row["n_eval_fraud"] == val[TARGET_COLUMN].sum()
    assert row["tn"] + row["fp"] + row["fn"] + row["tp"] == len(val)
    assert row["pr_auc_ci_low"] <= row["pr_auc"] <= row["pr_auc_ci_high"]
    assert row["recall_ci_low"] <= row["recall"] <= row["recall_ci_high"]
    assert row["threshold"] == 0.5
    assert row["cv_folds"] == len(cv_table) == 3
    assert row["cv_pr_auc_mean"] == pytest.approx(cv_table["pr_auc"].mean())
    assert json.loads(row["params"])["max_iter"] == 1000
    assert row["train_time_s"] >= 0


def test_run_experiment_without_cv(train_val: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, val = train_val
    row, cv_table = run_experiment("logreg", "smote", 0.5, train, val, run_cv=False, n_boot=20)
    assert cv_table is None
    assert row["cv_folds"] == 0
    assert row["sampling_ratio"] == 0.5
    assert row["n_eval"] == len(val)  # validation is scored as is, never resampled


def test_save_result_creates_appends_and_replaces(
    baseline: tuple[dict, pd.DataFrame], tmp_path: Path
) -> None:
    row, _ = baseline
    path = tmp_path / "reports" / "results.csv"

    save_result(row, path)
    assert list(pd.read_csv(path).columns) == RESULT_COLUMNS

    # Same configuration again: the row is replaced, not duplicated.
    save_result({**row, "pr_auc": 0.123}, path)
    saved = pd.read_csv(path)
    assert len(saved) == 1
    assert saved.loc[0, "pr_auc"] == pytest.approx(0.123)

    # A different configuration is added.
    save_result({**row, "strategy": "smote", "sampling_ratio": 0.5}, path)
    save_result({**row, "strategy": "smote", "sampling_ratio": 0.1}, path)
    save_result({**row, "strategy": "smote", "sampling_ratio": 0.5, "pr_auc": 0.9}, path)
    saved = pd.read_csv(path)
    assert len(saved) == 3
    assert sorted(saved["strategy"]) == ["none", "smote", "smote"]
    assert saved.loc[saved["sampling_ratio"] == 0.5, "pr_auc"].item() == pytest.approx(0.9)


def test_save_result_refuses_a_file_with_another_schema(
    baseline: tuple[dict, pd.DataFrame], tmp_path: Path
) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame({"model": ["x"], "accuracy": [0.99]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="different schema"):
        save_result(baseline[0], path)


def test_format_report_shows_metrics_with_intervals(baseline: tuple[dict, pd.DataFrame]) -> None:
    report = format_report(*baseline)
    for expected in ("PR-AUC", "ROC-AUC", "recall", "precision", "F1", "95 % CI", "val_fraud"):
        assert expected in report


def test_git_commit_is_a_short_string() -> None:
    commit = train_module.git_commit()
    assert isinstance(commit, str) and commit


def test_training_code_never_loads_the_test_split() -> None:
    source = Path(train_module.__file__).read_text(encoding="utf-8")
    assert "load_test" not in source
    assert "time_based_split" not in source
