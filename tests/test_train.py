import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE

from fraud import train as train_module
from fraud.cache import ResampleCache
from fraud.config import TARGET_COLUMN
from fraud.data import split_xy, time_based_split
from fraud.models import build_pipeline
from fraud.train import (
    RESULT_COLUMNS,
    cross_validate,
    fit_pipeline,
    format_report,
    run_experiment,
    save_result,
)
from tests.conftest import make_synthetic_transactions


@pytest.fixture(scope="module")
def train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    # More rows and fraud than the default fixture, so every CV fold has enough fraud cases.
    splits = time_based_split(make_synthetic_transactions(n_rows=6_000, fraud_rate=0.05))
    return splits["train"], splits["val"]


@pytest.fixture(scope="module")
def baseline(train_val: tuple[pd.DataFrame, pd.DataFrame]) -> tuple[dict, pd.DataFrame]:
    train, val = train_val
    result = run_experiment("logreg", "none", None, train, val, n_boot=50)
    assert result.val_scores.shape == (len(val),)
    return result.row, result.cv_table


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
    assert row["stage"] == "stage1"
    assert row["config_id"].startswith("logreg__none__na__")
    assert json.loads(row["cv_pr_auc_folds"]) == pytest.approx(list(cv_table["pr_auc"]), abs=1e-6)
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
    assert min(row["resample_time_s"], row["fit_time_s"], row["predict_time_s"]) >= 0


def test_run_experiment_without_cv(train_val: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, val = train_val
    result = run_experiment("logreg", "smote", 0.5, train, val, run_cv=False, n_boot=20)
    row = result.row
    assert result.cv_table is None
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
    smote_05 = {
        **row,
        "config_id": "logreg__smote__0.5__x",
        "strategy": "smote",
        "sampling_ratio": 0.5,
    }
    smote_01 = {
        **row,
        "config_id": "logreg__smote__0.1__x",
        "strategy": "smote",
        "sampling_ratio": 0.1,
    }
    save_result(smote_05, path)
    save_result(smote_01, path)
    save_result({**smote_05, "pr_auc": 0.9}, path)
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


# --- Cached resampling -----------------------------------------------------------------

CACHED_STRATEGIES = ["smote", "smote_tomek", "smote_enn", "safe_level_smote", "polynom_fit_smote"]


@pytest.mark.parametrize("strategy", CACHED_STRATEGIES)
def test_cached_and_uncached_results_are_identical(
    strategy: str, train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = train_val
    cache = ResampleCache(tmp_path / "cache")
    uncached = run_experiment("logreg", strategy, 0.5, train, val, n_boot=20)
    first = run_experiment("logreg", strategy, 0.5, train, val, n_boot=20, cache=cache)
    assert (cache.hits, cache.misses) == (0, 4)  # 3 CV folds + the full training split
    second = run_experiment("logreg", strategy, 0.5, train, val, n_boot=20, cache=cache)
    assert (cache.hits, cache.misses) == (4, 4)

    for cached in (first, second):
        np.testing.assert_array_equal(cached.val_scores, uncached.val_scores)
        pd.testing.assert_frame_equal(cached.cv_table, uncached.cv_table)
        for column in ("pr_auc", "pr_auc_ci_low", "roc_auc", "recall", "cv_pr_auc_mean", "tp"):
            assert cached.row[column] == uncached.row[column]
        assert cached.row["config_id"] == uncached.row["config_id"]


def test_fit_pipeline_matches_plain_pipeline_fit(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = train_val
    features, target = split_xy(train)
    val_features, _ = split_xy(val)
    plain = build_pipeline("lightgbm", "smote", 0.5, model_params={"n_estimators": 10})
    expected = plain.fit(features, target).predict_proba(val_features)

    cache = ResampleCache(tmp_path / "cache")
    template = build_pipeline("lightgbm", "smote", 0.5, model_params={"n_estimators": 10})
    for _ in range(2):  # a miss, then a hit
        fitted = fit_pipeline(template, features, target, cache)
        np.testing.assert_array_equal(fitted.pipeline.predict_proba(val_features), expected)
        assert [name for name, _ in fitted.pipeline.steps] == [
            "features",
            "scale",
            "sampler",
            "model",
        ]
    assert (cache.hits, cache.misses) == (1, 1)


def test_resampled_data_is_shared_between_classifiers(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, _ = train_val
    features, target = split_xy(train)
    calls: list[int] = []
    original = SMOTE.fit_resample

    def counting(self: SMOTE, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        calls.append(len(X))
        return original(self, X, y)

    monkeypatch.setattr(SMOTE, "fit_resample", counting)
    cache = ResampleCache(tmp_path / "cache")
    for model in ("logreg", "gaussian_nb", "decision_tree"):
        fit_pipeline(build_pipeline(model, "smote", 0.5), features, target, cache)
    assert calls == [len(features)]  # resampled once, reused twice


def test_cache_never_mixes_folds_ratios_seeds_or_samplers(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, _ = train_val
    features, target = split_xy(train)
    half = len(features) // 2
    cache = ResampleCache(tmp_path / "cache")

    def prepared_rows(pipeline, X=features, y=target) -> int:  # noqa: ANN001
        return len(cache.fit(pipeline.steps[:-1], X, y).X)

    full = prepared_rows(build_pipeline("logreg", "smote", 0.5))
    assert (
        prepared_rows(
            build_pipeline("logreg", "smote", 0.5), features.iloc[:half], target.iloc[:half]
        )
        < full
    )
    assert prepared_rows(build_pipeline("logreg", "smote", 0.1)) < full
    prepared_rows(build_pipeline("logreg", "smote", 0.5, random_state=1))
    prepared_rows(build_pipeline("logreg", "safe_level_smote", 0.5))
    assert (cache.hits, cache.misses) == (0, 5)  # five different keys

    # The scaler in a cached entry was fitted on exactly the rows of that entry.
    entry = cache.fit(
        build_pipeline("logreg", "smote", 0.5).steps[:-1], features.iloc[:half], target.iloc[:half]
    )
    assert cache.hits == 1
    assert dict(entry.steps)["scale"].n_samples_seen_ == half


def test_pipelines_without_a_sampler_are_not_cached(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, _ = train_val
    cache = ResampleCache(tmp_path / "cache")
    fit_pipeline(build_pipeline("logreg"), *split_xy(train), cache)
    assert (cache.hits, cache.misses) == (0, 0)
    assert not (tmp_path / "cache").exists()
