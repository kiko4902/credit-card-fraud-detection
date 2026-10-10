import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

import fraud
from fraud import data, explain, final_eval, report
from fraud.config import (
    FINAL_CONFIG_NAME,
    FINAL_MODEL_NAME,
    PCA_COLUMNS,
    EvaluationPaths,
    ExperimentPaths,
)
from fraud.experiments import ExperimentConfig, run_configs
from fraud.features import MODEL_FEATURES
from fraud.final_eval import SELECTED
from fraud.train import fraud_scores, load_results
from tests.conftest import make_synthetic_transactions

N_BOOT = 20
FIGURES = [
    "eval_cost_vs_threshold.png",
    "eval_cost_by_model.png",
    "eval_reliability.png",
    "eval_test_pr_curves.png",
    "eval_shap_summary.png",
]


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A complete final evaluation on synthetic data, with XGBoost as the selected model."""
    root = tmp_path_factory.mktemp("final")
    splits = data.time_based_split(make_synthetic_transactions(n_rows=6_000, fraud_rate=0.05))
    train, val, test = splits["train"], splits["val"], splits["test"]
    experiment_paths = ExperimentPaths(
        results=root / "results.csv",
        scores_dir=root / "scores",
        cache_dir=root / "cache",
        figures_dir=root / "figures",
        summary=root / "summary.md",
        models_dir=root / "models",
    )
    configs = [
        ExperimentConfig("xgboost", "class_weight"),
        ExperimentConfig("xgboost", "none"),
        ExperimentConfig("logreg", "none"),
    ]
    run_configs(configs, train, val, experiment_paths, n_boot=N_BOOT, log=lambda _: None)
    results = load_results(experiment_paths.results)
    selected = results[results["config_id"] == configs[0].id].iloc[0]
    report.save_final_model(selected, train, val, experiment_paths.models_dir)

    paths = EvaluationPaths(
        results=experiment_paths.results,
        scores_dir=experiment_paths.scores_dir,
        models_dir=experiment_paths.models_dir,
        figures_dir=root / "figures",
        report=root / "final_evaluation.md",
        metrics=root / "final_metrics.json",
        access_log=root / "test_set_access.log",
        sample=root / "app" / "explained_sample.csv",
        config_out=experiment_paths.models_dir / FINAL_CONFIG_NAME,
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(final_eval, "selected_row", lambda _: selected)
    patch.setitem(final_eval.REFERENCES, "mlp_none", ("logreg", "none", None))
    duplicates = test.iloc[:7]
    try:
        metrics = final_eval.run(train, val, test, duplicates, results, paths, n_boot=N_BOOT)
    finally:
        patch.undo()
    return {"paths": paths, "metrics": metrics, "splits": splits, "duplicates": duplicates}


def test_outputs_exist(finished: dict[str, Any]) -> None:
    paths: EvaluationPaths = finished["paths"]
    for name in FIGURES:
        assert (paths.figures_dir / name).stat().st_size > 0
    assert any(paths.figures_dir.glob("eval_shap_*fraud.png"))
    assert paths.report.exists() and paths.metrics.exists() and paths.sample.exists()
    assert json.loads(paths.metrics.read_text(encoding="utf-8")) == json.loads(
        json.dumps(finished["metrics"])
    )


def test_thresholds_come_from_validation_and_are_used_on_test(finished: dict[str, Any]) -> None:
    metrics, splits = finished["metrics"], finished["splits"]
    assert set(metrics["validation"]) == {SELECTED, *final_eval.REFERENCES}
    val = splits["val"]
    pipeline = joblib.load(finished["paths"].models_dir / FINAL_MODEL_NAME)
    scores = fraud_scores(pipeline, data.split_xy(val)[0])
    expected, cost = final_eval.CostCurve(val["Class"], scores, val["Amount"]).best()
    entry = metrics["validation"][SELECTED]
    assert entry["threshold"] == expected
    assert entry["at_cost_threshold"]["cost"]["value"] == pytest.approx(cost)
    assert entry["scores_match_phase3"]
    for key, model in metrics["test"]["models"].items():
        assert model["threshold"] == metrics["validation"][key]["threshold"]
    # No threshold can beat the cost-optimal one on the rows it was chosen on.
    assert cost <= entry["at_default_threshold"]["cost"]
    assert cost <= entry["at_f1_threshold"]["cost"]


def test_test_metrics_are_computed_on_the_test_rows(finished: dict[str, Any]) -> None:
    metrics, test = finished["metrics"], finished["splits"]["test"]
    selected = metrics["test"]["models"][SELECTED]["metrics"]
    counts = sum(selected[name]["value"] for name in ("tp", "fp", "fn", "tn"))
    assert counts == len(test) == metrics["splits"]["evaluation"]["rows"]
    assert selected["tp"]["value"] + selected["fn"]["value"] == test["Class"].sum()
    assert selected["cost_no_model"]["value"] == pytest.approx(
        test.loc[test["Class"] == 1, "Amount"].sum()
    )
    with_duplicates = metrics["test"]["with_duplicates"]
    assert with_duplicates["rows"] == len(test) + len(finished["duplicates"])
    assert metrics["test"]["hindsight"]["cost"] <= selected["cost"]["value"]
    assert len(metrics["hypothesis"]) == len(metrics["test"]["paired"]) == 3


def test_random_split_is_a_separate_refit(finished: dict[str, Any]) -> None:
    random_split = finished["metrics"]["random_split"]
    splits = finished["splits"]
    assert sum(random_split["rows"].values()) == sum(len(part) for part in splits.values())
    assert sum(random_split["fraud"].values()) == sum(p["Class"].sum() for p in splits.values())


def test_saved_config_holds_threshold_cost_model_and_metrics(finished: dict[str, Any]) -> None:
    paths: EvaluationPaths = finished["paths"]
    config = json.loads(paths.config_out.read_text(encoding="utf-8"))
    assert config["threshold"] == finished["metrics"]["validation"][SELECTED]["threshold"]
    assert config["model_version"] == fraud.config.MODEL_VERSION
    assert config["cost_model"]["review_cost_per_flagged_transaction"] == 5.0
    assert config["decision"]["refit_on_train_plus_val"] is False
    assert len(config["training_data"]["train_split_sha256"]) == 64
    assert {"validation_at_threshold", "test_at_threshold"} <= set(config["metrics"])
    assert "not a calibrated probability" in config["output"]


def test_phase3_report_keeps_the_phase4_keys(finished: dict[str, Any]) -> None:
    paths: EvaluationPaths = finished["paths"]
    before = json.loads(paths.config_out.read_text(encoding="utf-8"))
    results = load_results(paths.results)
    selected = results[results["config_id"] == before["config_id"]].iloc[0]
    splits = finished["splits"]
    report.save_final_model(selected, splits["train"], splits["val"], paths.models_dir)
    after = json.loads(paths.config_out.read_text(encoding="utf-8"))
    assert after["threshold"] == before["threshold"]
    assert after["metrics"] == before["metrics"]


def test_report_has_every_section_and_keeps_hand_written_text(finished: dict[str, Any]) -> None:
    paths: EvaluationPaths = finished["paths"]
    text = paths.report.read_text(encoding="utf-8")
    for heading in (
        "## Conclusions",
        "## Limitations",
        "## Threshold (chosen on validation only)",
        "## Thesis hypothesis test on validation",
        "## Final results on the test split",
        "### With and without deduplication",
        "### Secondary analysis: stratified random split",
        "## Score or probability?",
        "## Explainability (SHAP)",
    ):
        assert heading in text
    assert "DRY RUN" not in text
    edited = text.replace("_To be written after reviewing the results below._", "My notes.", 1)
    paths.report.write_text(edited, encoding="utf-8")
    final_eval.write_report(paths.report, final_eval.render(finished["metrics"]))
    assert "My notes." in paths.report.read_text(encoding="utf-8")


def test_dry_run_report_is_labelled(finished: dict[str, Any]) -> None:
    metrics = {**finished["metrics"], "dry_run": True, "evaluation_split": "validation (dry run)"}
    assert "**DRY RUN.**" in final_eval.render(metrics)


def test_explained_sample_reproduces_the_scores(finished: dict[str, Any]) -> None:
    paths: EvaluationPaths = finished["paths"]
    sample = pd.read_csv(paths.sample)
    assert {"score", "flagged", "outcome", "base_value"} <= set(sample.columns)
    assert [f"{explain.SHAP_PREFIX}{name}" for name in MODEL_FEATURES] == [
        column for column in sample.columns if column.startswith(explain.SHAP_PREFIX)
    ]
    assert sample["Time"].between(0, 86_400).all()  # second of the day, not the position
    assert (sample["outcome"] != "legitimate").sum() > 0
    assert (sample["outcome"] == "legitimate").sum() <= 100
    pipeline = joblib.load(paths.models_dir / FINAL_MODEL_NAME)
    scores = fraud_scores(pipeline, sample[["Time", *PCA_COLUMNS, "Amount"]])
    np.testing.assert_allclose(scores, sample["score"], atol=1e-4)
    # Base value plus SHAP values is the model's raw output: the log-odds of the score.
    shap_columns = [column for column in sample.columns if column.startswith(explain.SHAP_PREFIX)]
    log_odds = sample["base_value"] + sample[shap_columns].sum(axis=1)
    inside = sample["score"].between(1e-4, 1 - 1e-4)
    np.testing.assert_allclose(
        1 / (1 + np.exp(-log_odds[inside])), sample.loc[inside, "score"], atol=1e-3
    )


def test_pick_cases_uses_fixed_rules() -> None:
    y_true = pd.Series([1, 1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.1, 0.7, 0.95, 0.2])
    amount = pd.Series([10.0, 500.0, 40.0, 1.0, 1.0, 1.0])
    cases = explain.pick_cases(y_true, scores, amount, threshold=0.5)
    assert cases == {"caught_fraud": 1, "missed_fraud": 2, "false_alarm": 4}
    assert "missed_fraud" not in explain.pick_cases(y_true, scores, amount, threshold=0.05)


def test_dry_run_never_loads_the_test_split(
    finished: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("a dry run must not load the test split")

    val = finished["splits"]["val"]
    monkeypatch.setattr(final_eval, "load_test_with_duplicates", forbidden)
    monkeypatch.setattr(final_eval, "dropped_duplicates", lambda: val.iloc[[0, 1]])
    log = tmp_path / "access.log"
    rows, duplicates = final_eval.load_evaluation_rows(True, val, log)
    assert rows is val and len(duplicates) == 2
    assert not log.exists()


def test_real_run_loads_the_test_split_once_and_logs_it(
    finished: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test = finished["splits"]["test"]
    calls = []

    def fake_load(*, allow_test: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(allow_test)
        return test, test.iloc[:3]

    monkeypatch.setattr(final_eval, "load_test_with_duplicates", fake_load)
    log = tmp_path / "access.log"
    rows, _ = final_eval.load_evaluation_rows(False, finished["splits"]["val"], log)
    assert rows is test and calls == [True]
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and f"test rows {len(test)}" in lines[0]


def test_the_test_split_is_loaded_in_exactly_one_place() -> None:
    package = Path(fraud.__file__).parent
    callers = {
        path.name: path.read_text(encoding="utf-8").count("load_test")
        for path in package.glob("*.py")
        if path.name != "data.py"
    }
    # final_eval: one import and one call of load_test_with_duplicates, nothing else anywhere.
    assert {name: count for name, count in callers.items() if count} == {"final_eval.py": 2}
    assert "allow_test=True" in (package / "final_eval.py").read_text(encoding="utf-8")
