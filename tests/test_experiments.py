import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from fraud import experiments, report
from fraud.config import FINAL_CONFIG_NAME, FINAL_MODEL_NAME, ExperimentPaths
from fraud.data import split_xy, time_based_split
from fraud.experiments import (
    ExperimentConfig,
    grid_configs,
    run_configs,
    select_configs,
    tuning_configs,
    tuning_space,
)
from fraud.selection import STRATEGY_ORDER, rank_by_rule, select_best, strategy_label
from fraud.train import RESULT_COLUMNS, load_results
from tests.conftest import make_synthetic_transactions

N_BOOT = 20
SUBSET = [ExperimentConfig("logreg", "none"), ExperimentConfig("logreg", "smote", 0.1)]


@pytest.fixture(scope="module")
def train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = time_based_split(make_synthetic_transactions(n_rows=6_000, fraud_rate=0.05))
    return splits["train"], splits["val"]


def make_paths(root: Path) -> ExperimentPaths:
    return ExperimentPaths(
        results=root / "reports" / "results.csv",
        scores_dir=root / "scores",
        figures_dir=root / "reports" / "figures",
        summary=root / "reports" / "experiments_summary.md",
        models_dir=root / "models",
    )


@pytest.fixture
def paths(tmp_path: Path) -> ExperimentPaths:
    return make_paths(tmp_path)


def silent(message: str) -> None:
    pass


def forbid_training(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("run_experiment must not be called for finished configurations")


# --- Grid definition and subset selection ----------------------------------------------


def test_grid_has_36_unique_configurations_in_fixed_order() -> None:
    configs = grid_configs()
    assert len(configs) == 36
    assert len({config.id for config in configs}) == 36
    assert [config.model for config in configs[:12]] == ["logreg"] * 12
    assert [(c.strategy, c.sampling_ratio) for c in configs[:12]] == STRATEGY_ORDER
    assert len(STRATEGY_ORDER) == 12
    assert all(config.stage == "grid" for config in configs)
    assert grid_configs() == configs


def test_resampling_strategies_use_both_ratios() -> None:
    ratios: dict[str, set[float | None]] = {}
    for strategy, ratio in STRATEGY_ORDER:
        ratios.setdefault(strategy, set()).add(ratio)
    assert ratios.pop("none") == {None}
    assert ratios.pop("class_weight") == {None}
    assert set(ratios) == {"undersample", "smote", "borderline_smote", "adasyn", "smote_tomek"}
    assert all(values == {0.1, 0.5} for values in ratios.values())


def test_select_configs_filters() -> None:
    grid = grid_configs()
    subset = select_configs(grid, models=["lightgbm"], strategies=["smote"])
    assert [(c.model, c.strategy, c.sampling_ratio) for c in subset] == [
        ("lightgbm", "smote", 0.1),
        ("lightgbm", "smote", 0.5),
    ]
    assert len(select_configs(grid, models=["logreg", "lightgbm"])) == 24
    assert len(select_configs(grid, strategies=["none"])) == 3
    one_ratio = select_configs(grid, models=["logreg"], strategies=["none", "smote"], ratios=[0.1])
    assert [(c.strategy, c.sampling_ratio) for c in one_ratio] == [("none", None), ("smote", 0.1)]
    assert select_configs(grid) == grid


def test_config_id_distinguishes_hyperparameters() -> None:
    default = ExperimentConfig("logreg", "none")
    tuned = ExperimentConfig("logreg", "none", model_params={"C": 0.1}, stage="tuning")
    assert default.id != tuned.id
    assert default.id == ExperimentConfig("logreg", "none").id
    assert default.id.startswith("logreg__none__na__")
    assert ExperimentConfig("logreg", "smote", 0.1).id.startswith("logreg__smote__0.1__")


# --- Running, resuming, failing --------------------------------------------------------


def test_run_configs_writes_rows_and_validation_scores(
    train_val: tuple[pd.DataFrame, pd.DataFrame], paths: ExperimentPaths
) -> None:
    train, val = train_val
    outcome = run_configs(SUBSET, train, val, paths, n_boot=N_BOOT, log=silent)
    assert outcome == {"ran": [c.id for c in SUBSET], "skipped": [], "failed": []}

    results = load_results(paths.results)
    assert list(results.columns) == RESULT_COLUMNS
    assert list(results["config_id"]) == [c.id for c in SUBSET]
    assert list(results["stage"]) == ["grid", "grid"]
    assert (results["cv_folds"] == 3).all()
    assert all(len(json.loads(folds)) == 3 for folds in results["cv_pr_auc_folds"])
    for config in SUBSET:
        scores = np.load(paths.scores_dir / f"{config.id}.npy")
        assert scores.shape == (len(val),)  # one score per validation row, never resampled


def test_finished_configurations_are_skipped_on_resume(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
    paths: ExperimentPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, val = train_val
    run_configs(SUBSET[:1], train, val, paths, n_boot=N_BOOT, log=silent)
    first_timestamp = load_results(paths.results)["timestamp"].item()

    # An interrupted run is continued: only the missing configuration is trained.
    calls: list[str] = []
    original = experiments.run_experiment

    def counting(model: str, strategy: str, *args: Any, **kwargs: Any) -> Any:
        calls.append(strategy)
        return original(model, strategy, *args, **kwargs)

    monkeypatch.setattr(experiments, "run_experiment", counting)
    outcome = run_configs(SUBSET, train, val, paths, n_boot=N_BOOT, log=silent)
    assert calls == ["smote"]
    assert outcome["skipped"] == [SUBSET[0].id]
    assert outcome["ran"] == [SUBSET[1].id]
    results = load_results(paths.results)
    assert len(results) == 2
    assert results["timestamp"].iloc[0] == first_timestamp

    # Everything is finished now: nothing is trained again.
    monkeypatch.setattr(experiments, "run_experiment", forbid_training)
    outcome = run_configs(SUBSET, train, val, paths, n_boot=N_BOOT, log=silent)
    assert outcome == {"ran": [], "skipped": [c.id for c in SUBSET], "failed": []}
    assert len(load_results(paths.results)) == 2


def test_force_reruns_and_replaces_rows(
    train_val: tuple[pd.DataFrame, pd.DataFrame], paths: ExperimentPaths
) -> None:
    train, val = train_val
    run_configs(SUBSET[:1], train, val, paths, n_boot=N_BOOT, log=silent)
    before = load_results(paths.results)
    outcome = run_configs(SUBSET[:1], train, val, paths, force=True, n_boot=N_BOOT, log=silent)
    after = load_results(paths.results)
    assert outcome["ran"] == [SUBSET[0].id]
    assert len(after) == 1  # replaced, not appended
    assert after["pr_auc"].item() == pytest.approx(before["pr_auc"].item())  # reproducible


def test_missing_scores_file_means_not_finished(
    train_val: tuple[pd.DataFrame, pd.DataFrame], paths: ExperimentPaths
) -> None:
    train, val = train_val
    run_configs(SUBSET[:1], train, val, paths, n_boot=N_BOOT, log=silent)
    (paths.scores_dir / f"{SUBSET[0].id}.npy").unlink()
    outcome = run_configs(SUBSET[:1], train, val, paths, n_boot=N_BOOT, log=silent)
    assert outcome["ran"] == [SUBSET[0].id]


def test_one_failing_configuration_does_not_stop_the_others(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
    paths: ExperimentPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, val = train_val
    original = experiments.run_experiment

    def failing(model: str, strategy: str, *args: Any, **kwargs: Any) -> Any:
        if strategy == "none":
            raise RuntimeError("sampler exploded")
        return original(model, strategy, *args, **kwargs)

    monkeypatch.setattr(experiments, "run_experiment", failing)
    messages: list[str] = []
    outcome = run_configs(SUBSET, train, val, paths, n_boot=N_BOOT, log=messages.append)
    assert outcome["failed"] == [SUBSET[0].id]
    assert outcome["ran"] == [SUBSET[1].id]
    assert any("FAILED" in message and "sampler exploded" in message for message in messages)
    assert list(load_results(paths.results)["config_id"]) == [SUBSET[1].id]


# --- CLI -------------------------------------------------------------------------------


@pytest.fixture
def cli(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
    paths: ExperimentPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> ExperimentPaths:
    """The CLI wired to synthetic data and a temporary output directory."""
    monkeypatch.setattr(experiments, "load_train_val", lambda: train_val)
    return paths


def test_cli_runs_only_the_requested_subset(cli: ExperimentPaths) -> None:
    arguments = ["--stage", "grid", "--model", "logreg", "--strategy", "none", "undersample"]
    experiments.main([*arguments, "--n-boot", str(N_BOOT)], paths=cli)
    results = load_results(cli.results)
    assert set(results["model"]) == {"logreg"}
    assert sorted(zip(results["strategy"], results["sampling_ratio"].fillna(-1), strict=True)) == [
        ("none", -1),
        ("undersample", 0.1),
        ("undersample", 0.5),
    ]

    experiments.main(
        ["--stage", "grid", "--model", "logreg", "--strategy", "smote", "--sampling-ratio", "0.5"]
        + ["--n-boot", str(N_BOOT)],
        paths=cli,
    )
    results = load_results(cli.results)
    assert len(results) == 4
    assert results.iloc[-1][["strategy", "sampling_ratio"]].tolist() == ["smote", 0.5]


def test_cli_resumes_without_retraining(
    cli: ExperimentPaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--stage", "grid", "--model", "logreg", "--strategy", "none", "--n-boot", "20"]
    experiments.main(arguments, paths=cli)
    monkeypatch.setattr(experiments, "run_experiment", forbid_training)
    experiments.main(arguments, paths=cli)
    assert "already done, skipped" in capsys.readouterr().out
    assert len(load_results(cli.results)) == 1


def test_cli_stops_before_tuning_when_the_grid_is_incomplete(
    cli: ExperimentPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    experiments.main(["--model", "logreg", "--strategy", "none", "--n-boot", "20"], paths=cli)
    output = capsys.readouterr().out
    assert "1/36" in output
    assert "== Tuning ==" not in output
    assert not cli.summary.exists()
    with pytest.raises(SystemExit, match="1/36"):
        experiments.main(["--stage", "report"], paths=cli)


def test_cli_rejects_unknown_names(cli: ExperimentPaths) -> None:
    with pytest.raises(SystemExit):
        experiments.main(["--model", "svm"], paths=cli)


# --- Selection rule and tuning ---------------------------------------------------------


def fake_results(rows: list[tuple[str, str, float | None, float]]) -> pd.DataFrame:
    """Results with only the columns the selection rule and tuning need."""
    frame = pd.DataFrame(rows, columns=["model", "strategy", "sampling_ratio", "cv_pr_auc_mean"])
    frame["stage"] = "grid"
    frame["config_id"] = [
        ExperimentConfig(model, strategy, ratio).id for model, strategy, ratio, _ in rows
    ]
    return frame


def test_select_best_takes_the_highest_cv_mean() -> None:
    results = fake_results([("logreg", "none", None, 0.60), ("lightgbm", "smote", 0.1, 0.80)])
    assert select_best(results)["model"] == "lightgbm"


def test_select_best_breaks_near_ties_toward_the_simpler_strategy() -> None:
    results = fake_results(
        [
            ("lightgbm", "smote", 0.1, 0.800),
            ("lightgbm", "class_weight", None, 0.797),
            ("lightgbm", "none", None, 0.796),
            ("logreg", "none", None, 0.700),
        ]
    )
    best = select_best(results)
    assert (best["model"], best["strategy"]) == ("lightgbm", "none")
    # Outside the tolerance the higher mean wins, however complex.
    assert select_best(results, tolerance=0.001)["strategy"] == "smote"
    assert select_best(results, tolerance=0.0035)["strategy"] == "class_weight"


def test_rank_by_rule_orders_every_configuration_once() -> None:
    results = fake_results(
        [
            ("logreg", "none", None, 0.70),
            ("lightgbm", "smote", 0.1, 0.80),
            ("lightgbm", "none", None, 0.798),
            ("random_forest", "none", None, 0.75),
        ]
    )
    ranked = rank_by_rule(results)
    assert list(ranked.index) == [1, 2, 3, 4]
    assert list(zip(ranked["model"], ranked["strategy"], strict=True)) == [
        ("lightgbm", "none"),
        ("lightgbm", "smote"),
        ("random_forest", "none"),
        ("logreg", "none"),
    ]


def test_tuning_space_is_small_and_deterministic() -> None:
    assert tuning_space("lightgbm") == tuning_space("lightgbm")
    assert len(tuning_space("logreg")) == 5
    assert len(tuning_space("lightgbm")) == 12
    assert len(tuning_space("random_forest")) == 8
    assert {"num_leaves", "learning_rate", "n_estimators", "min_child_samples"} == set(
        tuning_space("lightgbm")[0]
    )
    with pytest.raises(ValueError, match="No tuning space"):
        tuning_space("svm")


def test_tuning_configs_cover_the_two_best_families_with_their_best_strategy() -> None:
    results = fake_results(
        [
            ("logreg", "none", None, 0.60),
            ("logreg", "class_weight", None, 0.70),
            ("random_forest", "none", None, 0.50),
            ("lightgbm", "none", None, 0.75),
            ("lightgbm", "smote", 0.1, 0.80),
        ]
    )
    configs = tuning_configs(results)
    assert {config.model for config in configs} == {"lightgbm", "logreg"}
    assert all(config.stage == "tuning" for config in configs)
    lightgbm = [config for config in configs if config.model == "lightgbm"]
    logreg = [config for config in configs if config.model == "logreg"]
    assert {(c.strategy, c.sampling_ratio) for c in lightgbm} == {("smote", 0.1)}
    assert {c.strategy for c in logreg} == {"class_weight"}
    assert (len(lightgbm), len(logreg)) == (12, 5)
    assert not {config.id for config in configs} & set(results["config_id"])
    assert len({config.id for config in configs}) == len(configs)


# --- Report ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def finished(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> ExperimentPaths:
    """A complete logistic regression grid plus its tuning runs, on synthetic data."""
    train, val = train_val
    paths = make_paths(tmp_path_factory.mktemp("finished"))
    run_configs(
        select_configs(grid_configs(), ["logreg"]), train, val, paths, n_boot=N_BOOT, log=silent
    )
    tuning = tuning_configs(load_results(paths.results))
    run_configs(tuning, train, val, paths, n_boot=N_BOOT, log=silent)
    report.build_report(load_results(paths.results), train, val, paths, n_boot=N_BOOT)
    return paths


def test_report_creates_figures_summary_and_model(finished: ExperimentPaths) -> None:
    results = load_results(finished.results)
    assert (results["stage"] == "grid").sum() == 12
    assert (results["stage"] == "tuning").sum() == 5
    for name in ("exp_pr_auc_by_config", "exp_pr_curves_best_per_model", "exp_cv_pr_auc_heatmap"):
        assert (finished.figures_dir / f"{name}.png").stat().st_size > 0
    summary = finished.summary.read_text(encoding="utf-8")
    for heading in (
        "## Selected configuration",
        "## Ranking of the 12 grid configurations",
        "## Resampling against class weights and no resampling",
        "## Paired comparisons on validation",
        "## Hyperparameter tuning",
    ):
        assert heading in summary
    assert strategy_label("smote_tomek", 0.5) in summary


def test_saved_final_model_is_the_selected_configuration(
    finished: ExperimentPaths, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, val = train_val
    config = json.loads((finished.models_dir / FINAL_CONFIG_NAME).read_text(encoding="utf-8"))
    results = load_results(finished.results)
    assert config["config_id"] == select_best(results)["config_id"]
    assert config["threshold"] is None  # chosen in Phase 4
    assert "Time" not in config["features"]
    assert config["val_pr_auc"] == pytest.approx(config["val_pr_auc_in_results"])

    pipeline = joblib.load(finished.models_dir / FINAL_MODEL_NAME)
    val_x, _ = split_xy(val)
    scores = pipeline.predict_proba(val_x)[:, 1]
    assert scores.shape == (len(val),)
    saved = np.load(finished.scores_dir / f"{config['config_id']}.npy")
    np.testing.assert_allclose(scores, saved)


def test_paired_comparisons_cover_both_references(
    finished: ExperimentPaths, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, val = train_val
    grid = load_results(finished.results).query("stage == 'grid'")
    top = rank_by_rule(grid).head(3)
    table = report.paired_comparisons(top, grid, val["Class"], finished.scores_dir, n_boot=N_BOOT)
    baseline_id = ExperimentConfig("logreg", "none").id
    assert (
        baseline_id not in set(table["config_id"]) or len(table) < 6
    )  # never compared with itself
    assert set(table["versus"]) <= {
        "same model, no resampling",
        "baseline (logistic regression, no resampling)",
    }
    assert (table["ci_low"] <= table["difference"]).all()
    assert (table["difference"] <= table["ci_high"]).all()
    assert np.allclose(table["difference"], table["a"] - table["b"])


def test_summary_keeps_hand_written_conclusions(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    report.write_summary(path, "first tables")
    text = path.read_text(encoding="utf-8").replace(
        "_To be written after reviewing the results below._", "- SMOTE did not help."
    )
    path.write_text(text, encoding="utf-8")

    report.write_summary(path, "second tables")
    updated = path.read_text(encoding="utf-8")
    assert "- SMOTE did not help." in updated
    assert "second tables" in updated
    assert "first tables" not in updated
    assert updated.count(report.GENERATED_START) == 1

    path.write_text("no markers here", encoding="utf-8")
    with pytest.raises(ValueError, match="markers"):
        report.write_summary(path, "tables")


def test_markdown_table() -> None:
    table = report.markdown_table(pd.DataFrame({"Model": ["A", "B"], "PR-AUC": ["0.5", "0.6"]}))
    assert table.splitlines() == ["| Model | PR-AUC |", "|---|---|", "| A | 0.5 |", "| B | 0.6 |"]
