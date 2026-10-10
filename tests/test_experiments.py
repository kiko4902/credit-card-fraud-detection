import json
from collections.abc import Iterator
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
    run_configs,
    select_configs,
    stage1_configs,
    stage2_configs,
    stage3_configs,
    tuning_space,
)
from fraud.models import MODELS
from fraud.selection import (
    STAGE1_STRATEGIES,
    STAGE2_STRATEGIES,
    STRATEGY_ORDER,
    rank_by_rule,
    rank_classifiers,
    select_best,
    strategy_label,
)
from fraud.train import RESULT_COLUMNS, load_results
from tests.conftest import make_synthetic_transactions

N_BOOT = 20
BOOT = ["--n-boot", str(N_BOOT)]
SUBSET = [ExperimentConfig("logreg", "none"), ExperimentConfig("logreg", "smote", 0.1)]
# Fast stand-ins for the full design: 3 classifiers screened, 2 go on to stage 2.
FAST_MODELS = ["logreg", "gaussian_nb", "decision_tree"]


@pytest.fixture(scope="module")
def train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = time_based_split(make_synthetic_transactions(n_rows=6_000, fraud_rate=0.05))
    return splits["train"], splits["val"]


def make_paths(root: Path) -> ExperimentPaths:
    return ExperimentPaths(
        results=root / "reports" / "results.csv",
        scores_dir=root / "scores",
        cache_dir=root / "cache",
        figures_dir=root / "reports" / "figures",
        summary=root / "reports" / "experiments_summary.md",
        models_dir=root / "models",
    )


@pytest.fixture
def paths(tmp_path: Path) -> ExperimentPaths:
    return make_paths(tmp_path)


def shrink_design(patch: pytest.MonkeyPatch, train_val: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    patch.setattr(experiments, "STAGE1_MODELS", FAST_MODELS)
    patch.setattr(experiments, "STAGE2_CLASSIFIERS", 2)
    patch.setattr(experiments, "load_train_val", lambda: train_val)


@pytest.fixture
def cli(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
    paths: ExperimentPaths,
    monkeypatch: pytest.MonkeyPatch,
) -> ExperimentPaths:
    """The CLI wired to synthetic data, a small design and a temporary output directory."""
    shrink_design(monkeypatch, train_val)
    return paths


def silent(message: str) -> None:
    pass


def forbid_training(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("run_experiment must not be called for finished configurations")


# --- Stage definitions -----------------------------------------------------------------


def test_stage1_screens_nine_classifiers_with_three_strategies() -> None:
    configs = stage1_configs()
    assert len(configs) == 24  # 9 x 3 minus class weights for Gaussian NB, k-NN and MLP
    assert len({config.id for config in configs}) == 24
    assert {config.model for config in configs} == set(MODELS)
    assert all(config.stage == "stage1" and config.model_params == {} for config in configs)
    by_model: dict[str, list[tuple[str, float | None]]] = {}
    for config in configs:
        by_model.setdefault(config.model, []).append((config.strategy, config.sampling_ratio))
    assert by_model["logreg"] == [("none", None), ("class_weight", None), ("smote", 0.1)]
    for model in ("gaussian_nb", "knn", "mlp"):
        assert by_model[model] == [("none", None), ("smote", 0.1)]
    assert stage1_configs() == configs


def test_strategy_lists_match_the_design() -> None:
    assert STAGE1_STRATEGIES == [("none", None), ("class_weight", None), ("smote", 0.1)]
    assert STAGE2_STRATEGIES == [
        ("smote", 0.5),
        ("smote", 1.0),
        ("smote_tomek", 0.1),
        ("smote_tomek", 0.5),
        ("smote_enn", 0.1),
        ("smote_enn", 0.5),
        ("safe_level_smote", 0.1),
        ("safe_level_smote", 0.5),
        ("polynom_fit_smote", 0.1),
        ("polynom_fit_smote", 0.5),
    ]
    assert [*STAGE1_STRATEGIES, *STAGE2_STRATEGIES] == STRATEGY_ORDER


def fake_results(
    rows: list[tuple[str, str, float | None, float]], stage: str = "stage1"
) -> pd.DataFrame:
    """Results with only the columns that ranking, selection and staging need."""
    frame = pd.DataFrame(rows, columns=["model", "strategy", "sampling_ratio", "cv_pr_auc_mean"])
    frame["stage"] = stage
    frame["config_id"] = [
        ExperimentConfig(model, strategy, ratio).id for model, strategy, ratio, _ in rows
    ]
    return frame


SCREENING = fake_results(
    [
        ("logreg", "none", None, 0.56),
        ("logreg", "class_weight", None, 0.66),
        ("logreg", "smote", 0.1, 0.74),
        ("gaussian_nb", "none", None, 0.10),
        ("gaussian_nb", "smote", 0.1, 0.12),
        ("knn", "none", None, 0.60),
        ("random_forest", "none", None, 0.78),
        ("random_forest", "class_weight", None, 0.75),
        ("xgboost", "none", None, 0.80),
        ("lightgbm", "none", None, 0.30),
        ("lightgbm", "class_weight", None, 0.77),
    ]
)


def test_classifiers_are_ranked_by_their_best_strategy() -> None:
    ranked = rank_classifiers(SCREENING)
    assert list(ranked["model"]) == [
        "xgboost",
        "random_forest",
        "lightgbm",
        "logreg",
        "knn",
        "gaussian_nb",
    ]
    assert list(ranked.index) == [1, 2, 3, 4, 5, 6]
    assert ranked.loc[3, "strategy"] == "class_weight"  # LightGBM's best, not its first
    assert ranked.loc[4, "cv_pr_auc_mean"] == 0.74


def test_stage2_takes_the_top_four_classifiers_and_the_whole_smote_family() -> None:
    configs = stage2_configs(SCREENING)
    assert len(configs) == 40
    assert [config.model for config in configs[::10]] == [
        "xgboost",
        "random_forest",
        "lightgbm",
        "logreg",
    ]
    assert [(c.strategy, c.sampling_ratio) for c in configs[:10]] == STAGE2_STRATEGIES
    assert all(config.stage == "stage2" for config in configs)
    assert not {config.id for config in configs} & set(
        SCREENING["config_id"]
    )  # SMOTE 0.1 not repeated


def test_tuning_space_is_small_and_deterministic() -> None:
    for model in MODELS:
        space = tuning_space(model)
        assert 5 <= len(space) <= 12
        assert space == tuning_space(model)
    assert {"num_leaves", "learning_rate", "n_estimators", "min_child_samples"} == set(
        tuning_space("lightgbm")[0]
    )
    with pytest.raises(ValueError, match="No tuning space"):
        tuning_space("svm")


def test_stage3_tunes_the_two_best_classifiers_with_their_best_strategy() -> None:
    stage2 = fake_results(
        [("xgboost", "smote_enn", 0.5, 0.83), ("logreg", "smote", 0.5, 0.72)], "stage2"
    )
    configs = stage3_configs(pd.concat([SCREENING, stage2], ignore_index=True))
    assert {config.model for config in configs} == {"xgboost", "random_forest"}
    assert all(config.stage == "stage3" for config in configs)
    xgboost = [config for config in configs if config.model == "xgboost"]
    forest = [config for config in configs if config.model == "random_forest"]
    assert {(c.strategy, c.sampling_ratio) for c in xgboost} == {("smote_enn", 0.5)}
    assert {(c.strategy, c.sampling_ratio) for c in forest} == {("none", None)}
    assert (len(xgboost), len(forest)) == (12, 8)
    assert len({config.id for config in configs}) == len(configs)


def test_select_configs_filters() -> None:
    configs = stage1_configs()
    subset = select_configs(configs, models=["lightgbm"], strategies=["smote"])
    assert [(c.model, c.strategy, c.sampling_ratio) for c in subset] == [("lightgbm", "smote", 0.1)]
    assert len(select_configs(configs, models=["logreg", "knn"])) == 5
    assert len(select_configs(configs, strategies=["none"])) == 9
    stage2 = stage2_configs(SCREENING)
    picked = select_configs(
        stage2, models=["xgboost"], strategies=["smote", "smote_enn"], ratios=[0.5]
    )
    assert [(c.strategy, c.sampling_ratio) for c in picked] == [("smote", 0.5), ("smote_enn", 0.5)]
    assert select_configs(configs) == configs


def test_config_id_distinguishes_hyperparameters() -> None:
    default = ExperimentConfig("logreg", "none")
    tuned = ExperimentConfig("logreg", "none", model_params={"C": 0.1}, stage="stage3")
    assert default.id != tuned.id
    assert default.id == ExperimentConfig("logreg", "none").id
    assert default.id.startswith("logreg__none__na__")
    assert ExperimentConfig("logreg", "smote", 0.1).id.startswith("logreg__smote__0.1__")


# --- Selection rule --------------------------------------------------------------------


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
    assert list(results["stage"]) == ["stage1", "stage1"]
    assert (results["cv_folds"] == 3).all()
    assert all(len(json.loads(folds)) == 3 for folds in results["cv_pr_auc_folds"])
    assert (results[["resample_time_s", "fit_time_s", "predict_time_s"]] >= 0).all().all()
    for config in SUBSET:
        scores = np.load(paths.scores_dir / f"{config.id}.npy")
        assert scores.shape == (len(val),)  # one score per validation row, never resampled
    assert any(paths.cache_dir.iterdir())  # the SMOTE run filled the resampling cache


def test_results_do_not_depend_on_the_cache(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path: Path
) -> None:
    train, val = train_val
    cached, uncached = make_paths(tmp_path / "cached"), make_paths(tmp_path / "uncached")
    run_configs(SUBSET, train, val, cached, n_boot=N_BOOT, log=silent)
    run_configs(SUBSET, train, val, uncached, n_boot=N_BOOT, log=silent, use_cache=False)
    assert not uncached.cache_dir.exists()
    columns = [
        "config_id",
        "pr_auc",
        "pr_auc_ci_low",
        "pr_auc_ci_high",
        "roc_auc",
        "recall",
        "cv_pr_auc_mean",
        "cv_pr_auc_folds",
    ]
    pd.testing.assert_frame_equal(
        load_results(cached.results)[columns], load_results(uncached.results)[columns]
    )
    for config in SUBSET:
        np.testing.assert_array_equal(
            np.load(cached.scores_dir / f"{config.id}.npy"),
            np.load(uncached.scores_dir / f"{config.id}.npy"),
        )


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


# --- CLI: subsets, stages, resume ------------------------------------------------------


def test_cli_runs_only_the_requested_subset(cli: ExperimentPaths) -> None:
    experiments.main(
        ["--stage", "1", "--model", "logreg", "--strategy", "none", "smote", *BOOT], paths=cli
    )
    results = load_results(cli.results)
    assert set(results["model"]) == {"logreg"}
    assert list(zip(results["strategy"], results["sampling_ratio"].fillna(-1), strict=True)) == [
        ("none", -1),
        ("smote", 0.1),
    ]
    assert set(results["stage"]) == {"stage1"}

    experiments.main(["--stage", "1", "--model", "gaussian_nb", *BOOT], paths=cli)
    results = load_results(cli.results)
    assert len(results) == 4
    assert list(results["model"].iloc[2:]) == ["gaussian_nb", "gaussian_nb"]


def test_cli_resumes_without_retraining(
    cli: ExperimentPaths, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arguments = ["--stage", "1", "--model", "logreg", "--strategy", "none", *BOOT]
    experiments.main(arguments, paths=cli)
    monkeypatch.setattr(experiments, "run_experiment", forbid_training)
    experiments.main(arguments, paths=cli)
    assert "already done, skipped" in capsys.readouterr().out
    assert len(load_results(cli.results)) == 1


def test_cli_does_not_start_a_stage_before_the_previous_one_is_complete(
    cli: ExperimentPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    experiments.main(["--model", "logreg", "--strategy", "none", *BOOT], paths=cli)
    output = capsys.readouterr().out
    assert "Stage 1 is incomplete" in output
    assert "Stage 2" not in output.replace("stopping before stage 2", "")
    assert not cli.summary.exists()
    for stage in ("2", "3", "report"):
        with pytest.raises(SystemExit, match=r"Stage 1 is incomplete \(1/8 done\)"):
            experiments.main(["--stage", stage, *BOOT], paths=cli)


def test_cli_estimate_lists_configurations_without_writing_results(
    cli: ExperimentPaths, capsys: pytest.CaptureFixture[str]
) -> None:
    experiments.main(["--stage", "1", "--estimate", "--model", "logreg"], paths=cli)
    output = capsys.readouterr().out
    assert "3 of 8 configurations" in output
    assert "Estimated total" in output
    assert output.count("logreg__") == 3
    assert not cli.results.exists()
    with pytest.raises(SystemExit, match="--estimate needs --stage"):
        experiments.main(["--estimate"], paths=cli)


def test_cli_rejects_unknown_names(cli: ExperimentPaths) -> None:
    with pytest.raises(SystemExit):
        experiments.main(["--model", "svm"], paths=cli)
    with pytest.raises(SystemExit):
        experiments.main(["--strategy", "adasyn"], paths=cli)


# --- Full staged run and report --------------------------------------------------------


@pytest.fixture(scope="module")
def finished(
    train_val: tuple[pd.DataFrame, pd.DataFrame], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[ExperimentPaths]:
    """All stages and the report, run through the CLI on synthetic data with a small design."""
    paths = make_paths(tmp_path_factory.mktemp("finished"))
    with pytest.MonkeyPatch.context() as patch:
        shrink_design(patch, train_val)
        experiments.main(BOOT, paths=paths)
        yield paths


def test_all_stages_are_recorded(finished: ExperimentPaths) -> None:
    results = load_results(finished.results)
    counts = results["stage"].value_counts()
    assert counts["stage1"] == 8  # 3 classifiers x 3 strategies, Gaussian NB without class weights
    assert counts["stage2"] == 20  # 2 classifiers x 10 SMOTE-family configurations
    assert counts["stage3"] >= 10
    assert results["config_id"].is_unique
    stage2 = results[results["stage"] == "stage2"]
    assert stage2["model"].nunique() == 2
    assert set(zip(stage2["strategy"], stage2["sampling_ratio"], strict=True)) == set(
        STAGE2_STRATEGIES
    )
    assert set(results.loc[results["stage"] == "stage3", "model"]) <= set(stage2["model"])


def test_rerunning_everything_trains_nothing(
    finished: ExperimentPaths, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = load_results(finished.results)
    monkeypatch.setattr(experiments, "run_experiment", forbid_training)
    experiments.main(BOOT, paths=finished)
    pd.testing.assert_frame_equal(load_results(finished.results), before)


def test_report_creates_figures_summary_and_model(finished: ExperimentPaths) -> None:
    for name in (
        "exp_stage1_ranking",
        "exp_stage2_cv_heatmap",
        "exp_pr_curves_best_per_model",
        "exp_pr_auc_vs_fit_time",
    ):
        assert (finished.figures_dir / f"{name}.png").stat().st_size > 0
    summary = finished.summary.read_text(encoding="utf-8")
    for heading in (
        "## Conclusions",
        "## Comparison with the thesis",
        "## Selected configuration",
        "## Stage 1: screening of 3 classifiers",
        "## Stage 2: SMOTE family for the top 2 classifiers",
        "### Each variant against plain SMOTE at the same ratio",
        "### Oversampling against no resampling and class weights",
        "## Paired comparisons on validation",
        "## Stage 3: hyperparameter tuning",
    ):
        assert heading in summary
    assert strategy_label("safe_level_smote", 0.5) in summary
    assert strategy_label("polynom_fit_smote", 0.1) in summary


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
    assert config["model_file_bytes"] == (finished.models_dir / FINAL_MODEL_NAME).stat().st_size

    pipeline = joblib.load(finished.models_dir / FINAL_MODEL_NAME)
    val_x, _ = split_xy(val)
    scores = pipeline.predict_proba(val_x)[:, 1]
    assert scores.shape == (len(val),)
    np.testing.assert_allclose(scores, np.load(finished.scores_dir / f"{config['config_id']}.npy"))


def test_paired_comparisons_cover_both_references(
    finished: ExperimentPaths, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    _, val = train_val
    screened = load_results(finished.results).query("stage in ['stage1', 'stage2']")
    top = rank_by_rule(screened).head(3)
    table = report.paired_comparisons(
        top, screened, val["Class"], finished.scores_dir, n_boot=N_BOOT
    )
    assert 1 <= len(table) <= 6
    assert set(table["versus"]) <= {
        "same classifier, no resampling",
        "baseline (logistic regression, no resampling)",
    }
    assert (table["ci_low"] <= table["difference"]).all()
    assert (table["difference"] <= table["ci_high"]).all()
    assert np.allclose(table["difference"], table["a"] - table["b"])


def test_summary_keeps_hand_written_sections(tmp_path: Path) -> None:
    path = tmp_path / "summary.md"
    report.write_summary(path, "first tables")
    text = path.read_text(encoding="utf-8").replace(
        "_To be written after reviewing the results below._", "- SMOTE did not help.", 1
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
