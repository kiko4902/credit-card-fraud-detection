"""Phase 3 driver: the experiment grid, hyperparameter tuning and the report.

    python -m fraud.experiments                                   # grid, tuning, report
    python -m fraud.experiments --stage grid --model lightgbm --strategy smote
    python -m fraud.experiments --stage report

Every configuration writes or replaces its own row in ``reports/results.csv``, so an
interrupted run continues where it stopped. Use ``--force`` to re-run finished ones.
The test split is never loaded here.
"""

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler

from fraud.config import N_BOOTSTRAP, RANDOM_SEED, TUNING_FAMILIES, ExperimentPaths
from fraud.data import load_train_val
from fraud.models import MODELS, STRATEGIES, build_pipeline
from fraud.report import build_report
from fraud.selection import MODEL_ORDER, STRATEGY_ORDER, select_best
from fraud.train import config_id, load_results, params_json, run_experiment, save_result

STAGES: list[str] = ["all", "grid", "tune", "report"]


@dataclass(frozen=True)
class ExperimentConfig:
    """One model / strategy / hyperparameter setting."""

    model: str
    strategy: str
    sampling_ratio: float | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    stage: str = "grid"

    @property
    def id(self) -> str:
        pipeline = build_pipeline(self.model, self.strategy, self.sampling_ratio, self.model_params)
        return config_id(self.model, self.strategy, self.sampling_ratio, params_json(pipeline))


def grid_configs() -> list[ExperimentConfig]:
    """The 36 configurations of the strategy comparison, in a fixed order."""
    return [
        ExperimentConfig(model, strategy, ratio)
        for model in MODEL_ORDER
        for strategy, ratio in STRATEGY_ORDER
    ]


def select_configs(
    configs: Sequence[ExperimentConfig],
    models: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    ratios: Sequence[float] | None = None,
) -> list[ExperimentConfig]:
    """Keep the configurations matching every given filter (``None`` = no filter).

    A ratio filter only applies to strategies that resample, so
    ``strategies=["none", "smote"], ratios=[0.1]`` keeps ``none`` and ``smote 0.1``.
    """
    return [
        config
        for config in configs
        if (models is None or config.model in models)
        and (strategies is None or config.strategy in strategies)
        and (ratios is None or config.sampling_ratio is None or config.sampling_ratio in ratios)
    ]


def scores_path(identifier: str, scores_dir: Path) -> Path:
    return scores_dir / f"{identifier}.npy"


def is_done(config: ExperimentConfig, results: pd.DataFrame, scores_dir: Path) -> bool:
    """A configuration is finished when it has a results row and its saved validation scores."""
    identifier = config.id
    return (
        bool((results["config_id"] == identifier).any())
        and scores_path(identifier, scores_dir).exists()
    )


def run_configs(
    configs: Sequence[ExperimentConfig],
    train: pd.DataFrame,
    val: pd.DataFrame,
    paths: ExperimentPaths,
    force: bool = False,
    n_boot: int = N_BOOTSTRAP,
    log: Callable[[str], None] = print,
) -> dict[str, list[str]]:
    """Run each configuration unless it is already finished; return ids by outcome."""
    outcome: dict[str, list[str]] = {"ran": [], "skipped": [], "failed": []}
    paths.scores_dir.mkdir(parents=True, exist_ok=True)
    for number, config in enumerate(configs, start=1):
        identifier = config.id
        prefix = f"[{number}/{len(configs)}] {identifier}"
        if not force and is_done(config, load_results(paths.results), paths.scores_dir):
            outcome["skipped"].append(identifier)
            log(f"{prefix}: already done, skipped")
            continue
        start = time.perf_counter()
        try:
            result = run_experiment(
                config.model,
                config.strategy,
                config.sampling_ratio,
                train,
                val,
                n_boot=n_boot,
                model_params=config.model_params,
                stage=config.stage,
            )
        except Exception as error:  # keep going: one failing sampler must not stop the grid
            outcome["failed"].append(identifier)
            log(f"{prefix}: FAILED with {type(error).__name__}: {error}")
            continue
        np.save(scores_path(identifier, paths.scores_dir), result.val_scores)
        save_result(result.row, paths.results)
        outcome["ran"].append(identifier)
        row = result.row
        log(
            f"{prefix}: CV PR-AUC {row['cv_pr_auc_mean']:.4f} | val PR-AUC {row['pr_auc']:.4f} "
            f"[{row['pr_auc_ci_low']:.4f}, {row['pr_auc_ci_high']:.4f}] "
            f"| {time.perf_counter() - start:.0f} s"
        )
    return outcome


def tuning_space(model: str) -> list[dict[str, Any]]:
    """The fixed, small set of hyperparameter settings tried for ``model``."""
    if model == "logreg":
        return [{"C": value} for value in (0.001, 0.01, 0.1, 10.0, 100.0)]
    if model == "lightgbm":
        space = {
            "num_leaves": [7, 15, 31, 63],
            "learning_rate": [0.02, 0.05, 0.1],
            "n_estimators": [200, 400, 800],
            "min_child_samples": [10, 20, 50, 100],
        }
        return list(ParameterSampler(space, n_iter=12, random_state=RANDOM_SEED))
    if model == "random_forest":
        space = {
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 3, 10],
            "max_features": ["sqrt", 0.3],
        }
        return list(ParameterSampler(space, n_iter=8, random_state=RANDOM_SEED))
    raise ValueError(f"No tuning space for model '{model}'.")


def tuning_configs(
    results: pd.DataFrame, n_families: int = TUNING_FAMILIES
) -> list[ExperimentConfig]:
    """Settings to try for the best model families, each with its best strategy from the grid."""
    grid = results[results["stage"] == "grid"]
    best_per_model = [select_best(grid[grid["model"] == model]) for model in grid["model"].unique()]
    best_per_model.sort(key=lambda row: row["cv_pr_auc_mean"], reverse=True)
    known_ids = set(grid["config_id"])

    configs = []
    for best in best_per_model[:n_families]:
        ratio = None if pd.isna(best["sampling_ratio"]) else float(best["sampling_ratio"])
        for params in tuning_space(best["model"]):
            config = ExperimentConfig(
                best["model"], best["strategy"], ratio, params, stage="tuning"
            )
            if config.id not in known_ids:  # a draw equal to the defaults is already in the grid
                configs.append(config)
    return configs


def grid_is_complete(results: pd.DataFrame) -> bool:
    return {config.id for config in grid_configs()} <= set(results["config_id"])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=None)
    parser.add_argument("--strategy", nargs="+", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--sampling-ratio", nargs="+", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="re-run finished configurations")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, paths: ExperimentPaths | None = None) -> None:
    args = parse_args(argv)
    paths = paths or ExperimentPaths()
    train, val = load_train_val()
    failed: list[str] = []

    def run(configs: Sequence[ExperimentConfig]) -> None:
        outcome = run_configs(configs, train, val, paths, force=args.force, n_boot=args.n_boot)
        failed.extend(outcome["failed"])
        print(
            f"Ran {len(outcome['ran'])}, skipped {len(outcome['skipped'])}, "
            f"failed {len(outcome['failed'])}."
        )

    if args.stage in ("grid", "all"):
        print("== Strategy grid ==")
        run(select_configs(grid_configs(), args.model, args.strategy, args.sampling_ratio))

    if args.stage != "grid":
        results = load_results(paths.results)
        if not grid_is_complete(results) or failed:
            done = len({config.id for config in grid_configs()} & set(results["config_id"]))
            total = len(grid_configs())
            message = f"Grid has {done}/{total} configurations; tuning and report need all."
            if args.stage == "all" and not failed:
                print(message)
                return
            raise SystemExit(message)

    if args.stage in ("tune", "all"):
        print("== Tuning ==")
        run(select_configs(tuning_configs(load_results(paths.results)), args.model))

    if failed:
        raise SystemExit(f"{len(failed)} configuration(s) failed: {failed}")

    if args.stage in ("report", "all"):
        print("== Report ==")
        for line in build_report(load_results(paths.results), train, val, paths, args.n_boot):
            print(line)


if __name__ == "__main__":
    main()
