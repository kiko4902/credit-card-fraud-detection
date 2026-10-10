"""Phase 3 driver: the staged experiments and the report (PLAN.md section 6).

    python -m fraud.experiments                       # stages 1, 2, 3 and the report
    python -m fraud.experiments --stage 1 --estimate  # list stage 1 and estimate its runtime
    python -m fraud.experiments --stage 2 --model lightgbm --strategy smote_enn
    python -m fraud.experiments --stage report

Stage 1 screens 9 classifiers, stage 2 runs the SMOTE family for the best 4, stage 3 tunes
the best 2. Every configuration writes or replaces its own row in ``reports/results.csv``,
so an interrupted run continues where it stopped; ``--force`` re-runs finished ones.
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

from fraud.cache import ResampleCache
from fraud.config import (
    N_BOOTSTRAP,
    RANDOM_SEED,
    STAGE2_CLASSIFIERS,
    TUNING_FAMILIES,
    ExperimentPaths,
)
from fraud.data import load_train_val, split_xy, time_ordered_folds
from fraud.models import MODELS, STRATEGIES, build_pipeline, is_supported
from fraud.report import build_report
from fraud.selection import (
    MODEL_ORDER,
    STAGE1_STRATEGIES,
    STAGE2_STRATEGIES,
    rank_by_rule,
    rank_classifiers,
    select_best,
)
from fraud.train import (
    config_id,
    fit_pipeline,
    fraud_scores,
    load_results,
    params_json,
    run_experiment,
    save_result,
)

STAGES: list[str] = ["all", "1", "2", "3", "report"]
# Classifiers screened in stage 1 (a module attribute so tests can use a faster subset).
STAGE1_MODELS: list[str] = MODEL_ORDER
# A full run fits on 25 %, 50 %, 75 % and 100 % of the training split; the timing run uses
# only the first fold (25 %), so a full configuration costs about ten times as much.
TIMING_SCALE: float = 10.0


@dataclass(frozen=True)
class ExperimentConfig:
    """One classifier / strategy / hyperparameter setting."""

    model: str
    strategy: str
    sampling_ratio: float | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    stage: str = "stage1"

    @property
    def id(self) -> str:
        pipeline = build_pipeline(self.model, self.strategy, self.sampling_ratio, self.model_params)
        return config_id(self.model, self.strategy, self.sampling_ratio, params_json(pipeline))


# --- Stage definitions -----------------------------------------------------------------


def stage1_configs() -> list[ExperimentConfig]:
    """Screening: every classifier with no resampling, class weights and SMOTE 0.1."""
    return [
        ExperimentConfig(model, strategy, ratio, stage="stage1")
        for model in STAGE1_MODELS
        for strategy, ratio in STAGE1_STRATEGIES
        if is_supported(model, strategy)
    ]


def stage2_models(results: pd.DataFrame) -> list[str]:
    """The classifiers with the best mean CV PR-AUC over their stage 1 strategies."""
    stage1 = results[results["stage"] == "stage1"]
    return list(rank_classifiers(stage1)["model"].head(STAGE2_CLASSIFIERS))


def stage2_configs(results: pd.DataFrame) -> list[ExperimentConfig]:
    """SMOTE family: the best stage 1 classifiers with every remaining sampler and ratio."""
    return [
        ExperimentConfig(model, strategy, ratio, stage="stage2")
        for model in stage2_models(results)
        for strategy, ratio in STAGE2_STRATEGIES
    ]


def tuning_space(model: str) -> list[dict[str, Any]]:
    """The fixed, small set of hyperparameter settings tried for ``model`` in stage 3."""

    def draws(space: dict[str, list[Any]], n_iter: int) -> list[dict[str, Any]]:
        return list(ParameterSampler(space, n_iter=n_iter, random_state=RANDOM_SEED))

    forest = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 3, 10],
        "max_features": ["sqrt", 0.3],
    }
    spaces: dict[str, list[dict[str, Any]]] = {
        "logreg": [{"C": value} for value in (0.001, 0.01, 0.1, 10.0, 100.0)],
        "gaussian_nb": [{"var_smoothing": value} for value in (1e-11, 1e-7, 1e-5, 1e-3, 1e-1)],
        "decision_tree": draws(
            {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [1, 10, 50, 200]}, 8
        ),
        "knn": draws({"n_neighbors": [3, 11, 21, 51], "weights": ["uniform", "distance"]}, 6),
        "random_forest": draws(forest, 8),
        "extra_trees": draws(forest, 8),
        "xgboost": draws(
            {
                "max_depth": [3, 4, 6, 8],
                "learning_rate": [0.03, 0.1, 0.3],
                "n_estimators": [200, 400, 800],
                "min_child_weight": [1, 5, 10],
                "subsample": [0.7, 1.0],
            },
            12,
        ),
        "lightgbm": draws(
            {
                "num_leaves": [7, 15, 31, 63],
                "learning_rate": [0.02, 0.05, 0.1],
                "n_estimators": [200, 400, 800],
                "min_child_samples": [10, 20, 50, 100],
            },
            12,
        ),
        "mlp": draws(
            {
                "hidden_layer_sizes": [[50], [100], [100, 50]],
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "learning_rate_init": [1e-3, 3e-3],
            },
            8,
        ),
    }
    if model not in spaces:
        raise ValueError(f"No tuning space for model '{model}'.")
    return spaces[model]


def stage3_configs(
    results: pd.DataFrame, n_families: int = TUNING_FAMILIES
) -> list[ExperimentConfig]:
    """Tuning: the best classifiers by the selection rule, each with its best strategy."""
    screened = results[results["stage"].isin(["stage1", "stage2"])]
    best_per_model = pd.DataFrame(
        [select_best(screened[screened["model"] == model]) for model in screened["model"].unique()]
    )
    known_ids = set(screened["config_id"])
    configs = []
    for _, best in rank_by_rule(best_per_model).head(n_families).iterrows():
        ratio = None if pd.isna(best["sampling_ratio"]) else float(best["sampling_ratio"])
        for params in tuning_space(best["model"]):
            config = ExperimentConfig(
                best["model"], best["strategy"], ratio, params, stage="stage3"
            )
            if config.id not in known_ids:  # a draw equal to the defaults is already there
                configs.append(config)
    return configs


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


def missing(configs: Sequence[ExperimentConfig], results: pd.DataFrame) -> list[str]:
    """Ids of ``configs`` that have no results row yet."""
    done = set(results["config_id"])
    return [config.id for config in configs if config.id not in done]


# --- Running ---------------------------------------------------------------------------


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
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """Run each configuration unless it is already finished; return ids by outcome."""
    outcome: dict[str, list[str]] = {"ran": [], "skipped": [], "failed": []}
    paths.scores_dir.mkdir(parents=True, exist_ok=True)
    cache = ResampleCache(paths.cache_dir) if use_cache else None
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
                cache=cache,
            )
        except Exception as error:  # keep going: one failing sampler must not stop the stage
            outcome["failed"].append(identifier)
            log(f"{prefix}: FAILED with {type(error).__name__}: {error}")
            continue
        np.save(scores_path(identifier, paths.scores_dir), result.val_scores)
        save_result(result.row, paths.results)
        outcome["ran"].append(identifier)
        row = result.row
        log(
            f"{prefix}: CV PR-AUC {row['cv_pr_auc_mean']:.4f} | val PR-AUC {row['pr_auc']:.4f} "
            f"[{row['pr_auc_ci_low']:.4f}, {row['pr_auc_ci_high']:.4f}] | fit "
            f"{row['fit_time_s']:.1f} s, predict {row['predict_time_s']:.1f} s, "
            f"total {time.perf_counter() - start:.0f} s"
        )
    return outcome


def estimate_runtime(
    configs: Sequence[ExperimentConfig], train: pd.DataFrame, paths: ExperimentPaths
) -> pd.DataFrame:
    """Time every configuration on the first CV fold and extrapolate to a full run.

    The resampling time is counted once per sampler and ratio, because the full run caches
    it and shares it between classifiers.
    """
    train_idx, val_idx = time_ordered_folds(train)[0]
    features, target = split_xy(train)
    cache = ResampleCache(paths.cache_dir)
    rows = []
    seen_samplers: set[tuple[str, float | None]] = set()
    for config in configs:
        pipeline = build_pipeline(
            config.model, config.strategy, config.sampling_ratio, config.model_params
        )
        fitted = fit_pipeline(pipeline, features.iloc[train_idx], target.iloc[train_idx], cache)
        start = time.perf_counter()
        fraud_scores(fitted.pipeline, features.iloc[val_idx])
        predict_time = time.perf_counter() - start
        sampler = (config.strategy, config.sampling_ratio)
        resample_time = 0.0 if sampler in seen_samplers else fitted.resample_time_s
        seen_samplers.add(sampler)
        rows.append(
            {
                "config": config.id,
                "fold1_resample_s": round(resample_time, 1),
                "fold1_fit_s": round(fitted.fit_time_s, 1),
                "fold1_predict_s": round(predict_time, 1),
                "estimate_min": round(
                    TIMING_SCALE * (resample_time + fitted.fit_time_s + predict_time) / 60 + 0.05, 1
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--model", nargs="+", choices=sorted(MODELS), default=None)
    parser.add_argument("--strategy", nargs="+", choices=sorted(STRATEGIES), default=None)
    parser.add_argument("--sampling-ratio", nargs="+", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="re-run finished configurations")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="only list the stage's configurations and estimate their runtime",
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, paths: ExperimentPaths | None = None) -> None:
    args = parse_args(argv)
    paths = paths or ExperimentPaths()
    if args.estimate and args.stage not in ("1", "2", "3"):
        raise SystemExit("--estimate needs --stage 1, 2 or 3.")
    train, val = load_train_val()
    failed: list[str] = []

    def require(configs: Sequence[ExperimentConfig], what: str) -> None:
        todo = missing(configs, load_results(paths.results))
        if todo:
            message = f"{what} is incomplete ({len(configs) - len(todo)}/{len(configs)} done)."
            raise SystemExit(f"{message} Run it first.")

    def run(name: str, configs: list[ExperimentConfig]) -> None:
        selected = select_configs(configs, args.model, args.strategy, args.sampling_ratio)
        print(f"== {name}: {len(selected)} of {len(configs)} configurations ==")
        if args.estimate:
            estimate = estimate_runtime(selected, train, paths)
            print(estimate.to_string(index=False))
            print(f"Estimated total: {estimate['estimate_min'].sum():.0f} minutes")
            return
        outcome = run_configs(selected, train, val, paths, force=args.force, n_boot=args.n_boot)
        failed.extend(outcome["failed"])
        print(
            f"Ran {len(outcome['ran'])}, skipped {len(outcome['skipped'])}, "
            f"failed {len(outcome['failed'])}."
        )
        if failed:
            raise SystemExit(f"{len(failed)} configuration(s) failed: {failed}")

    if args.stage in ("1", "all"):
        run("Stage 1 - screening", stage1_configs())
    if args.stage == "all" and missing(stage1_configs(), load_results(paths.results)):
        print("Stage 1 is incomplete; stopping before stage 2.")
        return
    if args.stage in ("2", "all"):
        require(stage1_configs(), "Stage 1")
        run("Stage 2 - SMOTE family", stage2_configs(load_results(paths.results)))
    if args.stage == "all" and missing(
        stage2_configs(load_results(paths.results)), load_results(paths.results)
    ):
        print("Stage 2 is incomplete; stopping before stage 3.")
        return
    if args.stage in ("3", "report", "all"):
        require(stage1_configs(), "Stage 1")
        require(stage2_configs(load_results(paths.results)), "Stage 2")
    if args.stage in ("3", "all"):
        run("Stage 3 - tuning", stage3_configs(load_results(paths.results)))
    if args.stage in ("report", "all") and not args.estimate:
        print("== Report ==")
        for line in build_report(load_results(paths.results), train, val, paths, args.n_boot):
            print(line)


if __name__ == "__main__":
    main()
