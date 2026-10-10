"""Train one model / imbalance strategy on the training split and evaluate it on validation.

    python -m fraud.train --model logreg --strategy none

One row per configuration is written to ``reports/results.csv``. The test split is never
loaded here. The full grid lives in ``fraud.experiments``.
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from fraud.config import (
    CV_N_SPLITS,
    DEFAULT_THRESHOLD,
    N_BOOTSTRAP,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_CSV,
)
from fraud.data import fold_summary, load_train_val, split_xy, time_ordered_folds
from fraud.evaluate import evaluate_scores
from fraud.models import MODELS, STRATEGIES, build_pipeline

RESULT_COLUMNS: list[str] = [
    "timestamp",
    "git_commit",
    "stage",
    "config_id",
    "model",
    "strategy",
    "sampling_ratio",
    "params",
    "eval_split",
    "deduplicated",
    "n_train",
    "n_train_fraud",
    "n_eval",
    "n_eval_fraud",
    "pr_auc",
    "pr_auc_ci_low",
    "pr_auc_ci_high",
    "roc_auc",
    "threshold",
    "precision",
    "recall",
    "recall_ci_low",
    "recall_ci_high",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "cv_folds",
    "cv_pr_auc_mean",
    "cv_pr_auc_std",
    "cv_pr_auc_folds",
    "cv_time_s",
    "train_time_s",
    "n_boot",
    "random_seed",
]
# A configuration is identified by these columns; re-running it replaces its row.
KEY_COLUMNS: list[str] = ["config_id", "eval_split", "deduplicated"]


@dataclass(frozen=True)
class ExperimentResult:
    """Everything one experiment produces."""

    row: dict[str, Any]
    cv_table: pd.DataFrame | None
    val_scores: np.ndarray
    pipeline: Pipeline


def git_commit() -> str:
    """Short hash of HEAD, with ``-dirty`` if code or dependencies have uncommitted changes."""

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def params_json(pipeline: Pipeline) -> str:
    """All parameters of the pipeline's model as canonical JSON."""
    params = pipeline.named_steps["model"].get_params()
    return json.dumps(params, sort_keys=True, default=str)


def config_id(model: str, strategy: str, sampling_ratio: float | None, params: str) -> str:
    """Stable identifier of a configuration, e.g. ``lightgbm__smote__0.1__3f2a9c1d``.

    The suffix is a hash of the model's full parameter set, so two settings of the same
    model and strategy get different ids.
    """
    ratio = "na" if sampling_ratio is None else f"{sampling_ratio:g}"
    digest = hashlib.sha1(params.encode("utf-8")).hexdigest()[:8]
    return f"{model}__{strategy}__{ratio}__{digest}"


def fraud_scores(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Predicted probability of fraud for each row."""
    return pipeline.predict_proba(features)[:, 1]


def cross_validate(
    pipeline: Pipeline, train: pd.DataFrame, n_splits: int = CV_N_SPLITS
) -> pd.DataFrame:
    """PR-AUC of ``pipeline`` on time-ordered folds of the training split.

    Each fold fits a fresh clone on its own training part, so scaling and resampling never
    see the fold's validation part.
    """
    folds = time_ordered_folds(train, n_splits=n_splits)
    table = fold_summary(train, folds)
    features, target = split_xy(train)
    scores = []
    for train_idx, val_idx in folds:
        fitted = clone(pipeline).fit(features.iloc[train_idx], target.iloc[train_idx])
        fold_scores = fraud_scores(fitted, features.iloc[val_idx])
        scores.append(average_precision_score(target.iloc[val_idx], fold_scores))
    return table.assign(pr_auc=scores)


def run_experiment(
    model: str,
    strategy: str,
    sampling_ratio: float | None,
    train: pd.DataFrame,
    val: pd.DataFrame,
    run_cv: bool = True,
    n_boot: int = N_BOOTSTRAP,
    threshold: float = DEFAULT_THRESHOLD,
    model_params: dict[str, Any] | None = None,
    stage: str = "grid",
) -> ExperimentResult:
    """Fit on ``train`` and evaluate on ``val``."""
    pipeline = build_pipeline(model, strategy, sampling_ratio, model_params=model_params)
    params = params_json(pipeline)

    start = time.perf_counter()
    cv_table = cross_validate(pipeline, train) if run_cv else None
    cv_time = time.perf_counter() - start

    train_x, train_y = split_xy(train)
    val_x, val_y = split_xy(val)
    start = time.perf_counter()
    pipeline.fit(train_x, train_y)
    train_time = time.perf_counter() - start
    val_scores = fraud_scores(pipeline, val_x)
    metrics = evaluate_scores(val_y, val_scores, threshold=threshold, n_boot=n_boot)

    fold_scores = [] if cv_table is None else [round(score, 6) for score in cv_table["pr_auc"]]
    row: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "stage": stage,
        "config_id": config_id(model, strategy, sampling_ratio, params),
        "model": model,
        "strategy": strategy,
        "sampling_ratio": sampling_ratio,
        "params": params,
        "eval_split": "val",
        "deduplicated": True,
        "n_train": len(train),
        "n_train_fraud": int(train_y.sum()),
        "n_eval": len(val),
        "n_eval_fraud": int(val_y.sum()),
        **metrics,
        "cv_folds": len(fold_scores),
        "cv_pr_auc_mean": cv_table["pr_auc"].mean() if cv_table is not None else None,
        "cv_pr_auc_std": cv_table["pr_auc"].std() if cv_table is not None else None,
        "cv_pr_auc_folds": json.dumps(fold_scores),
        "cv_time_s": round(cv_time, 2),
        "train_time_s": round(train_time, 2),
        "n_boot": n_boot,
        "random_seed": RANDOM_SEED,
    }
    ordered = {column: row[column] for column in RESULT_COLUMNS}
    return ExperimentResult(ordered, cv_table, val_scores, pipeline)


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Read the results CSV; an empty frame with the right columns if it does not exist."""
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    results = pd.read_csv(path)
    if list(results.columns) != RESULT_COLUMNS:
        raise ValueError(f"{path} has a different schema; move it away before writing.")
    return results


def save_result(row: dict[str, Any], path: Path = RESULTS_CSV) -> None:
    """Write ``row`` to the results CSV, replacing an earlier row of the same configuration."""
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    existing = load_results(path)
    if not existing.empty:
        new = pd.concat([existing.astype(object), new.astype(object)], ignore_index=True)
        new = new.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, lineterminator="\n")


def format_report(row: dict[str, Any], cv_table: pd.DataFrame | None) -> str:
    """Human-readable summary of one experiment."""
    ratio = "" if row["sampling_ratio"] is None else f" (ratio {row['sampling_ratio']})"
    lines = [
        f"Model: {row['model']} | strategy: {row['strategy']}{ratio} | commit {row['git_commit']}",
        f"Train: {row['n_train']:,} rows, {row['n_train_fraud']} fraud "
        f"| fitted in {row['train_time_s']:.1f} s",
    ]
    if cv_table is not None:
        lines += [
            "",
            "Time-ordered cross-validation on the training split:",
            cv_table.round({"pr_auc": 4}).to_string(),
            f"  CV PR-AUC: {row['cv_pr_auc_mean']:.4f} +/- {row['cv_pr_auc_std']:.4f} (mean, std)",
        ]
    pr_auc_ci = f"[{row['pr_auc_ci_low']:.4f}, {row['pr_auc_ci_high']:.4f}]"
    recall_ci = f"[{row['recall_ci_low']:.4f}, {row['recall_ci_high']:.4f}]"
    counts = f"TN {row['tn']:,}  FP {row['fp']:,}  FN {row['fn']:,}  TP {row['tp']:,}"
    lines += [
        "",
        f"Validation: {row['n_eval']:,} rows, {row['n_eval_fraud']} fraud "
        f"(95 % CI from {row['n_boot']} stratified bootstrap resamples)",
        f"  PR-AUC    {row['pr_auc']:.4f}  {pr_auc_ci}",
        f"  ROC-AUC   {row['roc_auc']:.4f}",
        f"  At threshold {row['threshold']}:",
        f"    recall    {row['recall']:.4f}  {recall_ci}",
        f"    precision {row['precision']:.4f}",
        f"    F1        {row['f1']:.4f}",
        f"    confusion matrix: {counts}",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="logreg")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="none")
    parser.add_argument(
        "--sampling-ratio",
        type=float,
        default=None,
        help="minority:majority ratio after resampling (default: full balancing)",
    )
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation on train")
    parser.add_argument("--results", type=Path, default=RESULTS_CSV)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train, val = load_train_val()
    result = run_experiment(
        args.model, args.strategy, args.sampling_ratio, train, val, run_cv=not args.no_cv
    )
    save_result(result.row, args.results)
    print(format_report(result.row, result.cv_table))
    print(f"\nSaved to {args.results}")


if __name__ == "__main__":
    main()
