"""Phase 4: threshold on validation, one final evaluation on the test split, SHAP.

    python -m fraud.final_eval                 # the real run: loads the test split, once
    python -m fraud.final_eval --dry-run DIR   # validation rows stand in for the test split
    python -m fraud.final_eval --render-only   # rewrite the report from the saved metrics

The model is the pipeline saved by Phase 3, fitted on the training split only. Every
threshold is chosen on validation. The test split is loaded in exactly one place
(:func:`load_evaluation_rows`), and every load is appended to ``reports/test_set_access.log``.
Use ``--dry-run`` to check code changes: it never touches the test split and writes all its
output to the given directory.
"""

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, precision_recall_curve

from fraud import explain
from fraud.config import (
    AMOUNT_COLUMN,
    DEFAULT_THRESHOLD,
    FINAL_CONFIG_NAME,
    FINAL_MODEL_NAME,
    MODEL_VERSION,
    N_BOOTSTRAP,
    RANDOM_SEED,
    RAW_CSV,
    REVIEW_COST_EUR,
    TARGET_COLUMN,
    TIME_COLUMN,
    EvaluationPaths,
)
from fraud.data import (
    dropped_duplicates,
    load_test_with_duplicates,
    load_train_val,
    split_xy,
    stratified_random_split,
)
from fraud.evaluate import paired_bootstrap_difference
from fraud.plotting import NEUTRAL_COLOR, RANK_COLORS, save_figure, set_style
from fraud.report import (
    GENERATED_END,
    GENERATED_START,
    find_row,
    load_scores,
    markdown_table,
    rebuild_pipeline,
)
from fraud.selection import SELECTION_STAGES, config_label, select_best
from fraud.threshold import (
    CostCurve,
    Decisions,
    bootstrap_optimal_thresholds,
    evaluate_decisions,
    f1_optimal_threshold,
    paired_cost_difference,
)
from fraud.train import fraud_scores, git_commit, load_results

SELECTED = "selected"
# Reference models of the thesis hypothesis test: (model, strategy, ratio), untuned rows.
REFERENCES: dict[str, tuple[str, str, float | None]] = {
    "xgb_class_weight": ("xgboost", "class_weight", None),
    "xgb_none": ("xgboost", "none", None),
    "mlp_none": ("mlp", "none", None),
}
MODEL_COLORS: dict[str, str] = dict(zip([SELECTED, *REFERENCES], RANK_COLORS, strict=True))
CALIBRATION_EDGES: list[float] = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
REPORT_TEMPLATE = f"""# Final evaluation (Phase 4)

## Conclusions

_To be written after reviewing the results below._

## Thesis hypothesis: does oversampling mainly shift the decision boundary?

_To be written after reviewing the results below._

## Limitations

_To be written after reviewing the results below._

{GENERATED_START}
{GENERATED_END}
"""


@dataclass(frozen=True)
class EvaluatedModel:
    """A fitted pipeline with its name in the report."""

    key: str
    label: str
    config_id: str
    pipeline: Pipeline


# --- Models ------------------------------------------------------------------------------


def selected_row(results: pd.DataFrame) -> pd.Series:
    """The configuration chosen by the Phase 3 selection rule."""
    candidates = results[
        (results["eval_split"] == "val")
        & results["cv_pr_auc_mean"].notna()
        & results["stage"].isin(SELECTION_STAGES)
    ]
    return select_best(candidates)


def load_models(
    results: pd.DataFrame, train: pd.DataFrame, models_dir: Path
) -> dict[str, EvaluatedModel]:
    """The saved final pipeline and the reference models, all fitted on the training split."""
    final = selected_row(results)
    config = json.loads((models_dir / FINAL_CONFIG_NAME).read_text(encoding="utf-8"))
    if config["config_id"] != final["config_id"]:
        raise ValueError(
            f"Saved model is {config['config_id']} but the selection rule gives "
            f"{final['config_id']}. Rebuild the Phase 3 report first."
        )
    models = {
        SELECTED: EvaluatedModel(
            SELECTED,
            f"{config_label(final)} (selected)",
            final["config_id"],
            joblib.load(models_dir / FINAL_MODEL_NAME),
        )
    }
    features, target = split_xy(train)
    for key, (model, strategy, ratio) in REFERENCES.items():
        row = find_row(results[results["stage"] == "stage1"], model, strategy, ratio)
        if row is None:
            raise ValueError(f"No stage 1 result for reference {model} / {strategy}.")
        pipeline = rebuild_pipeline(row).fit(features, target)
        models[key] = EvaluatedModel(key, config_label(row), row["config_id"], pipeline)
    return models


def score_all(models: dict[str, EvaluatedModel], rows: pd.DataFrame) -> dict[str, np.ndarray]:
    features, _ = split_xy(rows)
    return {key: fraud_scores(model.pipeline, features) for key, model in models.items()}


# --- Validation: thresholds, hypothesis test, calibration ---------------------------------


def point_metrics(
    y_true: np.ndarray, scores: np.ndarray, amount: np.ndarray, threshold: float
) -> dict[str, float]:
    return Decisions(y_true, scores, amount, threshold, REVIEW_COST_EUR).metrics()


def validation_analysis(
    models: dict[str, EvaluatedModel],
    val: pd.DataFrame,
    scores: dict[str, np.ndarray],
    scores_dir: Path,
    n_boot: int,
) -> dict[str, Any]:
    """Per model: cost-optimal and F1-optimal threshold on validation and what they cost."""
    y_true = val[TARGET_COLUMN].to_numpy()
    amount = val[AMOUNT_COLUMN].to_numpy(dtype=float)
    result = {}
    for key, model in models.items():
        curve = CostCurve(y_true, scores[key], amount)
        threshold, _ = curve.best()
        f1_threshold, f1 = f1_optimal_threshold(y_true, scores[key])
        saved = load_scores(model.config_id, scores_dir)
        result[key] = {
            "label": model.label,
            "config_id": model.config_id,
            "scores_match_phase3": bool(np.allclose(saved, scores[key], atol=1e-6)),
            "threshold": threshold,
            "f1_threshold": f1_threshold,
            "f1_max": f1,
            "at_cost_threshold": evaluate_decisions(
                y_true, scores[key], amount, threshold, n_boot=n_boot
            ),
            "at_default_threshold": point_metrics(y_true, scores[key], amount, DEFAULT_THRESHOLD),
            "at_f1_threshold": point_metrics(y_true, scores[key], amount, f1_threshold),
        }
    return result


def threshold_stability(val: pd.DataFrame, scores: np.ndarray, n_boot: int) -> dict[str, float]:
    """Spread of the cost-optimal threshold over bootstrap resamples of validation."""
    thresholds = bootstrap_optimal_thresholds(
        val[TARGET_COLUMN], scores, val[AMOUNT_COLUMN], n_boot=n_boot
    )
    low, median, high = np.quantile(thresholds, [0.025, 0.5, 0.975])
    return {"q025": float(low), "median": float(median), "q975": float(high)}


def hypothesis_tests(
    val: pd.DataFrame, scores: dict[str, np.ndarray], n_boot: int
) -> list[dict[str, Any]]:
    """Selected model against each reference: ranking, cost at 0.5, cost at own optimum."""
    y_true = val[TARGET_COLUMN].to_numpy()
    amount = val[AMOUNT_COLUMN].to_numpy(dtype=float)
    rows = []
    for key in REFERENCES:
        a, b = scores[SELECTED], scores[key]
        rows.append(
            {
                "reference": key,
                "pr_auc": paired_bootstrap_difference(y_true, a, b, n_boot=n_boot),
                "cost_default": paired_cost_difference(
                    y_true, amount, a, b, DEFAULT_THRESHOLD, DEFAULT_THRESHOLD, n_boot=n_boot
                ),
                "cost_optimal": paired_cost_difference(y_true, amount, a, b, n_boot=n_boot),
            }
        )
    return rows


def calibration(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """Mean score against observed fraud rate in fixed score bins."""
    bins = np.clip(np.digitize(scores, CALIBRATION_EDGES[1:-1]), 0, len(CALIBRATION_EDGES) - 2)
    rows = []
    for index in range(len(CALIBRATION_EDGES) - 1):
        inside = bins == index
        if inside.any():
            rows.append(
                {
                    "low": CALIBRATION_EDGES[index],
                    "high": CALIBRATION_EDGES[index + 1],
                    "n": int(inside.sum()),
                    "fraud": int(y_true[inside].sum()),
                    "mean_score": float(scores[inside].mean()),
                    "fraud_rate": float(y_true[inside].mean()),
                }
            )
    return {
        "bins": rows,
        "brier": float(brier_score_loss(y_true, scores)),
        "brier_of_base_rate": float(np.mean((y_true - y_true.mean()) ** 2)),
        "expected_fraud": float(scores.sum()),
        "actual_fraud": int(y_true.sum()),
    }


# --- Test --------------------------------------------------------------------------------


def log_test_access(path: Path, test: pd.DataFrame, duplicates: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{datetime.now(UTC).isoformat(timespec='seconds')}\tcommit {git_commit()}\t"
        f"fraud.final_eval\ttest rows {len(test)}\tfraud {int(test[TARGET_COLUMN].sum())}\t"
        f"duplicates in the test period {len(duplicates)}\n"
    )
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(line)


def load_evaluation_rows(
    dry_run: bool, val: pd.DataFrame, access_log: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of the final evaluation and the exact duplicates removed from their period.

    THE ONLY PLACE WHERE THE TEST SPLIT IS LOADED. In a dry run the validation split and
    its duplicates are returned instead and nothing is logged.
    """
    if dry_run:
        duplicates = dropped_duplicates()
        times = duplicates[TIME_COLUMN]
        inside = (times >= val[TIME_COLUMN].min()) & (times <= val[TIME_COLUMN].max())
        return val, duplicates[inside]
    test, duplicates = load_test_with_duplicates(allow_test=True)
    log_test_access(access_log, test, duplicates)
    return test, duplicates


def evaluation_analysis(
    models: dict[str, EvaluatedModel],
    thresholds: dict[str, float],
    test: pd.DataFrame,
    duplicates: pd.DataFrame,
    scores: dict[str, np.ndarray],
    n_boot: int,
) -> dict[str, Any]:
    """Every model at its validation threshold; the selected one also with duplicates."""
    y_true = test[TARGET_COLUMN].to_numpy()
    amount = test[AMOUNT_COLUMN].to_numpy(dtype=float)
    per_model = {
        key: {
            "label": model.label,
            "threshold": thresholds[key],
            "metrics": evaluate_decisions(
                y_true, scores[key], amount, thresholds[key], n_boot=n_boot
            ),
            "at_default_threshold": point_metrics(y_true, scores[key], amount, DEFAULT_THRESHOLD),
        }
        for key, model in models.items()
    }
    paired = [
        {
            "reference": key,
            "pr_auc": paired_bootstrap_difference(
                y_true, scores[SELECTED], scores[key], n_boot=n_boot
            ),
            "cost": paired_cost_difference(
                y_true,
                amount,
                scores[SELECTED],
                scores[key],
                thresholds[SELECTED],
                thresholds[key],
                n_boot=n_boot,
            ),
        }
        for key in REFERENCES
    ]
    with_duplicates = pd.concat([test, duplicates])
    duplicate_scores = fraud_scores(models[SELECTED].pipeline, split_xy(with_duplicates)[0])
    # Diagnostic only: the best threshold in hindsight shows what the validation threshold
    # leaves on the table. It is not used for anything.
    hindsight_threshold, hindsight_cost = CostCurve(y_true, scores[SELECTED], amount).best()
    return {
        "models": per_model,
        "paired": paired,
        "with_duplicates": {
            "rows": len(with_duplicates),
            "fraud": int(with_duplicates[TARGET_COLUMN].sum()),
            "duplicate_rows": len(duplicates),
            "duplicate_fraud": int(duplicates[TARGET_COLUMN].sum()),
            "metrics": evaluate_decisions(
                with_duplicates[TARGET_COLUMN],
                duplicate_scores,
                with_duplicates[AMOUNT_COLUMN],
                thresholds[SELECTED],
                n_boot=n_boot,
            ),
        },
        "hindsight": {"threshold": hindsight_threshold, "cost": hindsight_cost},
    }


def random_split_analysis(final: pd.Series, data: pd.DataFrame, n_boot: int) -> dict[str, Any]:
    """Secondary analysis: the same configuration on a stratified random split.

    Refit on the random training part, threshold from the random validation part, evaluated
    on the random test part. Not an estimate of performance on future data.
    """
    splits = stratified_random_split(data)
    pipeline = rebuild_pipeline(final).fit(*split_xy(splits["train"]))
    val, test = splits["val"], splits["test"]
    val_scores = fraud_scores(pipeline, split_xy(val)[0])
    threshold, _ = CostCurve(val[TARGET_COLUMN], val_scores, val[AMOUNT_COLUMN]).best()
    test_scores = fraud_scores(pipeline, split_xy(test)[0])
    return {
        "rows": {name: len(part) for name, part in splits.items()},
        "fraud": {name: int(part[TARGET_COLUMN].sum()) for name, part in splits.items()},
        "threshold": threshold,
        "metrics": evaluate_decisions(
            test[TARGET_COLUMN], test_scores, test[AMOUNT_COLUMN], threshold, n_boot=n_boot
        ),
    }


# --- SHAP --------------------------------------------------------------------------------


def shap_analysis(
    pipeline: Pipeline,
    rows: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    paths: EvaluationPaths,
    split_name: str,
) -> dict[str, Any]:
    """Global and local SHAP figures and the explained sample for the app."""
    features, target = split_xy(rows)
    explanation = explain.explain(pipeline, features)
    figures = [
        explain.plot_global(explanation, paths.figures_dir / "eval_shap_summary.png", split_name)
    ]
    cases = {}
    for name, position in explain.pick_cases(
        target, scores, rows[AMOUNT_COLUMN], threshold
    ).items():
        amount = float(rows[AMOUNT_COLUMN].iloc[position])
        title = (
            f"{explain.CASE_LABELS[name]}: amount €{amount:,.2f}, fraud score "
            f"{scores[position]:.3f} (threshold {threshold:.3f})"
        )
        path = paths.figures_dir / f"eval_shap_{name}.png"
        figures.append(explain.plot_local(explanation[position], title, path))
        contributions = pd.Series(explanation.values[position], explanation.feature_names)
        strongest = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
        cases[name] = {
            "amount": amount,
            "score": float(scores[position]),
            "top_features": {name: float(value) for name, value in strongest.head(5).items()},
        }
    sample = explain.explained_sample(rows, scores, explanation, threshold)
    paths.sample.parent.mkdir(parents=True, exist_ok=True)
    sample.round(6).to_csv(paths.sample, index=False, lineterminator="\n")
    importance = explain.mean_absolute_shap(explanation)
    return {
        "figures": [figure.name for figure in figures],
        "top_features": {name: float(value) for name, value in importance.head(10).items()},
        "additivity_error": explain.additivity_error(pipeline, features, explanation),
        "base_value": float(np.ravel(explanation.base_values)[0]),
        "cases": cases,
        "sample_rows": len(sample),
        "sample_outcomes": {k: int(v) for k, v in sample["outcome"].value_counts().items()},
    }


# --- Figures -----------------------------------------------------------------------------


def plot_cost_vs_threshold(
    val: pd.DataFrame, scores: np.ndarray, summary: dict[str, Any], figures_dir: Path
) -> Path:
    """Total validation cost of the selected model over the whole threshold range."""
    set_style()
    table = CostCurve(val[TARGET_COLUMN], scores, val[AMOUNT_COLUMN]).table()
    no_model = summary["at_cost_threshold"]["cost_no_model"]["value"]
    floor = 1e-5
    thresholds = np.clip(table["threshold"].to_numpy(), floor, 1.0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.step(thresholds, table["cost"], where="post", color=MODEL_COLORS[SELECTED], linewidth=2)
    ax.axhline(no_model, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
    ax.text(floor * 1.3, no_model * 1.02, f"no model (flag nothing): €{no_model:,.0f}", fontsize=9)
    marks = [
        (summary["threshold"], summary["at_cost_threshold"]["cost"]["value"], "cost-optimal"),
        (summary["f1_threshold"], summary["at_f1_threshold"]["cost"], "F1-optimal"),
        (DEFAULT_THRESHOLD, summary["at_default_threshold"]["cost"], "default 0.5"),
    ]
    for (threshold, cost, name), marker in zip(marks, ["o", "s", "D"], strict=True):
        ax.scatter(
            [np.clip(threshold, floor, 1.0)],
            [cost],
            marker=marker,
            s=70,
            zorder=3,
            color=MODEL_COLORS[SELECTED],
            edgecolor="white",
            linewidth=1.5,
            label=f"{name}: threshold {threshold:.3g}, cost €{cost:,.0f}",
        )
    ax.set_xscale("log")
    ax.set_xlim(floor, 1.0)
    ax.set_ylim(0, no_model * 1.6)
    ax.set_xlabel("Decision threshold on the fraud score (log scale)")
    ax.set_ylabel("Total cost on validation (€)")
    ax.set_title(
        "Cost against threshold on validation: missed fraud amounts + €"
        f"{REVIEW_COST_EUR:g} per flagged transaction"
    )
    ax.legend(loc="upper center", fontsize=9)
    path = save_figure(fig, "eval_cost_vs_threshold", figures_dir)
    plt.close(fig)
    return path


def plot_cost_by_model(validation: dict[str, Any], figures_dir: Path) -> Path:
    """Validation cost of each model at threshold 0.5 and at its own cost-optimal threshold."""
    set_style()
    keys = list(validation)
    default = [validation[key]["at_default_threshold"]["cost"] for key in keys]
    optimal = [validation[key]["at_cost_threshold"]["cost"]["value"] for key in keys]
    no_model = validation[SELECTED]["at_cost_threshold"]["cost_no_model"]["value"]
    positions = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = [
        ax.bar(positions - 0.19, default, 0.36, color="#b5b3ab", label="at threshold 0.5"),
        ax.bar(
            positions + 0.19, optimal, 0.36, color="#6250d6", label="at own cost-optimal threshold"
        ),
    ]
    for group in bars:
        ax.bar_label(group, fmt="€{:,.0f}", fontsize=9, padding=2)
    ax.set_ylim(0, max(default + optimal) * 1.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [
            validation[key]["label"].replace(", ", ",\n").replace(" (selected)", "\n(selected)")
            for key in keys
        ],
        fontsize=9,
    )
    ax.set_ylabel("Total cost on validation (€)")
    ax.set_title(
        "Validation cost per model at 0.5 and at its own threshold "
        f"(flagging nothing would cost €{no_model:,.0f})"
    )
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=9)
    path = save_figure(fig, "eval_cost_by_model", figures_dir)
    plt.close(fig)
    return path


def plot_reliability(table: dict[str, Any], figures_dir: Path) -> Path:
    """Reliability of the selected model on validation, on log axes.

    Almost all rows score close to zero, so linear axes would hide them. Bins without any
    observed fraud cannot be drawn on a log axis; they are shown hollow at the bottom edge.
    """
    set_style()
    bins = pd.DataFrame(table["bins"])
    floor = 1e-5
    observed = bins["fraud_rate"].clip(lower=floor)
    predicted = bins["mean_score"].clip(lower=floor)
    empty = bins["fraud"] == 0
    color = MODEL_COLORS[SELECTED]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(
        [floor, 1],
        [floor, 1],
        color=NEUTRAL_COLOR,
        linestyle="--",
        linewidth=1,
        label="perfectly calibrated",
    )
    ax.scatter(
        predicted[~empty],
        observed[~empty],
        s=80,
        color=color,
        zorder=3,
        label="score bin with observed fraud",
    )
    ax.scatter(
        predicted[empty],
        observed[empty],
        s=80,
        facecolor="none",
        edgecolor=color,
        linewidth=1.8,
        zorder=3,
        label="score bin with no fraud observed (drawn at the bottom edge)",
    )
    # Labels alternate above and below, because neighbouring bins can lie close together.
    for position, (x, y, (_, row)) in enumerate(
        zip(predicted, observed, bins.iterrows(), strict=True)
    ):
        ax.annotate(
            f"{int(row['fraud'])} of {int(row['n']):,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9 if position % 2 == 0 else -17),
            ha="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(floor / 2, 2)
    ax.set_ylim(floor / 2, 4)
    ax.set_xlabel("Mean fraud score in the bin (log scale)")
    ax.set_ylabel("Observed share of fraud in the bin (log scale)")
    ax.set_title("Reliability on validation: fraud cases of rows per score bin")
    ax.legend(loc="upper left", fontsize=8.5)
    path = save_figure(fig, "eval_reliability", figures_dir)
    plt.close(fig)
    return path


def plot_test_pr_curves(
    test: pd.DataFrame,
    scores: dict[str, np.ndarray],
    analysis: dict[str, Any],
    figures_dir: Path,
    split_name: str,
) -> Path:
    """Precision-recall curves with the operating point of every model's validation threshold."""
    set_style()
    y_true = test[TARGET_COLUMN].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in reversed(list(scores)):
        entry = analysis["models"][key]
        metrics = entry["metrics"]
        precision, recall, _ = precision_recall_curve(y_true, scores[key])
        ax.step(
            recall,
            precision,
            where="post",
            color=MODEL_COLORS[key],
            linewidth=2.4 if key == SELECTED else 1.4,
            label=f"{entry['label']} (PR-AUC {metrics['pr_auc']['value']:.3f})",
        )
        ax.scatter(
            [metrics["recall"]["value"]],
            [metrics["precision"]["value"]],
            s=80,
            zorder=3,
            color=MODEL_COLORS[key],
            edgecolor="white",
            linewidth=1.5,
        )
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flagged that is fraud)")
    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"{split_name} ({int(y_true.sum())} fraud cases): PR curves, dots mark each model's validation threshold"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", fontsize=9)
    path = save_figure(fig, "eval_test_pr_curves", figures_dir)
    plt.close(fig)
    return path


# --- Saved config ------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    """Hash of the values of a frame, row order included."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    ).hexdigest()


def _values(metrics: dict[str, dict[str, float]], names: Sequence[str]) -> dict[str, Any]:
    return {name: {key: round(value, 6) for key, value in metrics[name].items()} for name in names}


def update_config(
    metrics: dict[str, Any], train: pd.DataFrame, paths: EvaluationPaths
) -> dict[str, Any]:
    """Add threshold, cost model, version, data hash and metrics to the saved model config."""
    config = json.loads((paths.models_dir / FINAL_CONFIG_NAME).read_text(encoding="utf-8"))
    selected = metrics["validation"][SELECTED]
    reported = ["pr_auc", "roc_auc", "precision", "recall", "f1", "cost", "savings_vs_no_model"]
    config.update(
        {
            "model_version": MODEL_VERSION,
            "threshold": selected["threshold"],
            "decision": {
                "rule": "flag the transaction when fraud score >= threshold",
                "threshold_chosen_on": "validation split, minimum total cost",
                "f1_optimal_threshold_for_comparison": selected["f1_threshold"],
                "refit_on_train_plus_val": False,
            },
            "cost_model": {
                "currency": "EUR",
                "review_cost_per_flagged_transaction": REVIEW_COST_EUR,
                "missed_fraud": "transaction amount",
                "passed_legitimate_transaction": 0.0,
            },
            "training_data": {
                "source": "Kaggle mlg-ulb/creditcardfraud, creditcard.csv",
                "raw_file_sha256": file_sha256(RAW_CSV) if RAW_CSV.exists() else None,
                "train_split_sha256": frame_sha256(train),
                "rows": len(train),
                "fraud": int(train[TARGET_COLUMN].sum()),
            },
            "metrics": {
                "validation_at_threshold": _values(selected["at_cost_threshold"], reported),
                metrics["evaluation_split"] + "_at_threshold": _values(
                    metrics["test"]["models"][SELECTED]["metrics"], reported
                ),
                "evaluated_at_commit": metrics["git_commit"],
            },
            "output": (
                "A fraud score between 0 and 1 that ranks transactions. It is not a calibrated "
                "probability: the model is trained on oversampled data."
            ),
        }
    )
    paths.config_out.parent.mkdir(parents=True, exist_ok=True)
    paths.config_out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    return config


# --- Report ------------------------------------------------------------------------------


def eur(value: float) -> str:
    return f"€{value:,.0f}"


def signed_eur(value: float) -> str:
    return f"{'+' if value > 0 else '-' if value < 0 else ''}€{abs(value):,.0f}"


def with_ci(metric: dict[str, float], fmt: str = "{:.3f}") -> str:
    return f"{fmt.format(metric['value'])} [{fmt.format(metric['ci_low'])}, {fmt.format(metric['ci_high'])}]"


def eur_ci(metric: dict[str, float]) -> str:
    return f"{eur(metric['value'])} [{eur(metric['ci_low'])}, {eur(metric['ci_high'])}]"


def difference(result: dict[str, float], money: bool) -> str:
    star = " *" if result["ci_low"] > 0 or result["ci_high"] < 0 else ""
    if money:
        return f"{signed_eur(result['difference'])} [{signed_eur(result['ci_low'])}, {signed_eur(result['ci_high'])}]{star}"
    return f"{result['difference']:+.3f} [{result['ci_low']:+.3f}, {result['ci_high']:+.3f}]{star}"


def threshold_table(validation: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in validation.values():
        best, default, f1 = (
            entry["at_cost_threshold"],
            entry["at_default_threshold"],
            entry["at_f1_threshold"],
        )
        fraud = int(best["tp"]["value"] + best["fn"]["value"])
        rows.append(
            {
                "Model": entry["label"],
                "Cost-optimal threshold": f"{entry['threshold']:.4g}",
                "Flagged": f"{int(best['tp']['value'] + best['fp']['value'])}",
                "Fraud caught": f"{int(best['tp']['value'])} of {fraud}",
                "Cost at optimum": eur(best["cost"]["value"]),
                "Cost at 0.5": eur(default["cost"]),
                "F1-optimal threshold": f"{entry['f1_threshold']:.4g}",
                "Best F1": f"{entry['f1_max']:.3f}",
                "Cost at F1 optimum": eur(f1["cost"]),
            }
        )
    return pd.DataFrame(rows)


def hypothesis_table(tests: list[dict[str, Any]], validation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Selected minus": validation[test["reference"]]["label"],
                "PR-AUC": difference(test["pr_auc"], money=False),
                "Cost, both at 0.5": difference(test["cost_default"], money=True),
                "Cost, each at its own optimum": difference(test["cost_optimal"], money=True),
            }
            for test in tests
        ]
    )


def selected_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = [
        ("PR-AUC", with_ci(metrics["pr_auc"])),
        ("ROC-AUC", with_ci(metrics["roc_auc"])),
        ("Precision", with_ci(metrics["precision"])),
        ("Recall", with_ci(metrics["recall"])),
        ("F1", with_ci(metrics["f1"])),
        ("Total cost", eur_ci(metrics["cost"])),
        ("Cost with no model (flag nothing)", eur_ci(metrics["cost_no_model"])),
        ("Cost of flagging everything", eur(metrics["cost_flag_everything"]["value"])),
        ("Savings against no model", eur_ci(metrics["savings_vs_no_model"])),
        (
            "Savings as a share of the no-model cost",
            with_ci(metrics["savings_share_of_no_model"], "{:.1%}"),
        ),
        ("Savings against flagging everything", eur_ci(metrics["savings_vs_flag_everything"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value [95 % CI]"])


def confusion_table(metrics: dict[str, dict[str, float]]) -> pd.DataFrame:
    count = {name: int(metrics[name]["value"]) for name in ("tn", "fp", "fn", "tp")}
    return pd.DataFrame(
        [
            ("Actually legitimate", f"{count['tn']:,}", f"{count['fp']:,} (false alarms)"),
            ("Actually fraud", f"{count['fn']:,} (missed)", f"{count['tp']:,} (caught)"),
        ],
        columns=["", "Passed", "Flagged"],
    )


def models_table(models: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for entry in models.values():
        metrics = entry["metrics"]
        rows.append(
            {
                "Model": entry["label"],
                "Threshold (from validation)": f"{entry['threshold']:.4g}",
                "PR-AUC": with_ci(metrics["pr_auc"]),
                "ROC-AUC": f"{metrics['roc_auc']['value']:.3f}",
                "Precision": f"{metrics['precision']['value']:.3f}",
                "Recall": with_ci(metrics["recall"], "{:.2f}"),
                "F1": f"{metrics['f1']['value']:.3f}",
                "TP / FP / FN": f"{int(metrics['tp']['value'])} / {int(metrics['fp']['value'])} / {int(metrics['fn']['value'])}",
                "Cost": eur(metrics["cost"]["value"]),
                "Savings against no model": eur_ci(metrics["savings_vs_no_model"]),
            }
        )
    return pd.DataFrame(rows)


def comparison_columns(metrics: dict[str, dict[str, float]]) -> dict[str, str]:
    return {
        "PR-AUC": with_ci(metrics["pr_auc"]),
        "Precision": f"{metrics['precision']['value']:.3f}",
        "Recall": with_ci(metrics["recall"], "{:.2f}"),
        "TP / FP / FN": f"{int(metrics['tp']['value'])} / {int(metrics['fp']['value'])} / {int(metrics['fn']['value'])}",
        "Savings share of no-model cost": with_ci(metrics["savings_share_of_no_model"], "{:.1%}"),
    }


def calibration_table(table: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Score bin": f"{row['low']:g} to {row['high']:g}",
                "Rows": f"{row['n']:,}",
                "Fraud": row["fraud"],
                "Mean score": f"{row['mean_score']:.4f}",
                "Observed fraud rate": f"{row['fraud_rate']:.4f}",
            }
            for row in table["bins"]
        ]
    )


def render(m: dict[str, Any]) -> str:
    """The generated part of the report, from the saved metrics."""
    split = m["evaluation_split"]
    validation, test = m["validation"], m["test"]
    selected_val = validation[SELECTED]
    selected_test = test["models"][SELECTED]["metrics"]
    stability = m["threshold_stability"]
    duplicates, random_split, shap_part = test["with_duplicates"], m["random_split"], m["shap"]
    star = "`*` marks a 95 % interval that excludes zero."
    banner = (
        [
            "> **DRY RUN.** Validation rows stand in for the test split. None of the numbers "
            "below are test results."
        ]
        if m["dry_run"]
        else []
    )
    mismatched = [
        entry["label"] for entry in validation.values() if not entry["scores_match_phase3"]
    ]
    sections = [
        *banner,
        "## Setup",
        f"- Model: **{selected_val['label']}**, `{selected_val['config_id']}`, model version {m['model_version']}, "
        "fitted on the training split only. It is **not** refitted on train + validation: the threshold below is "
        "chosen on validation scores of exactly this model, and a refitted model would have a different score distribution.",
        f"- Rows: train {m['splits']['train']['rows']:,} ({m['splits']['train']['fraud']} fraud), validation "
        f"{m['splits']['val']['rows']:,} ({m['splits']['val']['fraud']} fraud), {split} "
        f"{m['splits']['evaluation']['rows']:,} ({m['splits']['evaluation']['fraud']} fraud). Exact duplicates removed before splitting.",
        f"- Cost model: a missed fraud costs its amount; every flagged transaction costs a €{REVIEW_COST_EUR:g} review, fraud or not.",
        f"- Intervals: 95 %, stratified bootstrap with {m['n_boot']} resamples. Evaluated at commit `{m['git_commit']}`.",
        "- Validation scores of every model reproduce the Phase 3 scores."
        if not mismatched
        else f"- **Warning:** refitted scores differ from Phase 3 for: {', '.join(mismatched)}.",
        "## Threshold (chosen on validation only)",
        f"The cost-minimising threshold of the selected model is **{selected_val['threshold']:.4g}**. "
        f"Over bootstrap resamples of validation the optimal threshold ranges from {stability['q025']:.4g} to "
        f"{stability['q975']:.4g} (median {stability['median']:.4g}), so the cost curve is flat around its minimum "
        "and the exact value is not well determined by this few fraud cases.",
        markdown_table(threshold_table(validation)),
        "Costs in this table are in-sample: the threshold was chosen on the same validation rows.",
        "![Cost against threshold](figures/eval_cost_vs_threshold.png)",
        "## Thesis hypothesis test on validation",
        "Selected model minus each reference, on the same validation rows (paired bootstrap). For cost, a negative "
        "difference means the selected model is cheaper. In the last column every model gets its own cost-optimal "
        f"threshold, chosen again inside each resample. {star}",
        markdown_table(hypothesis_table(m["hypothesis"], validation)),
        "![Cost by model](figures/eval_cost_by_model.png)",
        f"## Final results on the {split} split",
        f"Selected model at the validation threshold {selected_val['threshold']:.4g}. "
        + ("" if m["dry_run"] else "The test split was loaded once, for this evaluation."),
        markdown_table(selected_table(selected_test)),
        markdown_table(confusion_table(selected_test)),
        f"In hindsight the cheapest threshold on these rows would have been {test['hindsight']['threshold']:.4g} "
        f"(cost {eur(test['hindsight']['cost'])}); the validation threshold costs {eur(selected_test['cost']['value'])}. "
        "This is a diagnostic only and was not used.",
        "### Reference models (reported after selection, not used for it)",
        "Each model at its own validation threshold.",
        markdown_table(models_table(test["models"])),
        f"Selected minus reference on the {split} split, each at its validation threshold. {star}",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "Selected minus": test["models"][row["reference"]]["label"],
                        "PR-AUC": difference(row["pr_auc"], money=False),
                        "Cost": difference(row["cost"], money=True),
                    }
                    for row in test["paired"]
                ]
            )
        ),
        "![PR curves](figures/eval_test_pr_curves.png)",
        "### With and without deduplication",
        f"The same model and threshold on the {split} period with its {duplicates['duplicate_rows']} exact duplicate rows "
        f"({duplicates['duplicate_fraud']} of them fraud) put back.",
        markdown_table(
            pd.DataFrame(
                [
                    {"Rows": "Deduplicated (primary)", **comparison_columns(selected_test)},
                    {"Rows": "With exact duplicates", **comparison_columns(duplicates["metrics"])},
                ]
            )
        ),
        "### Secondary analysis: stratified random split",
        "**Not the primary result.** The same configuration refitted on a stratified random 70/15/15 split of the "
        f"deduplicated data ({random_split['fraud']['test']} fraud cases in its test part), with its own validation "
        f"threshold ({random_split['threshold']:.4g}). A random split tests on transactions from the same hours the "
        "model was trained on, so it does not estimate performance on future data.",
        markdown_table(
            pd.DataFrame(
                [
                    {
                        "Split": f"Time-based (primary), {split}",
                        **comparison_columns(selected_test),
                    },
                    {
                        "Split": "Stratified random (secondary), test part",
                        **comparison_columns(random_split["metrics"]),
                    },
                ]
            )
        ),
        "## Score or probability?",
        f"Reliability of the selected model on validation, in fixed score bins. Brier score {m['calibration']['brier']:.5f} "
        f"(always predicting the base rate: {m['calibration']['brier_of_base_rate']:.5f}). The scores sum to "
        f"{m['calibration']['expected_fraud']:.1f} expected fraud cases; validation has {m['calibration']['actual_fraud']}.",
        markdown_table(calibration_table(m["calibration"])),
        "![Reliability](figures/eval_reliability.png)",
        "## Explainability (SHAP)",
        "TreeExplainer on the XGBoost step of the pipeline. SHAP values are in log-odds of the fraud score; base value "
        f"{shap_part['base_value']:.2f}. Base value plus the SHAP values of a row reproduces the model output to within "
        f"{shap_part['additivity_error']:.1e}.",
        markdown_table(
            pd.DataFrame(
                [(name, f"{value:.3f}") for name, value in shap_part["top_features"].items()],
                columns=["Feature", f"Mean absolute SHAP value ({split} rows)"],
            )
        ),
        "![SHAP summary](figures/eval_shap_summary.png)",
        *[
            f"**{explain.CASE_LABELS[name]}** (amount €{case['amount']:,.2f}, score {case['score']:.3f}): strongest "
            + ", ".join(
                f"{feature} {value:+.2f}" for feature, value in case["top_features"].items()
            )
            + f".\n\n![{explain.CASE_LABELS[name]}](figures/eval_shap_{name}.png)"
            for name, case in shap_part["cases"].items()
        ],
        f"A sample of {shap_part['sample_rows']} {split} transactions with score, decision and SHAP values is saved to "
        "`app/data/explained_sample.csv` for the Streamlit app ("
        + ", ".join(
            f"{count} {name.replace('_', ' ')}"
            for name, count in shap_part["sample_outcomes"].items()
        )
        + "). It holds no row "
        "identifiers; `Time` is reduced to the second of the day.",
    ]
    return "\n\n".join(section for section in sections if section)


def write_report(path: Path, generated: str) -> None:
    """Replace the generated block of the report, keeping the hand-written sections."""
    text = path.read_text(encoding="utf-8") if path.exists() else REPORT_TEMPLATE
    before = text.split(GENERATED_START)[0]
    after = text.split(GENERATED_END)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"{GENERATED_START}\n{generated.strip()}\n{GENERATED_END}"
    path.write_text(before + block + after, encoding="utf-8", newline="\n")


# --- Driver ------------------------------------------------------------------------------


def run(
    train: pd.DataFrame,
    val: pd.DataFrame,
    evaluation: pd.DataFrame,
    duplicates: pd.DataFrame,
    results: pd.DataFrame,
    paths: EvaluationPaths,
    n_boot: int = N_BOOTSTRAP,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The whole final evaluation on the given frames; returns the metrics that were saved."""
    split = "validation (dry run)" if dry_run else "test"
    models = load_models(results, train, paths.models_dir)
    val_scores = score_all(models, val)
    validation = validation_analysis(models, val, val_scores, paths.scores_dir, n_boot)
    thresholds = {key: entry["threshold"] for key, entry in validation.items()}
    eval_scores = score_all(models, evaluation)
    test = evaluation_analysis(models, thresholds, evaluation, duplicates, eval_scores, n_boot)
    metrics: dict[str, Any] = {
        "dry_run": dry_run,
        "evaluation_split": split,
        "git_commit": git_commit(),
        "model_version": MODEL_VERSION,
        "n_boot": n_boot,
        "random_seed": RANDOM_SEED,
        "review_cost_eur": REVIEW_COST_EUR,
        "splits": {
            name: {"rows": len(part), "fraud": int(part[TARGET_COLUMN].sum())}
            for name, part in (("train", train), ("val", val), ("evaluation", evaluation))
        },
        "validation": validation,
        "threshold_stability": threshold_stability(val, val_scores[SELECTED], n_boot),
        "hypothesis": hypothesis_tests(val, val_scores, n_boot),
        "calibration": calibration(val[TARGET_COLUMN].to_numpy(), val_scores[SELECTED]),
        "test": test,
        "random_split": random_split_analysis(
            selected_row(results),
            pd.concat([train, val] if dry_run else [train, val, evaluation]),
            n_boot,
        ),
        "shap": shap_analysis(
            models[SELECTED].pipeline,
            evaluation,
            eval_scores[SELECTED],
            thresholds[SELECTED],
            paths,
            split,
        ),
    }
    plot_cost_vs_threshold(val, val_scores[SELECTED], validation[SELECTED], paths.figures_dir)
    plot_cost_by_model(validation, paths.figures_dir)
    plot_reliability(metrics["calibration"], paths.figures_dir)
    plot_test_pr_curves(evaluation, eval_scores, test, paths.figures_dir, split.capitalize())
    update_config(metrics, train, paths)
    paths.metrics.parent.mkdir(parents=True, exist_ok=True)
    paths.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_report(paths.report, render(metrics))
    return metrics


def dry_run_paths(directory: Path) -> EvaluationPaths:
    """Paths of a dry run: inputs as usual, every output under ``directory``."""
    return replace(
        EvaluationPaths(),
        figures_dir=directory / "figures",
        report=directory / "final_evaluation.md",
        metrics=directory / "final_metrics.json",
        access_log=directory / "test_set_access.log",
        sample=directory / "explained_sample.csv",
        config_out=directory / FINAL_CONFIG_NAME,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        type=Path,
        metavar="DIR",
        default=None,
        help="use validation rows in place of the test split and write all output to DIR",
    )
    parser.add_argument(
        "--render-only", action="store_true", help="rewrite the report from the saved metrics"
    )
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    paths = EvaluationPaths() if args.dry_run is None else dry_run_paths(args.dry_run)
    if args.render_only:
        write_report(paths.report, render(json.loads(paths.metrics.read_text(encoding="utf-8"))))
        print(f"Report: {paths.report}")
        return
    dry_run = args.dry_run is not None
    train, val = load_train_val()
    evaluation, duplicates = load_evaluation_rows(dry_run, val, paths.access_log)
    if not dry_run:
        print(f"Test split loaded ({len(evaluation):,} rows); logged in {paths.access_log}")
    metrics = run(
        train, val, evaluation, duplicates, load_results(paths.results), paths, args.n_boot, dry_run
    )
    selected = metrics["test"]["models"][SELECTED]["metrics"]
    print(f"Threshold (validation): {metrics['validation'][SELECTED]['threshold']:.4g}")
    print(
        f"{metrics['evaluation_split']}: PR-AUC {with_ci(selected['pr_auc'])}, recall "
        f"{with_ci(selected['recall'], '{:.2f}')}, savings against no model "
        f"EUR {selected['savings_vs_no_model']['value']:,.0f}"
    )
    print(f"Report: {paths.report}")


if __name__ == "__main__":
    main()
