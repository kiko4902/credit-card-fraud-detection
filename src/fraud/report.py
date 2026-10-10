"""Phase 3 report: rankings, paired comparisons, figures, summary and the saved final model.

Everything here reads ``reports/results.csv`` and the saved validation scores. Nothing is
tuned or selected by looking at the test split, which is never loaded.
"""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.config import (
    FINAL_CONFIG_NAME,
    FINAL_MODEL_NAME,
    N_BOOTSTRAP,
    RANDOM_SEED,
    TOP_N_PAIRED,
    ExperimentPaths,
)
from fraud.data import split_xy
from fraud.evaluate import paired_bootstrap_difference
from fraud.features import MODEL_FEATURES
from fraud.models import build_pipeline
from fraud.plotting import (
    GRID_COLOR,
    NEUTRAL_COLOR,
    RANK_COLORS,
    STRATEGY_COLORS,
    SURFACE_COLOR,
    TEXT_COLOR,
    save_figure,
    set_style,
)
from fraud.selection import (
    EXTRA_STAGE,
    MODEL_LABELS,
    OVERSAMPLING,
    SELECTION_STAGES,
    SMOTE_VARIANTS,
    STAGE1_STRATEGIES,
    STRATEGY_ORDER,
    VARIANT_RATIOS,
    config_label,
    rank_by_rule,
    rank_classifiers,
    select_best,
    strategy_label,
)
from fraud.train import config_id, fraud_scores, git_commit, params_json

GENERATED_START = "<!-- generated:start -->"
GENERATED_END = "<!-- generated:end -->"
SUMMARY_TEMPLATE = f"""# Experiment summary (Phase 3)

## Conclusions

_To be written after reviewing the results below._

## Comparison with the thesis

_To be written after reviewing the results below._

{GENERATED_START}
{GENERATED_END}
"""
OTHER_COLOR = "#b5b3ab"


# --- Helpers ---------------------------------------------------------------------------


def load_scores(identifier: str, scores_dir: Path) -> np.ndarray:
    """Validation scores saved by the experiment run for one configuration."""
    return np.load(scores_dir / f"{identifier}.npy")


def find_row(
    results: pd.DataFrame, model: str, strategy: str, ratio: float | None
) -> pd.Series | None:
    """The untuned row of one classifier / strategy / ratio, if it exists."""
    rows = results[
        (results["model"] == model)
        & (results["strategy"] == strategy)
        & (results["stage"] != "stage3")
    ]
    rows = (
        rows[rows["sampling_ratio"].isna()]
        if ratio is None
        else rows[rows["sampling_ratio"] == ratio]
    )
    return rows.iloc[0] if len(rows) == 1 else None


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a frame of already formatted values as a GitHub markdown table."""
    lines = [
        "| " + " | ".join(str(column) for column in frame.columns) + " |",
        "|" + "|".join("---" for _ in frame.columns) + "|",
    ]
    lines += ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join(lines)


def _interval(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def _cv(row: pd.Series | None) -> str:
    return "n/a" if row is None else f"{row['cv_pr_auc_mean']:.3f}"


def _ratio(row: pd.Series) -> float | None:
    return None if pd.isna(row["sampling_ratio"]) else float(row["sampling_ratio"])


def changed_params(row: pd.Series, reference: pd.Series) -> str:
    """Model parameters of ``row`` that differ from ``reference``, as ``name=value`` pairs."""
    params, base = json.loads(row["params"]), json.loads(reference["params"])
    changed = {name: value for name, value in params.items() if base.get(name) != value}
    return ", ".join(f"{name}={value}" for name, value in sorted(changed.items())) or "defaults"


def paired(
    a: pd.Series, b: pd.Series, val_y: pd.Series, scores_dir: Path, n_boot: int
) -> dict[str, float]:
    """Paired bootstrap of validation PR-AUC, configuration ``a`` minus ``b``."""
    return paired_bootstrap_difference(
        val_y,
        load_scores(a["config_id"], scores_dir),
        load_scores(b["config_id"], scores_dir),
        n_boot=n_boot,
    )


def _difference(result: dict[str, float]) -> str:
    marker = " *" if result["ci_low"] > 0 or result["ci_high"] < 0 else ""
    return f"{result['difference']:+.3f} {_interval(result['ci_low'], result['ci_high'])}{marker}"


# --- Tables ----------------------------------------------------------------------------


def stage1_table(stage1: pd.DataFrame) -> pd.DataFrame:
    """Classifier ranking of the screening stage."""
    rows = []
    for rank, best in rank_classifiers(stage1).iterrows():
        row: dict[str, Any] = {"Rank": rank, "Classifier": MODEL_LABELS[best["model"]]}
        for strategy, ratio in STAGE1_STRATEGIES:
            row[f"CV {strategy_label(strategy, ratio)}"] = _cv(
                find_row(stage1, best["model"], strategy, ratio)
            )
        row["Best strategy"] = strategy_label(best["strategy"], best["sampling_ratio"])
        row["CV folds of best"] = " / ".join(
            f"{s:.2f}" for s in json.loads(best["cv_pr_auc_folds"])
        )
        row["Val PR-AUC of best"] = (
            f"{best['pr_auc']:.3f} {_interval(best['pr_auc_ci_low'], best['pr_auc_ci_high'])}"
        )
        row["Fit (s)"] = f"{best['resample_time_s'] + best['fit_time_s']:.1f}"
        row["Predict (s)"] = f"{best['predict_time_s']:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def boundary_shift_table(stage1: pd.DataFrame) -> pd.DataFrame:
    """No resampling against SMOTE 0.1: ranking quality next to the decisions at 0.5."""
    rows = []
    for model in rank_classifiers(stage1)["model"]:
        plain, smote = find_row(stage1, model, "none", None), find_row(stage1, model, "smote", 0.1)
        if plain is None or smote is None:
            continue
        rows.append(
            {
                "Classifier": MODEL_LABELS[model],
                "CV PR-AUC none → SMOTE": f"{plain['cv_pr_auc_mean']:.3f} → {smote['cv_pr_auc_mean']:.3f}",
                "Val PR-AUC none → SMOTE": f"{plain['pr_auc']:.3f} → {smote['pr_auc']:.3f}",
                "Recall at 0.5": f"{plain['recall']:.2f} → {smote['recall']:.2f}",
                "Precision at 0.5": f"{plain['precision']:.2f} → {smote['precision']:.2f}",
                "Flagged at 0.5": f"{plain['tp'] + plain['fp']} → {smote['tp'] + smote['fp']}",
            }
        )
    return pd.DataFrame(rows)


def stage2_table(screened: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Mean CV PR-AUC of every strategy for the stage 2 classifiers."""
    rows = []
    for strategy, ratio in STRATEGY_ORDER:
        row = {"Strategy": strategy_label(strategy, ratio)}
        for model in models:
            row[MODEL_LABELS[model]] = _cv(find_row(screened, model, strategy, ratio))
        rows.append(row)
    return pd.DataFrame(rows)


def variant_table(
    screened: pd.DataFrame, models: list[str], val_y: pd.Series, scores_dir: Path, n_boot: int
) -> pd.DataFrame:
    """Every SMOTE variant against plain SMOTE at the same ratio, per classifier."""
    rows = []
    for model in models:
        for variant in SMOTE_VARIANTS:
            for ratio in VARIANT_RATIOS:
                config = find_row(screened, model, variant, ratio)
                smote = find_row(screened, model, "smote", ratio)
                if config is None or smote is None:
                    continue
                result = paired(config, smote, val_y, scores_dir, n_boot)
                rows.append(
                    {
                        "Classifier": MODEL_LABELS[model],
                        "Variant": strategy_label(variant, ratio),
                        "CV PR-AUC": f"{config['cv_pr_auc_mean']:.3f}",
                        "CV of SMOTE": f"{smote['cv_pr_auc_mean']:.3f}",
                        "CV difference": f"{config['cv_pr_auc_mean'] - smote['cv_pr_auc_mean']:+.3f}",
                        "Folds won": sum(
                            a > b
                            for a, b in zip(
                                json.loads(config["cv_pr_auc_folds"]),
                                json.loads(smote["cv_pr_auc_folds"]),
                                strict=True,
                            )
                        ),
                        "Val difference [95 % CI]": _difference(result),
                    }
                )
    return pd.DataFrame(rows)


def oversampling_table(
    screened: pd.DataFrame, models: list[str], val_y: pd.Series, scores_dir: Path, n_boot: int
) -> pd.DataFrame:
    """Best oversampling run of each classifier against no resampling and class weights."""
    rows = []
    for model in models:
        of_model = screened[screened["model"] == model]
        sampled = of_model[of_model["strategy"].isin(OVERSAMPLING)]
        if sampled.empty:
            continue
        best = sampled.loc[sampled["cv_pr_auc_mean"].idxmax()]
        plain = find_row(screened, model, "none", None)
        weighted = find_row(screened, model, "class_weight", None)
        row = {
            "Classifier": MODEL_LABELS[model],
            "Best oversampling (by CV)": strategy_label(best["strategy"], best["sampling_ratio"]),
            "CV: none / weights / oversampling": f"{_cv(plain)} / {_cv(weighted)} / {_cv(best)}",
            "Val: oversampling minus none [95 % CI]": "n/a"
            if plain is None
            else _difference(paired(best, plain, val_y, scores_dir, n_boot)),
            "Val: oversampling minus weights [95 % CI]": "n/a"
            if weighted is None
            else _difference(paired(best, weighted, val_y, scores_dir, n_boot)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity_table(screened: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """How much the choice of oversampling method moves each classifier's CV PR-AUC."""
    rows = []
    for model in models:
        sampled = screened[(screened["model"] == model) & screened["strategy"].isin(OVERSAMPLING)]
        if sampled.empty:
            continue
        values = sampled["cv_pr_auc_mean"]
        best, worst = sampled.loc[values.idxmax()], sampled.loc[values.idxmin()]
        rows.append(
            {
                "Classifier": MODEL_LABELS[model],
                "Oversampling runs": len(sampled),
                "Worst": f"{values.min():.3f} ({strategy_label(worst['strategy'], worst['sampling_ratio'])})",
                "Best": f"{values.max():.3f} ({strategy_label(best['strategy'], best['sampling_ratio'])})",
                "Range": f"{values.max() - values.min():.3f}",
                "Std": f"{values.std():.3f}",
            }
        )
    return pd.DataFrame(rows)


def ranking_table(ranked: pd.DataFrame) -> pd.DataFrame:
    """Formatted ranking: CV result first, validation as confirmation."""
    return pd.DataFrame(
        {
            "Rank": ranked.index,
            "Classifier": ranked["model"].map(MODEL_LABELS).to_numpy(),
            "Strategy": [
                strategy_label(s, r)
                for s, r in zip(ranked["strategy"], ranked["sampling_ratio"], strict=True)
            ],
            "CV PR-AUC (mean ± std)": [
                f"{m:.3f} ± {s:.3f}"
                for m, s in zip(ranked["cv_pr_auc_mean"], ranked["cv_pr_auc_std"], strict=True)
            ],
            "CV folds": [
                " / ".join(f"{score:.2f}" for score in json.loads(folds))
                for folds in ranked["cv_pr_auc_folds"]
            ],
            "Val PR-AUC [95 % CI]": [
                f"{p:.3f} {_interval(low, high)}"
                for p, low, high in zip(
                    ranked["pr_auc"], ranked["pr_auc_ci_low"], ranked["pr_auc_ci_high"], strict=True
                )
            ],
            "Val ROC-AUC": [f"{value:.3f}" for value in ranked["roc_auc"]],
            "Fit (s)": [
                f"{value:.1f}" for value in ranked["resample_time_s"] + ranked["fit_time_s"]
            ],
            "Predict (s)": [f"{value:.1f}" for value in ranked["predict_time_s"]],
        }
    )


def paired_comparisons(
    configs: pd.DataFrame,
    screened: pd.DataFrame,
    val_y: pd.Series,
    scores_dir: Path,
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compare configurations with their classifier's no-resampling run and with the baseline.

    Paired stratified bootstrap of the validation PR-AUC difference (configuration minus
    reference). A comparison of a configuration with itself is skipped.
    """
    rows = []
    for _, config in configs.iterrows():
        references = {
            "same classifier, no resampling": find_row(screened, config["model"], "none", None),
            "baseline (logistic regression, no resampling)": find_row(
                screened, "logreg", "none", None
            ),
        }
        for name, reference in references.items():
            if reference is None or reference["config_id"] == config["config_id"]:
                continue
            result = paired(config, reference, val_y, scores_dir, n_boot)
            rows.append(
                {
                    "config_id": config["config_id"],
                    "config": config_label(config),
                    "versus": name,
                    **result,
                }
            )
    columns = [
        "config_id",
        "config",
        "versus",
        "a",
        "b",
        "difference",
        "ci_low",
        "ci_high",
        "share_a_better",
    ]
    return pd.DataFrame(rows, columns=columns)


def comparison_table(comparisons: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Configuration": comparisons["config"],
            "Versus": comparisons["versus"],
            "Val PR-AUC": [f"{value:.3f}" for value in comparisons["a"]],
            "Reference PR-AUC": [f"{value:.3f}" for value in comparisons["b"]],
            "Difference": [f"{value:+.3f}" for value in comparisons["difference"]],
            "95 % CI of difference": [
                _interval(low, high)
                for low, high in zip(comparisons["ci_low"], comparisons["ci_high"], strict=True)
            ],
            "Resamples where config wins": [
                f"{value:.0%}" for value in comparisons["share_a_better"]
            ],
        }
    )


def tuning_table(results: pd.DataFrame) -> pd.DataFrame:
    """Every tried hyperparameter setting next to the untuned run of the same strategy."""
    tuning = results[results["stage"] == "stage3"]
    rows = []
    for model in tuning["model"].unique():
        tried = tuning[tuning["model"] == model]
        first = tried.iloc[0]
        reference = find_row(results, model, first["strategy"], _ratio(first))
        candidates = pd.concat([reference.to_frame().T, tried]).sort_values(
            "cv_pr_auc_mean", ascending=False
        )
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "Classifier": MODEL_LABELS[model],
                    "Strategy": strategy_label(row["strategy"], row["sampling_ratio"]),
                    "Changed parameters": changed_params(row, reference),
                    "CV PR-AUC (mean ± std)": f"{row['cv_pr_auc_mean']:.3f} ± {row['cv_pr_auc_std']:.3f}",
                    "Val PR-AUC [95 % CI]": f"{row['pr_auc']:.3f} {_interval(row['pr_auc_ci_low'], row['pr_auc_ci_high'])}",
                    "Fit (s)": f"{row['resample_time_s'] + row['fit_time_s']:.1f}",
                }
            )
    return pd.DataFrame(rows)


# --- Figures ---------------------------------------------------------------------------


def plot_stage1_ranking(stage1: pd.DataFrame, figures_dir: Path) -> Path:
    """Classifiers ordered by their best CV PR-AUC; validation PR-AUC with CI per strategy."""
    set_style()
    models = list(rank_classifiers(stage1)["model"])
    offsets = np.linspace(-0.25, 0.25, len(STAGE1_STRATEGIES))
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(models) + 1.8))
    for position, model in enumerate(models):
        for offset, (strategy, ratio) in zip(offsets, STAGE1_STRATEGIES, strict=True):
            row = find_row(stage1, model, strategy, ratio)
            if row is None:
                continue
            color, y = STRATEGY_COLORS[strategy], position + offset
            ax.hlines(y, row["pr_auc_ci_low"], row["pr_auc_ci_high"], color=color, linewidth=2)
            ax.plot(row["pr_auc"], y, "o", color=color, markersize=6)
            ax.plot(
                row["cv_pr_auc_mean"],
                y,
                "D",
                markerfacecolor=SURFACE_COLOR,
                markeredgecolor=color,
                markersize=6,
            )
    ax.set_yticks(range(len(models)), [MODEL_LABELS[model] for model in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC")
    ax.grid(axis="y", visible=False)
    handles = [
        Line2D(
            [], [], color=STRATEGY_COLORS[s], marker="o", linewidth=2, label=strategy_label(s, r)
        )
        for s, r in STAGE1_STRATEGIES
    ]
    handles += [
        Line2D([], [], color=NEUTRAL_COLOR, marker="o", linewidth=2, label="Validation, 95 % CI"),
        Line2D(
            [],
            [],
            color=NEUTRAL_COLOR,
            marker="D",
            markerfacecolor=SURFACE_COLOR,
            linestyle="none",
            label="Mean of 3 CV folds",
        ),
    ]
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=5, fontsize=8.5
    )
    ax.set_title("Stage 1: classifiers ranked by their best mean CV PR-AUC")
    path = save_figure(fig, "exp_stage1_ranking", figures_dir)
    plt.close(fig)
    return path


def plot_stage2_heatmap(
    screened: pd.DataFrame, models: list[str], n_ranked: int, figures_dir: Path
) -> Path:
    """Mean CV PR-AUC for the stage 2 classifiers (rows) and every strategy (columns).

    The first ``n_ranked`` rows are the top classifiers; rows below the line are the extras.
    """
    set_style()
    values = np.array(
        [
            [
                np.nan
                if (row := find_row(screened, model, s, r)) is None
                else row["cv_pr_auc_mean"]
                for s, r in STRATEGY_ORDER
            ]
            for model in models
        ]
    )
    cmap = LinearSegmentedColormap.from_list("single_hue", [SURFACE_COLOR, "#9ec5f4", "#0d366b"])
    cmap = cmap.with_extremes(bad=GRID_COLOR)
    fig, ax = plt.subplots(figsize=(12, 1.6 + 0.75 * len(models)))
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    midpoint = (np.nanmin(values) + np.nanmax(values)) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                continue
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > midpoint else TEXT_COLOR,
                fontweight="bold" if value == np.nanmax(values[row]) else "normal",
            )
    ax.set_xticks(
        range(len(STRATEGY_ORDER)),
        [strategy_label(s, r) for s, r in STRATEGY_ORDER],
        rotation=35,
        ha="right",
    )
    labels = [
        MODEL_LABELS[model] + ("" if position < n_ranked else " (extra)")
        for position, model in enumerate(models)
    ]
    ax.set_yticks(range(len(models)), labels)
    if n_ranked < len(models):
        ax.axhline(n_ranked - 0.5, color=SURFACE_COLOR, linewidth=4)
    ax.grid(visible=False)
    ax.set_title("Stage 2: mean PR-AUC over 3 time-ordered CV folds (best per classifier in bold)")
    fig.colorbar(image, ax=ax, pad=0.01, label="Mean CV PR-AUC")
    path = save_figure(fig, "exp_stage2_cv_heatmap", figures_dir)
    plt.close(fig)
    return path


def plot_pr_curves(
    best_per_model: pd.DataFrame,
    highlighted: list[str],
    val_y: pd.Series,
    scores_dir: Path,
    figures_dir: Path,
) -> Path:
    """Precision-recall curves on validation for the best configuration of each classifier.

    The stage 2 classifiers get a colour each; the others are drawn in grey.
    """
    set_style()
    colors = dict(zip(highlighted, RANK_COLORS, strict=False))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ordered = pd.concat(
        [
            best_per_model[~best_per_model["model"].isin(colors)],
            best_per_model[best_per_model["model"].isin(colors)],
        ]
    )
    for _, row in ordered.iterrows():
        scores = load_scores(row["config_id"], scores_dir)
        precision, recall, _ = precision_recall_curve(val_y, scores)
        label = f"{config_label(row)} ({average_precision_score(val_y, scores):.3f})"
        emphasised = row["model"] in colors
        ax.step(
            recall,
            precision,
            where="post",
            color=colors.get(row["model"], OTHER_COLOR),
            linewidth=2 if emphasised else 1,
            label=label,
        )
    prevalence = float(np.mean(val_y))
    ax.axhline(prevalence, color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
    ax.text(
        1.0,
        prevalence + 0.015,
        f"random guessing: precision = fraud rate ({prevalence:.2%})",
        ha="right",
        fontsize=8.5,
    )
    ax.set_xlabel("Recall (share of fraud caught)")
    ax.set_ylabel("Precision (share of flagged that is fraud)")
    ax.set_xlim(0, 1.005)
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Validation ({int(np.sum(val_y))} fraud cases): best configuration per classifier, with PR-AUC"
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="lower left", fontsize=8.5)
    path = save_figure(fig, "exp_pr_curves_best_per_model", figures_dir)
    plt.close(fig)
    return path


def plot_pr_auc_vs_fit_time(
    results: pd.DataFrame, highlighted: list[str], figures_dir: Path
) -> Path:
    """Mean CV PR-AUC against training time (resampling + fit on the full training split)."""
    set_style()
    colors = dict(zip(highlighted, RANK_COLORS, strict=False))
    seconds = (results["resample_time_s"] + results["fit_time_s"]).clip(lower=0.05)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model in [*[m for m in results["model"].unique() if m not in colors], *highlighted]:
        of_model = results["model"] == model
        ax.scatter(
            seconds[of_model],
            results.loc[of_model, "cv_pr_auc_mean"],
            s=42,
            color=colors.get(model, OTHER_COLOR),
            edgecolor=SURFACE_COLOR,
            linewidth=0.8,
            label=MODEL_LABELS[model] if model in colors else None,
            zorder=3 if model in colors else 2,
        )
        best = results.loc[results.loc[of_model, "cv_pr_auc_mean"].idxmax()]
        ax.annotate(
            MODEL_LABELS[model],
            (seconds[best.name], best["cv_pr_auc_mean"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training time on the full training split, resampling + fit (seconds, log scale)")
    ax.set_ylabel("Mean CV PR-AUC")
    ax.set_title(
        "PR-AUC against training time (one point per configuration; label = best of the classifier)"
    )
    ax.legend(loc="lower right", title="Stage 2 classifiers (others in grey)")
    path = save_figure(fig, "exp_pr_auc_vs_fit_time", figures_dir)
    plt.close(fig)
    return path


# --- Final model -----------------------------------------------------------------------


def rebuild_pipeline(row: pd.Series) -> Any:
    """Recreate the unfitted pipeline of a results row and check it is the same configuration."""
    pipeline = build_pipeline(
        row["model"], row["strategy"], _ratio(row), model_params=json.loads(row["params"])
    )
    rebuilt_id = config_id(row["model"], row["strategy"], _ratio(row), params_json(pipeline))
    if rebuilt_id != row["config_id"]:
        raise ValueError(f"Rebuilt pipeline is {rebuilt_id}, expected {row['config_id']}.")
    return pipeline


def save_final_model(
    final: pd.Series, train: pd.DataFrame, val: pd.DataFrame, models_dir: Path
) -> dict[str, Any]:
    """Fit the selected configuration on the training split and save it with its config."""
    pipeline = rebuild_pipeline(final)
    pipeline.fit(*split_xy(train))
    val_x, val_y = split_xy(val)
    val_pr_auc = float(average_precision_score(val_y, fraud_scores(pipeline, val_x)))

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / FINAL_MODEL_NAME
    joblib.dump(pipeline, model_path, compress=3)
    config = {
        "config_id": final["config_id"],
        "model": final["model"],
        "strategy": final["strategy"],
        "sampling_ratio": _ratio(final),
        "model_params": json.loads(final["params"]),
        "features": MODEL_FEATURES,
        "trained_on": "train split (deduplicated, earliest 70 % by time)",
        "n_train": len(train),
        "n_train_fraud": int(train["Class"].sum()),
        "cv_pr_auc_mean": float(final["cv_pr_auc_mean"]),
        "val_pr_auc": val_pr_auc,
        "val_pr_auc_in_results": float(final["pr_auc"]),
        "threshold": None,
        "random_seed": RANDOM_SEED,
        "git_commit": git_commit(),
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in (
                    "scikit-learn",
                    "imbalanced-learn",
                    "lightgbm",
                    "xgboost",
                    "pandas",
                    "numpy",
                )
            },
        },
        "model_file": FINAL_MODEL_NAME,
        "model_file_bytes": model_path.stat().st_size,
    }
    (models_dir / FINAL_CONFIG_NAME).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return config


# --- Summary ---------------------------------------------------------------------------


def write_summary(path: Path, generated: str) -> None:
    """Replace the generated block of the summary, keeping the hand-written sections."""
    text = path.read_text(encoding="utf-8") if path.exists() else SUMMARY_TEMPLATE
    if GENERATED_START not in text or GENERATED_END not in text:
        raise ValueError(f"{path} has no generated-block markers.")
    before = text.split(GENERATED_START)[0]
    after = text.split(GENERATED_END)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"{GENERATED_START}\n{generated.strip()}\n{GENERATED_END}"
    path.write_text(before + block + after, encoding="utf-8", newline="\n")


def confirmation_note(
    final: pd.Series, results: pd.DataFrame, val_y: pd.Series, scores_dir: Path, n_boot: int
) -> str:
    """Does validation agree with the configuration chosen by cross-validation?"""
    best_on_val = results.loc[results["pr_auc"].idxmax()]
    if best_on_val["config_id"] == final["config_id"]:
        return "The selected configuration is also the best one on validation."
    check = paired(final, best_on_val, val_y, scores_dir, n_boot)
    verdict = (
        "**The interval excludes zero: the CV winner is clearly worse on validation. Flagged for review.**"
        if check["ci_high"] < 0
        else "The interval includes zero, so validation does not contradict the CV choice."
    )
    return (
        f"Best on validation is {config_label(best_on_val)} (PR-AUC {best_on_val['pr_auc']:.3f}, "
        f"CV {best_on_val['cv_pr_auc_mean']:.3f}). Selected minus best-on-validation: "
        f"{check['difference']:+.3f}, 95 % CI {_interval(check['ci_low'], check['ci_high'])}. {verdict}"
    )


def build_report(
    results: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    paths: ExperimentPaths,
    n_boot: int = N_BOOTSTRAP,
) -> list[str]:
    """Create figures, the summary and the saved final model; return progress messages."""
    results = results[(results["eval_split"] == "val") & results["cv_pr_auc_mean"].notna()]
    everything = results.reset_index(drop=True)
    # Extra stage 2 rows inform the thesis comparison only; they never compete.
    results = everything[everything["stage"].isin(SELECTION_STAGES)]
    untuned = everything[everything["stage"] != "stage3"]
    stage1 = results[results["stage"] == "stage1"]
    screened = results[results["stage"].isin(["stage1", "stage2"])]
    by_rank = list(rank_classifiers(stage1)["model"])
    stage2_models = [
        m for m in by_rank if m in set(results.loc[results["stage"] == "stage2", "model"])
    ]
    extra_models = [
        m for m in by_rank if m in set(everything.loc[everything["stage"] == EXTRA_STAGE, "model"])
    ]
    thesis_models = [*stage2_models, *extra_models]
    extra_note = (
        f" {' and '.join(MODEL_LABELS[m] for m in extra_models)} did not reach the top "
        f"{len(stage2_models)} and were run in addition, for the comparison with the thesis only: "
        "they are not part of the ranking or of the model selection."
        if extra_models
        else ""
    )
    _, val_y = split_xy(val)
    scores_dir = paths.scores_dir

    final = select_best(results)
    ranked = rank_by_rule(screened)
    top = ranked.head(TOP_N_PAIRED)
    compared = (
        top if final["config_id"] in set(top["config_id"]) else pd.concat([top, final.to_frame().T])
    )
    comparisons = paired_comparisons(compared, screened, val_y, scores_dir, n_boot)
    best_per_model = pd.DataFrame(
        [
            select_best(results[results["model"] == model])
            for model in rank_classifiers(stage1)["model"]
        ]
    )

    figures = [
        plot_stage1_ranking(stage1, paths.figures_dir),
        plot_stage2_heatmap(untuned, thesis_models, len(stage2_models), paths.figures_dir),
        plot_pr_curves(best_per_model, stage2_models, val_y, scores_dir, paths.figures_dir),
        plot_pr_auc_vs_fit_time(everything, stage2_models, paths.figures_dir),
    ]
    config = save_final_model(final, train, val, paths.models_dir)
    size_mb = config["model_file_bytes"] / 1e6

    reference = find_row(screened, final["model"], final["strategy"], _ratio(final))
    final_params = "defaults" if reference is None else changed_params(final, reference)
    final_ci = _interval(final["pr_auc_ci_low"], final["pr_auc_ci_high"])
    tuning = tuning_table(results)
    star = "`*` marks a 95 % interval that excludes zero."
    sections = [
        "## Selected configuration",
        f"**{config_label(final)}** (parameters: {final_params}; from {final['stage']}). Mean CV PR-AUC "
        f"{final['cv_pr_auc_mean']:.3f}; validation PR-AUC {final['pr_auc']:.3f}, 95 % CI {final_ci}.",
        f"Validation check: {confirmation_note(final, results, val_y, scores_dir, n_boot)}",
        f"Saved to `models/{FINAL_MODEL_NAME}` ({size_mb:.2f} MB, fitted on the training split) with "
        f"`models/{FINAL_CONFIG_NAME}`. No decision threshold is chosen yet and the test split was not used.",
        f"## Stage 1: screening of {stage1['model'].nunique()} classifiers",
        "Ranked by the best mean CV PR-AUC over the strategies. CV columns are means over the 3 "
        f"time-ordered folds; validation has {int(val_y.sum())} fraud cases, so its intervals are wide. "
        "Fit time is resampling plus fitting on the full training split; predict time is scoring the validation split.",
        markdown_table(stage1_table(stage1)),
        "### Ranking quality against decisions at threshold 0.5",
        "No resampling against SMOTE 0.1 for every classifier. If oversampling mainly moves the decision "
        "boundary, recall and the number of flagged transactions change much more than PR-AUC.",
        markdown_table(boundary_shift_table(stage1)),
        f"## Stage 2: SMOTE family for the top {len(stage2_models)} classifiers",
        f"Mean CV PR-AUC.{extra_note}",
        markdown_table(stage2_table(untuned, thesis_models)),
        "### Each variant against plain SMOTE at the same ratio",
        f'CV difference is variant minus SMOTE; "folds won" is out of 3. The validation difference is a paired bootstrap. {star}',
        markdown_table(variant_table(untuned, thesis_models, val_y, scores_dir, n_boot)),
        "### Oversampling against no resampling and class weights",
        f"The best oversampling run of each classifier (chosen by CV). {star}",
        markdown_table(oversampling_table(untuned, thesis_models, val_y, scores_dir, n_boot)),
        "### How much the choice of oversampling method matters per classifier",
        "Spread of the mean CV PR-AUC over all oversampling runs of a classifier.",
        markdown_table(sensitivity_table(untuned, thesis_models)),
        "## Overall ranking (stages 1 and 2, top 15)",
        "Ordered by the selection rule: mean CV PR-AUC, ties within 0.005 broken toward the simpler strategy.",
        markdown_table(ranking_table(ranked.head(15))),
        "## Paired comparisons on validation",
        f"Paired stratified bootstrap of the PR-AUC difference (configuration minus reference) for the top "
        f"{TOP_N_PAIRED} configurations of stages 1 and 2 and the selected one. An interval that includes zero means "
        "the difference cannot be told apart from noise on this validation set.",
        markdown_table(comparison_table(comparisons)),
        "## Stage 3: hyperparameter tuning",
        "Every setting that was tried, with the untuned run of the same strategy for reference."
        if not tuning.empty
        else "No tuning rows yet.",
        markdown_table(tuning) if not tuning.empty else "",
        "## Figures",
        "\n".join(f"- `reports/figures/{figure.name}`" for figure in figures),
    ]
    write_summary(paths.summary, "\n\n".join(section for section in sections if section))
    return [
        f"Selected: {config_label(final)} [{final['config_id']}]",
        f"Figures: {', '.join(figure.name for figure in figures)}",
        f"Final model: {paths.models_dir / FINAL_MODEL_NAME} ({size_mb:.2f} MB)",
        f"Summary: {paths.summary}",
    ]
