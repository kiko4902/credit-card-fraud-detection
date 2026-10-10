"""Phase 3 report: ranking, paired comparisons, figures, summary and the saved final model.

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
    MODEL_COLORS,
    NEUTRAL_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    save_figure,
    set_style,
)
from fraud.selection import (
    MODEL_LABELS,
    MODEL_ORDER,
    SMOTE_FAMILY,
    STRATEGY_ORDER,
    config_label,
    rank_by_rule,
    select_best,
    strategy_label,
)
from fraud.train import config_id, fraud_scores, git_commit, params_json

GENERATED_START = "<!-- generated:start -->"
GENERATED_END = "<!-- generated:end -->"
SUMMARY_TEMPLATE = f"""# Experiment summary (Phase 3)

## Conclusions

_To be written after reviewing the results below._

{GENERATED_START}
{GENERATED_END}
"""


# --- Helpers ---------------------------------------------------------------------------


def load_scores(identifier: str, scores_dir: Path) -> np.ndarray:
    """Validation scores saved by the grid run for one configuration."""
    return np.load(scores_dir / f"{identifier}.npy")


def grid_row(grid: pd.DataFrame, model: str, strategy: str, ratio: float | None) -> pd.Series:
    """The grid row of one model / strategy / ratio."""
    match = grid[(grid["model"] == model) & (grid["strategy"] == strategy)]
    match = (
        match[match["sampling_ratio"].isna()]
        if ratio is None
        else match[match["sampling_ratio"] == ratio]
    )
    if len(match) != 1:
        raise ValueError(
            f"Expected one grid row for {model}/{strategy}/{ratio}, found {len(match)}."
        )
    return match.iloc[0]


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


def changed_params(row: pd.Series, reference: pd.Series) -> str:
    """Model parameters of ``row`` that differ from ``reference``, as ``name=value`` pairs."""
    params, base = json.loads(row["params"]), json.loads(reference["params"])
    changed = {name: value for name, value in params.items() if base.get(name) != value}
    return ", ".join(f"{name}={value}" for name, value in sorted(changed.items())) or "defaults"


# --- Paired comparisons ----------------------------------------------------------------


def paired_comparisons(
    configs: pd.DataFrame,
    grid: pd.DataFrame,
    val_y: pd.Series,
    scores_dir: Path,
    n_boot: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compare each configuration with its model's no-resampling run and with the baseline.

    Paired stratified bootstrap of the validation PR-AUC difference (configuration minus
    reference). A comparison of a configuration with itself is skipped.
    """
    baseline = grid_row(grid, "logreg", "none", None)
    rows = []
    for _, config in configs.iterrows():
        references = {
            "same model, no resampling": grid_row(grid, config["model"], "none", None),
            "baseline (logistic regression, no resampling)": baseline,
        }
        scores = load_scores(config["config_id"], scores_dir)
        for name, reference in references.items():
            if reference["config_id"] == config["config_id"]:
                continue
            reference_scores = load_scores(reference["config_id"], scores_dir)
            result = paired_bootstrap_difference(val_y, scores, reference_scores, n_boot=n_boot)
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


# --- Figures ---------------------------------------------------------------------------


def plot_pr_auc_by_config(grid: pd.DataFrame, figures_dir: Path) -> Path:
    """Validation PR-AUC with CI and mean CV PR-AUC for every grid configuration."""
    set_style()
    baseline = grid_row(grid, "logreg", "none", None)
    labels = [strategy_label(strategy, ratio) for strategy, ratio in STRATEGY_ORDER]
    positions = np.arange(len(STRATEGY_ORDER))
    models = [model for model in MODEL_ORDER if model in set(grid["model"])]

    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 5), sharey=True, sharex=True)
    for ax, model in zip(np.atleast_1d(axes), models, strict=True):
        rows = [grid_row(grid, model, strategy, ratio) for strategy, ratio in STRATEGY_ORDER]
        point = np.array([row["pr_auc"] for row in rows])
        low = np.array([row["pr_auc_ci_low"] for row in rows])
        high = np.array([row["pr_auc_ci_high"] for row in rows])
        cv_mean = np.array([row["cv_pr_auc_mean"] for row in rows])
        ax.axvline(baseline["pr_auc"], color=NEUTRAL_COLOR, linestyle="--", linewidth=1)
        ax.hlines(positions, low, high, color=MODEL_COLORS[model], linewidth=2)
        ax.plot(point, positions, "o", color=MODEL_COLORS[model], markersize=7)
        ax.plot(
            cv_mean,
            positions,
            "D",
            markerfacecolor="none",
            markeredgecolor=TEXT_COLOR,
            markersize=6,
        )
        ax.set_title(MODEL_LABELS[model])
        ax.set_xlabel("PR-AUC")
        ax.grid(axis="y", visible=False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(positions, labels)
    first.invert_yaxis()
    handles = [
        Line2D(
            [], [], color=NEUTRAL_COLOR, marker="o", linewidth=2, label="Validation PR-AUC, 95 % CI"
        ),
        Line2D(
            [],
            [],
            color=TEXT_COLOR,
            marker="D",
            markerfacecolor="none",
            linestyle="none",
            label="Mean CV PR-AUC (3 time-ordered folds)",
        ),
        Line2D(
            [],
            [],
            color=NEUTRAL_COLOR,
            linestyle="--",
            linewidth=1,
            label="Baseline on validation (logistic regression, no resampling)",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle("PR-AUC of every configuration, by model", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    path = save_figure(fig, "exp_pr_auc_by_config", figures_dir)
    plt.close(fig)
    return path


def plot_cv_heatmap(grid: pd.DataFrame, figures_dir: Path) -> Path:
    """Mean CV PR-AUC for every model (rows) and strategy (columns)."""
    set_style()
    models = [model for model in MODEL_ORDER if model in set(grid["model"])]
    values = np.array(
        [
            [
                grid_row(grid, model, strategy, ratio)["cv_pr_auc_mean"]
                for strategy, ratio in STRATEGY_ORDER
            ]
            for model in models
        ]
    )
    cmap = LinearSegmentedColormap.from_list("single_hue", [SURFACE_COLOR, "#9ec5f4", "#0d366b"])
    fig, ax = plt.subplots(figsize=(11, 1.2 + 0.8 * len(models)))
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    midpoint = (values.min() + values.max()) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > midpoint else TEXT_COLOR,
                fontweight="bold" if value == values[row].max() else "normal",
            )
    ax.set_xticks(
        range(len(STRATEGY_ORDER)),
        [strategy_label(s, r) for s, r in STRATEGY_ORDER],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(models)), [MODEL_LABELS[model] for model in models])
    ax.grid(visible=False)
    ax.set_title("Mean PR-AUC over 3 time-ordered CV folds (best per model in bold)")
    fig.colorbar(image, ax=ax, pad=0.01, label="Mean CV PR-AUC")
    path = save_figure(fig, "exp_cv_pr_auc_heatmap", figures_dir)
    plt.close(fig)
    return path


def plot_pr_curves(
    best_per_model: pd.DataFrame, val_y: pd.Series, scores_dir: Path, figures_dir: Path
) -> Path:
    """Precision-recall curves on validation for the best configuration of each model."""
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in best_per_model.iterrows():
        scores = load_scores(row["config_id"], scores_dir)
        precision, recall, _ = precision_recall_curve(val_y, scores)
        label = f"{config_label(row)} (PR-AUC {average_precision_score(val_y, scores):.3f})"
        ax.step(recall, precision, where="post", color=MODEL_COLORS[row["model"]], label=label)
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
        f"Precision-recall on validation ({int(np.sum(val_y))} fraud cases): "
        "best configuration per model"
    )
    ax.legend(loc="lower left")
    path = save_figure(fig, "exp_pr_curves_best_per_model", figures_dir)
    plt.close(fig)
    return path


# --- Final model -----------------------------------------------------------------------


def rebuild_pipeline(row: pd.Series) -> Any:
    """Recreate the unfitted pipeline of a results row and check it is the same configuration."""
    ratio = None if pd.isna(row["sampling_ratio"]) else float(row["sampling_ratio"])
    pipeline = build_pipeline(
        row["model"], row["strategy"], ratio, model_params=json.loads(row["params"])
    )
    rebuilt_id = config_id(row["model"], row["strategy"], ratio, params_json(pipeline))
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
    ratio = None if pd.isna(final["sampling_ratio"]) else float(final["sampling_ratio"])
    config = {
        "config_id": final["config_id"],
        "model": final["model"],
        "strategy": final["strategy"],
        "sampling_ratio": ratio,
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
                for name in ("scikit-learn", "imbalanced-learn", "lightgbm", "pandas", "numpy")
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


def ranking_table(ranked: pd.DataFrame) -> pd.DataFrame:
    """Formatted ranking: CV result first, validation as confirmation."""
    return pd.DataFrame(
        {
            "Rank": ranked.index,
            "Model": ranked["model"].map(MODEL_LABELS).to_numpy(),
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
            "Val PR-AUC": [f"{value:.3f}" for value in ranked["pr_auc"]],
            "Val 95 % CI": [
                _interval(low, high)
                for low, high in zip(ranked["pr_auc_ci_low"], ranked["pr_auc_ci_high"], strict=True)
            ],
            "Val ROC-AUC": [f"{value:.3f}" for value in ranked["roc_auc"]],
            "Fit time (s)": [f"{value:.1f}" for value in ranked["train_time_s"]],
        }
    )


def family_table(grid: pd.DataFrame) -> pd.DataFrame:
    """Per model: no resampling, class weights, best undersampling and best SMOTE-family run."""
    rows = []
    for model in [model for model in MODEL_ORDER if model in set(grid["model"])]:
        of_model = grid[grid["model"] == model]
        picks = {
            "No resampling": of_model[of_model["strategy"] == "none"],
            "Class weights": of_model[of_model["strategy"] == "class_weight"],
            "Best undersampling": of_model[of_model["strategy"] == "undersample"],
            "Best SMOTE-family": of_model[of_model["strategy"].isin(SMOTE_FAMILY)],
        }
        for name, candidates in picks.items():
            if candidates.empty:
                continue
            best = candidates.loc[candidates["cv_pr_auc_mean"].idxmax()]
            rows.append(
                {
                    "Model": MODEL_LABELS[model],
                    "Approach": name,
                    "Configuration": strategy_label(best["strategy"], best["sampling_ratio"]),
                    "CV PR-AUC": f"{best['cv_pr_auc_mean']:.3f}",
                    "Val PR-AUC": f"{best['pr_auc']:.3f}",
                    "Val 95 % CI": _interval(best["pr_auc_ci_low"], best["pr_auc_ci_high"]),
                }
            )
    return pd.DataFrame(rows)


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
    """Every tried hyperparameter setting next to the untuned grid run of the same strategy."""
    tuning = results[results["stage"] == "tuning"]
    grid = results[results["stage"] == "grid"]
    rows = []
    for model in [model for model in MODEL_ORDER if model in set(tuning["model"])]:
        tried = tuning[tuning["model"] == model]
        first = tried.iloc[0]
        ratio = None if pd.isna(first["sampling_ratio"]) else float(first["sampling_ratio"])
        reference = grid_row(grid, model, first["strategy"], ratio)
        candidates = pd.concat([reference.to_frame().T, tried]).sort_values(
            "cv_pr_auc_mean", ascending=False
        )
        for _, row in candidates.iterrows():
            rows.append(
                {
                    "Model": MODEL_LABELS[model],
                    "Strategy": strategy_label(row["strategy"], row["sampling_ratio"]),
                    "Changed parameters": changed_params(row, reference),
                    "CV PR-AUC (mean ± std)": (
                        f"{row['cv_pr_auc_mean']:.3f} ± {row['cv_pr_auc_std']:.3f}"
                    ),
                    "Val PR-AUC": f"{row['pr_auc']:.3f}",
                    "Val 95 % CI": _interval(row["pr_auc_ci_low"], row["pr_auc_ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def write_summary(path: Path, generated: str) -> None:
    """Replace the generated block of the summary, keeping the hand-written conclusions."""
    text = path.read_text(encoding="utf-8") if path.exists() else SUMMARY_TEMPLATE
    if GENERATED_START not in text or GENERATED_END not in text:
        raise ValueError(f"{path} has no generated-block markers.")
    before = text.split(GENERATED_START)[0]
    after = text.split(GENERATED_END)[1]
    path.parent.mkdir(parents=True, exist_ok=True)
    block = f"{GENERATED_START}\n{generated.strip()}\n{GENERATED_END}"
    path.write_text(before + block + after, encoding="utf-8", newline="\n")


def build_report(
    results: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    paths: ExperimentPaths,
    n_boot: int = N_BOOTSTRAP,
) -> list[str]:
    """Create figures, the summary and the saved final model; return progress messages."""
    results = results[(results["eval_split"] == "val") & results["cv_pr_auc_mean"].notna()]
    grid = results[results["stage"] == "grid"]
    _, val_y = split_xy(val)

    ranked_grid = rank_by_rule(grid)
    final = select_best(results)
    top = ranked_grid.head(TOP_N_PAIRED)
    compared = (
        top if final["config_id"] in set(top["config_id"]) else pd.concat([top, final.to_frame().T])
    )
    comparisons = paired_comparisons(compared, grid, val_y, paths.scores_dir, n_boot)

    best_on_val = results.loc[results["pr_auc"].idxmax()]
    confirmation = "The selected configuration is also the best one on validation."
    if best_on_val["config_id"] != final["config_id"]:
        check = paired_bootstrap_difference(
            val_y,
            load_scores(final["config_id"], paths.scores_dir),
            load_scores(best_on_val["config_id"], paths.scores_dir),
            n_boot=n_boot,
        )
        verdict = (
            "**The interval excludes zero: the CV winner is clearly worse on validation. "
            "Flagged for review.**"
            if check["ci_high"] < 0
            else "The interval includes zero, so validation does not contradict the CV choice."
        )
        confirmation = (
            f"Best on validation is {config_label(best_on_val)} "
            f"(PR-AUC {best_on_val['pr_auc']:.3f}, "
            f"CV {best_on_val['cv_pr_auc_mean']:.3f}). Selected minus best-on-validation: "
            f"{check['difference']:+.3f}, "
            f"95 % CI {_interval(check['ci_low'], check['ci_high'])}. {verdict}"
        )

    best_per_model = pd.DataFrame(
        [
            select_best(results[results["model"] == model])
            for model in MODEL_ORDER
            if model in set(results["model"])
        ]
    )
    figures = [
        plot_pr_auc_by_config(grid, paths.figures_dir),
        plot_pr_curves(best_per_model, val_y, paths.scores_dir, paths.figures_dir),
        plot_cv_heatmap(grid, paths.figures_dir),
    ]
    config = save_final_model(final, train, val, paths.models_dir)

    tuning = tuning_table(results)
    size_mb = config["model_file_bytes"] / 1e6
    final_params = changed_params(final, grid_row(grid, final["model"], "none", None))
    final_ci = _interval(final["pr_auc_ci_low"], final["pr_auc_ci_high"])
    sections = [
        "## Selected configuration",
        f"**{config_label(final)}** ({final_params}; stage: {final['stage']}). "
        f"Mean CV PR-AUC {final['cv_pr_auc_mean']:.3f}; validation PR-AUC {final['pr_auc']:.3f}, "
        f"95 % CI {final_ci}.",
        f"Validation check: {confirmation}",
        f"Saved to `models/{FINAL_MODEL_NAME}` ({size_mb:.2f} MB, fitted on the training split) "
        f"with `models/{FINAL_CONFIG_NAME}`. No decision threshold is chosen yet and the test "
        "split was not used.",
        f"## Ranking of the {len(ranked_grid)} grid configurations",
        "Ordered by the selection rule (mean CV PR-AUC, ties within 0.005 broken toward the "
        f"simpler strategy). Validation has {int(val_y.sum())} fraud cases, so its intervals "
        "are wide.",
        markdown_table(ranking_table(ranked_grid)),
        "## Resampling against class weights and no resampling",
        "Best run of each approach per model, chosen by CV.",
        markdown_table(family_table(grid)),
        "## Paired comparisons on validation",
        "Paired stratified bootstrap of the PR-AUC difference (configuration minus reference) "
        f"for the top {TOP_N_PAIRED} grid configurations and the selected one. An interval that "
        "includes zero means the difference cannot be told apart from noise on this "
        "validation set.",
        markdown_table(comparison_table(comparisons)),
        "## Hyperparameter tuning",
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
