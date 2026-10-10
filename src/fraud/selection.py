"""Fixed ordering of the staged experiments and the selection rule (PLAN.md sections 6, 14)."""

import pandas as pd

from fraud.config import CV_TIE_TOLERANCE

MODEL_ORDER: list[str] = [
    "logreg",
    "decision_tree",
    "gaussian_nb",
    "knn",
    "random_forest",
    "extra_trees",
    "xgboost",
    "lightgbm",
    "mlp",
]
MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic regression",
    "decision_tree": "Decision tree",
    "gaussian_nb": "Gaussian NB",
    "knn": "k-NN",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "mlp": "MLP",
}

Strategy = tuple[str, float | None]

# Stage 1: every classifier with these strategies (class weights only where supported).
STAGE1_STRATEGIES: list[Strategy] = [("none", None), ("class_weight", None), ("smote", 0.1)]
# Stage 2: the SMOTE family, for the best classifiers of stage 1.
SMOTE_VARIANTS: list[str] = ["smote_tomek", "smote_enn", "safe_level_smote", "polynom_fit_smote"]
VARIANT_RATIOS: tuple[float, float] = (0.1, 0.5)
STAGE2_STRATEGIES: list[Strategy] = [
    ("smote", 0.5),
    ("smote", 1.0),
    *[(variant, ratio) for variant in SMOTE_VARIANTS for ratio in VARIANT_RATIOS],
]
OVERSAMPLING: list[str] = ["smote", *SMOTE_VARIANTS]
# Classifiers that also get the stage 2 runs when they are not in the top group, so that the
# thesis question about linear and neighbour-based classifiers can be answered. Their rows
# are labelled EXTRA_STAGE and never take part in ranking or model selection.
EXTRA_STAGE2_MODELS: list[str] = ["logreg", "knn"]
EXTRA_STAGE: str = "stage2_extra"
# Stages whose rows compete in the ranking and in the selection of the final model.
SELECTION_STAGES: list[str] = ["stage1", "stage2", "stage3"]
# Display order of all strategies in tables and figures.
STRATEGY_ORDER: list[Strategy] = [
    ("none", None),
    ("class_weight", None),
    ("smote", 0.1),
    *STAGE2_STRATEGIES,
]
STRATEGY_LABELS: dict[str, str] = {
    "none": "No resampling",
    "class_weight": "Class weights",
    "smote": "SMOTE",
    "smote_tomek": "SMOTE-Tomek",
    "smote_enn": "SMOTE-ENN",
    "safe_level_smote": "Safe-Level-SMOTE",
    "polynom_fit_smote": "Polynom-Fit SMOTE",
}


def strategy_label(strategy: str, sampling_ratio: float | None) -> str:
    """Readable name of a strategy, e.g. ``SMOTE 0.1``."""
    label = STRATEGY_LABELS[strategy]
    if sampling_ratio is None or pd.isna(sampling_ratio):
        return label
    return f"{label} {sampling_ratio:g}"


def config_label(row: pd.Series) -> str:
    """Readable name of a configuration, e.g. ``LightGBM, SMOTE 0.1``."""
    return f"{MODEL_LABELS[row['model']]}, {strategy_label(row['strategy'], row['sampling_ratio'])}"


def simplicity(strategy: str) -> int:
    """Lower is simpler: no resampling, then class weights, then any resampling."""
    return {"none": 0, "class_weight": 1}.get(strategy, 2)


def select_best(results: pd.DataFrame, tolerance: float = CV_TIE_TOLERANCE) -> pd.Series:
    """Apply the selection rule: best mean CV PR-AUC, ties broken toward the simplest.

    Configurations within ``tolerance`` of the best mean count as tied. Among them the
    simplest strategy wins, then the higher mean.
    """
    candidates = results.dropna(subset=["cv_pr_auc_mean"])
    if candidates.empty:
        raise ValueError("No configuration has a cross-validation result.")
    best = candidates["cv_pr_auc_mean"].max()
    tied = candidates[candidates["cv_pr_auc_mean"] >= best - tolerance]
    order = tied.assign(_simplicity=tied["strategy"].map(simplicity)).sort_values(
        ["_simplicity", "cv_pr_auc_mean"], ascending=[True, False], kind="stable"
    )
    return order.iloc[0].drop("_simplicity")


def rank_by_rule(results: pd.DataFrame, tolerance: float = CV_TIE_TOLERANCE) -> pd.DataFrame:
    """Order configurations by applying :func:`select_best` repeatedly."""
    remaining = results.dropna(subset=["cv_pr_auc_mean"])
    picked = []
    while not remaining.empty:
        best = select_best(remaining, tolerance)
        picked.append(best)
        remaining = remaining[remaining["config_id"] != best["config_id"]]
    ranked = pd.DataFrame(picked, columns=results.columns).reset_index(drop=True)
    ranked.index = ranked.index + 1
    return ranked.rename_axis("rank")


def rank_classifiers(results: pd.DataFrame) -> pd.DataFrame:
    """One row per classifier: its configuration with the best mean CV PR-AUC, best first.

    This is the stage 1 ranking. It uses the plain maximum (no tie-break): the question is
    how good a classifier can be, not which of its strategies to prefer.
    """
    candidates = results.dropna(subset=["cv_pr_auc_mean"])
    best = candidates.loc[candidates.groupby("model")["cv_pr_auc_mean"].idxmax()]
    ranked = best.sort_values("cv_pr_auc_mean", ascending=False, kind="stable")
    ranked = ranked.reset_index(drop=True)
    ranked.index = ranked.index + 1
    return ranked.rename_axis("rank")
