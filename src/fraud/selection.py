"""Fixed ordering of the experiment grid and the model selection rule (PLAN.md section 14)."""

import pandas as pd

from fraud.config import CV_TIE_TOLERANCE

MODEL_ORDER: list[str] = ["logreg", "random_forest", "lightgbm"]
MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}

GRID_RATIOS: tuple[float, float] = (0.1, 0.5)
RESAMPLING_STRATEGIES: list[str] = [
    "undersample",
    "smote",
    "borderline_smote",
    "adasyn",
    "smote_tomek",
]
SMOTE_FAMILY: list[str] = ["smote", "borderline_smote", "adasyn", "smote_tomek"]
# The 12 strategies of the grid, in the order used in every table and figure.
STRATEGY_ORDER: list[tuple[str, float | None]] = [
    ("none", None),
    ("class_weight", None),
    *[(strategy, ratio) for strategy in RESAMPLING_STRATEGIES for ratio in GRID_RATIOS],
]
STRATEGY_LABELS: dict[str, str] = {
    "none": "No resampling",
    "class_weight": "Class weights",
    "undersample": "Undersampling",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "smote_tomek": "SMOTE-Tomek",
}


def strategy_label(strategy: str, sampling_ratio: float | None) -> str:
    """Readable name of a strategy, e.g. ``SMOTE 0.1``."""
    label = STRATEGY_LABELS[strategy]
    return (
        label
        if sampling_ratio is None or pd.isna(sampling_ratio)
        else f"{label} {sampling_ratio:g}"
    )


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
