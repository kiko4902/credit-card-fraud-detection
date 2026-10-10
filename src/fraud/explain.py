"""SHAP explanations for the final tree model (PLAN.md section 9).

The classifier is the last step of the pipeline and sees transformed, standardised
features. SHAP values are computed for exactly that input, but shown with the readable
feature names and the unscaled feature values, so a plot says "Amount (log) = 4.6" and not
"feature 28 = 0.8".

SHAP values of a boosted tree model are in log-odds: a positive value pushes the fraud
score up, a negative one down, and base value plus all values of a row is the raw model
output of that row.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline

from fraud.config import AMOUNT_COLUMN, PCA_COLUMNS, RANDOM_SEED, TARGET_COLUMN, TIME_COLUMN
from fraud.evaluate import predict_at_threshold
from fraud.features import FEATURE_LABELS, MODEL_FEATURES, SECONDS_PER_DAY
from fraud.plotting import SAVE_DPI
from fraud.serving import model_inputs

CASE_LABELS: dict[str, str] = {
    "caught_fraud": "Caught fraud",
    "missed_fraud": "Missed fraud",
    "false_alarm": "False alarm",
}
SHAP_PREFIX: str = "shap_"


def readable_features(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Model features before scaling, for display next to the SHAP values."""
    return pipeline.steps[0][1].transform(features)[MODEL_FEATURES]


def explain(pipeline: Pipeline, features: pd.DataFrame) -> shap.Explanation:
    """SHAP values (log-odds) of the tree model for ``features``, with readable names."""
    inputs = model_inputs(pipeline, features)
    explanation = shap.TreeExplainer(pipeline.steps[-1][1])(inputs)
    return shap.Explanation(
        values=explanation.values,
        base_values=explanation.base_values,
        data=readable_features(pipeline, features).to_numpy(),
        feature_names=[FEATURE_LABELS[name] for name in inputs.columns],
    )


def additivity_error(
    pipeline: Pipeline, features: pd.DataFrame, explanation: shap.Explanation
) -> float:
    """Largest gap between base value + SHAP values and the model's raw output."""
    raw = pipeline.steps[-1][1].predict(model_inputs(pipeline, features), output_margin=True)
    return float(np.abs(explanation.values.sum(axis=1) + explanation.base_values - raw).max())


def mean_absolute_shap(explanation: shap.Explanation) -> pd.Series:
    """Global importance: mean absolute SHAP value per feature, largest first."""
    importance = pd.Series(np.abs(explanation.values).mean(axis=0), explanation.feature_names)
    return importance.sort_values(ascending=False).rename("mean_abs_shap")


def pick_cases(
    y_true: pd.Series, scores: np.ndarray, amount: pd.Series, threshold: float
) -> dict[str, int]:
    """Positions of one example per outcome, chosen by a fixed rule.

    Caught and missed fraud: the one with the largest amount (the most money at stake).
    False alarm: the legitimate transaction with the highest score. An outcome that does not
    occur is left out.
    """
    fraud = y_true.to_numpy() == 1
    flagged = predict_at_threshold(scores, threshold) == 1
    amounts = amount.to_numpy(dtype=float)
    groups = {
        "caught_fraud": (fraud & flagged, amounts),
        "missed_fraud": (fraud & ~flagged, amounts),
        "false_alarm": (~fraud & flagged, np.asarray(scores, dtype=float)),
    }
    cases = {}
    for name, (mask, order_by) in groups.items():
        if mask.any():
            cases[name] = int(np.flatnonzero(mask)[np.argmax(order_by[mask])])
    return cases


def plot_global(
    explanation: shap.Explanation, path: Path, split_name: str = "test", max_display: int = 15
) -> Path:
    """Beeswarm summary: one dot per transaction and feature, coloured by feature value."""
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False, plot_size=(9, 6.5))
    figure = plt.gcf()
    figure.axes[0].set_title(
        f"What drives the fraud score: SHAP values (log-odds), {split_name} transactions",
        loc="left",
        fontweight="bold",
        fontsize=11,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_local(explanation_row: shap.Explanation, title: str, path: Path) -> Path:
    """Waterfall of one transaction: how each feature moves its score from the base value."""
    plt.figure()
    shap.plots.waterfall(explanation_row, max_display=10, show=False)
    figure = plt.gcf()
    figure.set_size_inches(9, 5.5)
    figure.axes[0].set_title(title, loc="left", fontweight="bold", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def outcome(is_fraud: bool, is_flagged: bool) -> str:
    if is_fraud:
        return "caught_fraud" if is_flagged else "missed_fraud"
    return "false_alarm" if is_flagged else "legitimate"


def explained_sample(
    rows: pd.DataFrame,
    scores: np.ndarray,
    explanation: shap.Explanation,
    threshold: float,
    n_legitimate: int = 100,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """A small table of transactions with score, decision and SHAP values, for the app.

    Holds every fraud, every false alarm and a random sample of correctly passed legitimate
    rows. The rows carry no identifier and no position in the recording: the index is
    dropped and ``Time`` is replaced by the second of the day, which gives the pipeline the
    same time-of-day features. The remaining columns are the dataset's PCA components.
    """
    fraud = rows[TARGET_COLUMN].to_numpy() == 1
    flagged = predict_at_threshold(scores, threshold) == 1
    quiet = np.flatnonzero(~fraud & ~flagged)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(quiet, size=min(n_legitimate, len(quiet)), replace=False)
    keep = np.sort(np.r_[np.flatnonzero(fraud | flagged), sampled])

    sample = rows.iloc[keep][[TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN, TARGET_COLUMN]].copy()
    sample[TIME_COLUMN] = sample[TIME_COLUMN] % SECONDS_PER_DAY
    sample["score"] = np.asarray(scores)[keep]
    sample["flagged"] = flagged[keep].astype(int)
    sample["outcome"] = [outcome(f, g) for f, g in zip(fraud[keep], flagged[keep], strict=True)]
    sample["base_value"] = np.broadcast_to(explanation.base_values, (len(rows),))[keep]
    shap_values = pd.DataFrame(
        explanation.values[keep],
        columns=[f"{SHAP_PREFIX}{name}" for name in MODEL_FEATURES],
        index=sample.index,
    )
    # Shuffled, so the row order does not give away the order in time either.
    table = pd.concat([sample, shap_values], axis=1).sample(frac=1, random_state=seed)
    return table.reset_index(drop=True)
