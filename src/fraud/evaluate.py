"""Metrics, bootstrap confidence intervals and the business cost function.

Threshold selection from cost is Phase 4; here the cost of a given set of decisions can
already be computed.
"""

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from fraud.config import (
    CONFIDENCE_LEVEL,
    DEFAULT_THRESHOLD,
    N_BOOTSTRAP,
    RANDOM_SEED,
    REVIEW_COST_EUR,
)

Metric = Callable[[np.ndarray, np.ndarray], float]


def predict_at_threshold(y_score: ArrayLike, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Flag a transaction as fraud when its score reaches the threshold."""
    return (np.asarray(y_score) >= threshold).astype(int)


def ranking_metrics(y_true: ArrayLike, y_score: ArrayLike) -> dict[str, float]:
    """Threshold-free metrics. PR-AUC (average precision) is the primary metric."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def threshold_metrics(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float = DEFAULT_THRESHOLD
) -> dict[str, float]:
    """Precision, recall, F1 and the confusion matrix counts at ``threshold``."""
    y_pred = predict_at_threshold(y_score, threshold)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def bootstrap_cis(
    y_true: ArrayLike,
    y_score: ArrayLike,
    metrics: Mapping[str, Metric],
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict[str, tuple[float, float]]:
    """Percentile confidence intervals from a stratified bootstrap.

    Fraud and legitimate rows are resampled separately, with replacement, so every resample
    has the same number of fraud cases as the original data. A plain bootstrap would let
    that number vary, which with about 50 fraud cases adds noise that has nothing to do with
    the model. All metrics are computed on the same resamples.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Both classes must be present to bootstrap.")

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot):
        index = np.concatenate(
            [
                rng.choice(positives, size=len(positives), replace=True),
                rng.choice(negatives, size=len(negatives), replace=True),
            ]
        )
        for name, metric in metrics.items():
            samples[name].append(metric(y_true[index], y_score[index]))

    tail = (1 - confidence) / 2
    return {
        name: tuple(float(bound) for bound in np.quantile(values, [tail, 1 - tail]))
        for name, values in samples.items()
    }


def evaluate_scores(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """All reported metrics for one set of scores, with CIs for PR-AUC and recall."""

    def recall_at_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
        return float(recall_score(labels, predict_at_threshold(scores, threshold)))

    intervals = bootstrap_cis(
        y_true,
        y_score,
        {"pr_auc": average_precision_score, "recall": recall_at_threshold},
        n_boot=n_boot,
        seed=seed,
    )
    result = {**ranking_metrics(y_true, y_score), **threshold_metrics(y_true, y_score, threshold)}
    for name, (low, high) in intervals.items():
        result[f"{name}_ci_low"] = low
        result[f"{name}_ci_high"] = high
    return result


def total_cost(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    amount: ArrayLike,
    review_cost: float = REVIEW_COST_EUR,
) -> float:
    """Business cost of a set of decisions (PLAN.md section 8).

    A missed fraud costs its transaction amount. Every flagged transaction, whether it turns
    out to be fraud or not, costs one manual review. Passing a legitimate transaction is free.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    amount = np.asarray(amount, dtype=float)
    missed_fraud = (y_true == 1) & (y_pred == 0)
    flagged = y_pred == 1
    return float(amount[missed_fraud].sum() + review_cost * flagged.sum())


def no_model_cost(y_true: ArrayLike, amount: ArrayLike) -> float:
    """Cost of flagging nothing: every fraud is missed and costs its amount."""
    return total_cost(y_true, np.zeros(len(np.asarray(y_true)), dtype=int), amount)
