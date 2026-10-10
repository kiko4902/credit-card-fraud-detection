"""Metrics, bootstrap confidence intervals and the business cost function.

Threshold selection from cost is Phase 4; here the cost of a given set of decisions can
already be computed.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
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


def stratified_resamples(y_true: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    """Yield ``n_boot`` index arrays, resampling each class separately with replacement."""
    positives = np.flatnonzero(y_true == 1)
    negatives = np.flatnonzero(y_true == 0)
    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("Both classes must be present to bootstrap.")
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate(
            [
                rng.choice(positives, size=len(positives), replace=True),
                rng.choice(negatives, size=len(negatives), replace=True),
            ]
        )


def _percentile_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


class WeightedAveragePrecision:
    """Average precision of fixed scores under varying integer row weights.

    A bootstrap resample is the original rows with a count for each row. Sorting the scores
    once and reweighting per resample gives the same value as recomputing average precision
    on the resampled arrays, about twenty times faster.
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self._order = np.argsort(-y_score, kind="stable")
        sorted_scores = y_score[self._order]
        self._positive = (y_true[self._order] == 1).astype(float)
        # Last position of every run of equal scores: one precision/recall point per threshold.
        self._group_end = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]

    def __call__(self, counts: np.ndarray) -> float:
        weights = counts[self._order].astype(float)
        true_positives = np.cumsum(weights * self._positive)[self._group_end]
        flagged = np.cumsum(weights)[self._group_end]
        precision = np.divide(true_positives, flagged, out=np.ones_like(flagged), where=flagged > 0)
        recall = true_positives / true_positives[-1]
        return float(np.sum(np.diff(recall, prepend=0.0) * precision))


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
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for index in stratified_resamples(y_true, n_boot, seed):
        for name, metric in metrics.items():
            samples[name].append(metric(y_true[index], y_score[index]))
    return {name: _percentile_interval(values, confidence) for name, values in samples.items()}


def paired_bootstrap_difference(
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """PR-AUC difference ``a - b`` with a paired bootstrap confidence interval.

    Both score vectors are evaluated on the same stratified resamples, so the interval
    reflects how much the two models differ on the same transactions. This is much tighter
    than comparing two separate intervals, which ignores that both models find the same
    cases easy or hard. ``share_a_better`` is the share of resamples in which ``a`` wins.
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)
    precision_a = WeightedAveragePrecision(y_true, score_a)
    precision_b = WeightedAveragePrecision(y_true, score_b)
    differences = []
    for index in stratified_resamples(y_true, n_boot, seed):
        counts = np.bincount(index, minlength=len(y_true))
        differences.append(precision_a(counts) - precision_b(counts))
    low, high = _percentile_interval(differences, confidence)
    value_a = float(average_precision_score(y_true, score_a))
    value_b = float(average_precision_score(y_true, score_b))
    return {
        "a": value_a,
        "b": value_b,
        "difference": value_a - value_b,
        "ci_low": low,
        "ci_high": high,
        "share_a_better": float(np.mean(np.asarray(differences) > 0)),
    }


def evaluate_scores(
    y_true: ArrayLike,
    y_score: ArrayLike,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """All reported metrics for one set of scores, with CIs for PR-AUC and recall."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    average_precision = WeightedAveragePrecision(y_true, y_score)
    caught = ((y_true == 1) & (predict_at_threshold(y_score, threshold) == 1)).astype(float)
    fraud = (y_true == 1).astype(float)
    samples: dict[str, list[float]] = {"pr_auc": [], "recall": []}
    for index in stratified_resamples(y_true, n_boot, seed):
        counts = np.bincount(index, minlength=len(y_true))
        samples["pr_auc"].append(average_precision(counts))
        samples["recall"].append(float(counts @ caught / (counts @ fraud)))
    intervals = {
        name: _percentile_interval(values, CONFIDENCE_LEVEL) for name, values in samples.items()
    }
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
