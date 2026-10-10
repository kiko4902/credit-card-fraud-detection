"""Decision thresholds from the business cost model (PLAN.md section 8).

A transaction is flagged when its fraud score reaches the threshold. Under the cost model a
missed fraud costs its amount and every flagged transaction costs one review, so the total
cost of a threshold is::

    sum of the amounts of fraud scored below it + review cost * number of rows at or above it

Thresholds are chosen on validation only. The cost of every possible threshold is computed
in one pass over the rows sorted by score, which also makes bootstrap resampling cheap.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from fraud.config import CONFIDENCE_LEVEL, N_BOOTSTRAP, RANDOM_SEED, REVIEW_COST_EUR
from fraud.evaluate import (
    WeightedAveragePrecision,
    WeightedRocAuc,
    predict_at_threshold,
    stratified_resamples,
)

# Threshold that flags nothing: no score reaches it.
FLAG_NOTHING: float = float("inf")
# Threshold that flags everything: every score reaches it.
FLAG_EVERYTHING: float = 0.0


def flag_everything_cost(n_rows: int, review_cost: float = REVIEW_COST_EUR) -> float:
    """Cost of reviewing every transaction: no fraud is missed."""
    return float(review_cost * n_rows)


class CostCurve:
    """Total cost of every distinct threshold for one set of scores.

    Cut ``0`` flags nothing; cut ``k`` flags the rows with the ``k`` highest distinct
    scores; the last cut flags everything. The threshold of a cut lies halfway between the
    lowest flagged score and the highest score that is not flagged, so that it does not
    depend on the last digits of a single validation score.
    """

    def __init__(
        self,
        y_true: ArrayLike,
        y_score: ArrayLike,
        amount: ArrayLike,
        review_cost: float = REVIEW_COST_EUR,
    ) -> None:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)
        amount = np.asarray(amount, dtype=float)
        if not (len(y_true) == len(y_score) == len(amount)):
            raise ValueError("y_true, y_score and amount must have the same length.")
        if len(y_true) == 0:
            raise ValueError("At least one row is needed.")
        self.review_cost = review_cost
        self._order = np.argsort(-y_score, kind="stable")
        sorted_scores = y_score[self._order]
        self._fraud = (y_true[self._order] == 1).astype(float)
        self._fraud_amount = amount[self._order] * self._fraud
        # Last position of every run of equal scores: one cut per distinct score.
        self._group_end = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
        distinct = sorted_scores[self._group_end]
        midpoints = (distinct[:-1] + distinct[1:]) / 2
        self.thresholds: np.ndarray = np.r_[FLAG_NOTHING, midpoints, FLAG_EVERYTHING]

    def _weights(self, counts: np.ndarray | None) -> np.ndarray:
        if counts is None:
            return np.ones(len(self._order))
        return np.asarray(counts, dtype=float)[self._order]

    def costs(self, counts: np.ndarray | None = None) -> np.ndarray:
        """Total cost of every cut; ``counts`` weights the rows (a bootstrap resample)."""
        weights = self._weights(counts)
        fraud_amount = weights * self._fraud_amount
        caught_amount = np.r_[0.0, np.cumsum(fraud_amount)[self._group_end]]
        flagged = np.r_[0.0, np.cumsum(weights)[self._group_end]]
        return fraud_amount.sum() - caught_amount + self.review_cost * flagged

    def table(self) -> pd.DataFrame:
        """One row per cut: threshold, flagged rows, caught and missed fraud, cost."""
        flagged = np.r_[0, self._group_end + 1]
        caught = np.r_[0.0, np.cumsum(self._fraud)[self._group_end]]
        return pd.DataFrame(
            {
                "threshold": self.thresholds,
                "flagged": flagged,
                "caught_fraud": caught.astype(int),
                "missed_fraud": (self._fraud.sum() - caught).astype(int),
                "cost": self.costs(),
            }
        )

    def best(self, counts: np.ndarray | None = None) -> tuple[float, float]:
        """``(threshold, cost)`` of the cheapest cut. Ties go to the cut that flags fewer rows."""
        costs = self.costs(counts)
        cut = int(np.argmin(costs))
        return float(self.thresholds[cut]), float(costs[cut])


def cost_optimal_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    review_cost: float = REVIEW_COST_EUR,
) -> tuple[float, float]:
    """The threshold with the lowest total cost, and that cost."""
    return CostCurve(y_true, y_score, amount, review_cost).best()


def f1_optimal_threshold(y_true: ArrayLike, y_score: ArrayLike) -> tuple[float, float]:
    """The threshold with the highest F1, and that F1 (for comparison with the cost optimum)."""
    y_true = np.asarray(y_true)
    # With unit amounts and no review cost, the curve's bookkeeping gives the counts per cut.
    table = CostCurve(y_true, y_score, np.ones(len(y_true)), review_cost=0.0).table()
    true_positives = table["caught_fraud"].to_numpy(dtype=float)
    denominator = table["flagged"].to_numpy(dtype=float) + float((y_true == 1).sum())
    f1 = np.divide(
        2 * true_positives, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    cut = int(np.argmax(f1))
    return float(table["threshold"].iloc[cut]), float(f1[cut])


def row_costs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    threshold: float,
    review_cost: float = REVIEW_COST_EUR,
) -> np.ndarray:
    """Cost contributed by each row at ``threshold``; the sum is the total cost."""
    y_true = np.asarray(y_true)
    flagged = predict_at_threshold(y_score, threshold) == 1
    missed = (y_true == 1) & ~flagged
    return np.asarray(amount, dtype=float) * missed + review_cost * flagged


def _interval(values: list[float], confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


@dataclass(frozen=True)
class Decisions:
    """Per-row ingredients of every reported metric at one threshold."""

    y_true: np.ndarray
    y_score: np.ndarray
    amount: np.ndarray
    threshold: float
    review_cost: float

    def metrics(self, counts: np.ndarray | None = None) -> dict[str, float]:
        """All metrics, with every row weighted by ``counts`` (default: once each)."""
        weights = np.ones(len(self.y_true)) if counts is None else counts.astype(float)
        fraud = self.y_true == 1
        flagged = predict_at_threshold(self.y_score, self.threshold) == 1
        tp = float(weights @ (fraud & flagged))
        fp = float(weights @ (~fraud & flagged))
        fn = float(weights @ (fraud & ~flagged))
        tn = float(weights @ (~fraud & ~flagged))
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        cost = float(weights @ (self.amount * (fraud & ~flagged))) + self.review_cost * (tp + fp)
        no_model = float(weights @ (self.amount * fraud))
        everything = self.review_cost * float(weights.sum())
        return {
            "precision": precision,
            "recall": recall,
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "cost": cost,
            "cost_no_model": no_model,
            "cost_flag_everything": everything,
            "savings_vs_no_model": no_model - cost,
            "savings_vs_flag_everything": everything - cost,
            "savings_share_of_no_model": _ratio(no_model - cost, no_model),
        }


def evaluate_decisions(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    threshold: float,
    review_cost: float = REVIEW_COST_EUR,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict[str, dict[str, float]]:
    """Ranking, decision and cost metrics at ``threshold`` with stratified bootstrap CIs.

    Returns ``{metric: {"value": ..., "ci_low": ..., "ci_high": ...}}`` for PR-AUC,
    ROC-AUC and everything in :meth:`Decisions.metrics`.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    decisions = Decisions(y_true, y_score, np.asarray(amount, dtype=float), threshold, review_cost)
    average_precision = WeightedAveragePrecision(y_true, y_score)
    roc_auc = WeightedRocAuc(y_true, y_score)
    ones = np.ones(len(y_true))

    def everything(counts: np.ndarray) -> dict[str, float]:
        return {
            "pr_auc": average_precision(counts),
            "roc_auc": roc_auc(counts),
            **decisions.metrics(counts),
        }

    point = everything(ones)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for index in stratified_resamples(y_true, n_boot, seed):
        resampled = everything(np.bincount(index, minlength=len(y_true)).astype(float))
        for name, value in resampled.items():
            samples[name].append(value)
    result = {}
    for name, value in point.items():
        low, high = _interval(samples[name], confidence)
        result[name] = {"value": value, "ci_low": low, "ci_high": high}
    return result


def paired_cost_difference(
    y_true: ArrayLike,
    amount: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    review_cost: float = REVIEW_COST_EUR,
    n_boot: int = N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    """Cost difference ``a - b`` on the same rows, with a paired bootstrap interval.

    With both thresholds given, each model is costed at its fixed threshold. With both
    ``None``, each model gets its own cost-optimal threshold, and the threshold is chosen
    again inside every resample: the interval then also covers the uncertainty of picking a
    threshold from this few fraud cases. A negative difference means ``a`` is cheaper.
    """
    if (threshold_a is None) != (threshold_b is None):
        raise ValueError("Give both thresholds or neither.")
    y_true = np.asarray(y_true)
    if threshold_a is None:
        curve_a = CostCurve(y_true, score_a, amount, review_cost)
        curve_b = CostCurve(y_true, score_b, amount, review_cost)

        def costs(counts: np.ndarray | None) -> tuple[float, float]:
            return curve_a.best(counts)[1], curve_b.best(counts)[1]
    else:
        rows_a = row_costs(y_true, score_a, amount, threshold_a, review_cost)
        rows_b = row_costs(y_true, score_b, amount, threshold_b, review_cost)

        def costs(counts: np.ndarray | None) -> tuple[float, float]:
            weights = np.ones(len(y_true)) if counts is None else counts
            return float(weights @ rows_a), float(weights @ rows_b)

    cost_a, cost_b = costs(None)
    differences = []
    for index in stratified_resamples(y_true, n_boot, seed):
        resampled_a, resampled_b = costs(np.bincount(index, minlength=len(y_true)).astype(float))
        differences.append(resampled_a - resampled_b)
    low, high = _interval(differences, confidence)
    return {
        "a": cost_a,
        "b": cost_b,
        "difference": cost_a - cost_b,
        "ci_low": low,
        "ci_high": high,
        "share_a_cheaper": float(np.mean(np.asarray(differences) < 0)),
    }


def bootstrap_optimal_thresholds(
    y_true: ArrayLike,
    y_score: ArrayLike,
    amount: ArrayLike,
    review_cost: float = REVIEW_COST_EUR,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """The cost-optimal threshold of every bootstrap resample: how stable is the choice?"""
    y_true = np.asarray(y_true)
    curve = CostCurve(y_true, y_score, amount, review_cost)
    return np.array(
        [
            curve.best(np.bincount(index, minlength=len(y_true)).astype(float))[0]
            for index in stratified_resamples(y_true, n_boot, seed)
        ]
    )
