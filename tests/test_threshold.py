import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from fraud import threshold as th
from fraud.config import RANDOM_SEED, REVIEW_COST_EUR
from fraud.evaluate import (
    WeightedRocAuc,
    no_model_cost,
    predict_at_threshold,
    threshold_metrics,
    total_cost,
)

# Two fraud cases (100 and 20) and two legitimate rows; a review costs 5.
Y_TRUE = np.array([1, 0, 1, 0])
Y_SCORE = np.array([0.9, 0.8, 0.3, 0.1])
AMOUNT = np.array([100.0, 50.0, 20.0, 10.0])


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imbalanced labels, informative scores with ties, and skewed amounts."""
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = (rng.random(3_000) < 0.03).astype(int)
    y_score = np.clip(rng.normal(0.2 + 0.4 * y_true, 0.2), 0, 1).round(2)
    amount = rng.lognormal(3.0, 1.5, 3_000).round(2)
    return y_true, y_score, amount


def brute_force_cost(y_true: np.ndarray, y_score: np.ndarray, amount: np.ndarray, t: float):
    return total_cost(y_true, predict_at_threshold(y_score, t), amount)


def test_cost_of_every_cut_follows_the_cost_model() -> None:
    table = th.CostCurve(Y_TRUE, Y_SCORE, AMOUNT, review_cost=5.0).table()
    # flag nothing, top 1, top 2, top 3, everything
    assert list(table["cost"]) == [120.0, 25.0, 30.0, 15.0, 20.0]
    assert list(table["flagged"]) == [0, 1, 2, 3, 4]
    assert list(table["caught_fraud"]) == [0, 1, 1, 2, 2]
    assert list(table["missed_fraud"]) == [2, 1, 1, 0, 0]


def test_thresholds_lie_between_neighbouring_scores() -> None:
    thresholds = th.CostCurve(Y_TRUE, Y_SCORE, AMOUNT).thresholds
    assert thresholds[0] == th.FLAG_NOTHING and thresholds[-1] == th.FLAG_EVERYTHING
    np.testing.assert_allclose(thresholds[1:-1], [0.85, 0.55, 0.2])


def test_cost_optimal_threshold_minimises_total_cost() -> None:
    threshold, cost = th.cost_optimal_threshold(Y_TRUE, Y_SCORE, AMOUNT, review_cost=5.0)
    assert (threshold, cost) == (pytest.approx(0.2), 15.0)
    assert brute_force_cost(Y_TRUE, Y_SCORE, AMOUNT, threshold) == cost


def test_curve_matches_a_brute_force_search(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    curve = th.CostCurve(y_true, y_score, amount)
    table = curve.table()
    for threshold, cost in zip(table["threshold"], table["cost"], strict=True):
        assert brute_force_cost(y_true, y_score, amount, threshold) == pytest.approx(cost)
    candidates = np.r_[np.unique(y_score), 2.0]
    cheapest = min(brute_force_cost(y_true, y_score, amount, t) for t in candidates)
    assert curve.best()[1] == pytest.approx(cheapest)
    assert table["cost"].iloc[0] == pytest.approx(no_model_cost(y_true, amount))
    assert table["cost"].iloc[-1] == th.flag_everything_cost(len(y_true))


def test_equal_scores_are_one_cut() -> None:
    curve = th.CostCurve([1, 0, 0, 1], [0.7, 0.7, 0.2, 0.2], [10, 10, 10, 10])
    assert len(curve.thresholds) == 3  # nothing, the 0.7 rows, everything


def test_fraud_cheaper_than_a_review_is_not_worth_flagging() -> None:
    # Missing a 1-euro fraud costs less than the 5-euro review that would catch it.
    threshold, cost = th.cost_optimal_threshold([1, 0, 0], [0.9, 0.5, 0.1], [1.0, 80.0, 80.0])
    assert threshold == th.FLAG_NOTHING and cost == 1.0
    assert predict_at_threshold(np.array([0.9, 1.0]), threshold).sum() == 0


def test_a_cost_tie_goes_to_the_threshold_that_flags_fewer_rows() -> None:
    # Flagging the single fraud costs one review (5), exactly its amount.
    threshold, cost = th.cost_optimal_threshold([1, 0], [0.9, 0.1], [REVIEW_COST_EUR, 30.0])
    assert threshold == th.FLAG_NOTHING and cost == REVIEW_COST_EUR


def test_a_higher_review_cost_never_lowers_the_threshold(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    thresholds = [
        th.cost_optimal_threshold(y_true, y_score, amount, review_cost=cost)[0]
        for cost in (0.5, 5.0, 50.0)
    ]
    assert thresholds == sorted(thresholds)


def test_inputs_must_have_the_same_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        th.CostCurve([1, 0], [0.5], [1.0, 2.0])


def test_f1_optimal_threshold_matches_a_brute_force_search(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, _ = scored
    threshold, f1 = th.f1_optimal_threshold(y_true, y_score)
    best = max(f1_score(y_true, y_score >= t, zero_division=0) for t in np.unique(y_score))
    assert f1 == pytest.approx(best)
    assert f1_score(y_true, predict_at_threshold(y_score, threshold)) == pytest.approx(f1)


def test_row_costs_sum_to_the_total_cost(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    rows = th.row_costs(y_true, y_score, amount, 0.4)
    assert rows.sum() == pytest.approx(brute_force_cost(y_true, y_score, amount, 0.4))


def test_evaluate_decisions_agrees_with_the_basic_metrics(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    result = th.evaluate_decisions(y_true, y_score, amount, 0.4, n_boot=50)
    basic = threshold_metrics(y_true, y_score, 0.4)
    for name in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
        assert result[name]["value"] == pytest.approx(basic[name])
    cost = brute_force_cost(y_true, y_score, amount, 0.4)
    assert result["cost"]["value"] == pytest.approx(cost)
    assert result["pr_auc"]["value"] == pytest.approx(average_precision_score(y_true, y_score))
    assert result["roc_auc"]["value"] == pytest.approx(roc_auc_score(y_true, y_score))
    no_model = no_model_cost(y_true, amount)
    assert result["savings_vs_no_model"]["value"] == pytest.approx(no_model - cost)
    assert result["savings_vs_flag_everything"]["value"] == pytest.approx(
        REVIEW_COST_EUR * len(y_true) - cost
    )
    assert result["savings_share_of_no_model"]["value"] == pytest.approx(1 - cost / no_model)
    for metric in result.values():
        assert metric["ci_low"] <= metric["value"] <= metric["ci_high"]


def test_evaluate_decisions_is_reproducible(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    first = th.evaluate_decisions(*scored, 0.4, n_boot=30)
    assert first == th.evaluate_decisions(*scored, 0.4, n_boot=30)
    assert first != th.evaluate_decisions(*scored, 0.4, n_boot=30, seed=RANDOM_SEED + 1)


def test_weighted_roc_auc_matches_sklearn_with_ties(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, _ = scored
    counts = np.random.default_rng(1).integers(0, 4, len(y_true)).astype(float)
    expected = roc_auc_score(y_true, y_score, sample_weight=counts)
    assert WeightedRocAuc(y_true, y_score)(counts) == pytest.approx(expected)


def test_paired_cost_difference_of_identical_scores_is_zero(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    for thresholds in ((None, None), (0.4, 0.4)):
        result = th.paired_cost_difference(y_true, amount, y_score, y_score, *thresholds, n_boot=30)
        assert result["difference"] == result["ci_low"] == result["ci_high"] == 0.0


def test_paired_cost_difference_detects_a_clearly_cheaper_model(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    noise = np.random.default_rng(3).random(len(y_true))
    result = th.paired_cost_difference(y_true, amount, y_score, noise, n_boot=100)
    assert result["difference"] < 0 and result["ci_high"] < 0
    assert result["share_a_cheaper"] == 1.0
    assert result["a"] == pytest.approx(th.cost_optimal_threshold(y_true, y_score, amount)[1])


def test_paired_cost_difference_at_fixed_thresholds(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    y_true, y_score, amount = scored
    result = th.paired_cost_difference(y_true, amount, y_score, y_score, 0.4, 0.9, n_boot=30)
    assert result["a"] == pytest.approx(brute_force_cost(y_true, y_score, amount, 0.4))
    assert result["b"] == pytest.approx(brute_force_cost(y_true, y_score, amount, 0.9))
    with pytest.raises(ValueError, match="both thresholds or neither"):
        th.paired_cost_difference(y_true, amount, y_score, y_score, 0.4, None)


def test_bootstrap_optimal_thresholds_are_reproducible_and_valid(
    scored: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    first = th.bootstrap_optimal_thresholds(*scored, n_boot=40)
    np.testing.assert_array_equal(first, th.bootstrap_optimal_thresholds(*scored, n_boot=40))
    assert len(first) == 40
    assert set(first) <= set(th.CostCurve(*scored).thresholds)
