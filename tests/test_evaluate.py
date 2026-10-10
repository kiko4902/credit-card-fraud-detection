import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraud import evaluate
from fraud.config import RANDOM_SEED, REVIEW_COST_EUR


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    """Imbalanced labels with informative but imperfect scores."""
    rng = np.random.default_rng(RANDOM_SEED)
    y_true = (rng.random(3_000) < 0.03).astype(int)
    y_score = np.clip(rng.normal(0.2 + 0.4 * y_true, 0.2), 0, 1)
    return y_true, y_score


def test_ranking_metrics_perfect_and_random() -> None:
    y_true = np.array([0, 0, 0, 1, 1])
    perfect = evaluate.ranking_metrics(y_true, np.array([0.1, 0.2, 0.3, 0.8, 0.9]))
    assert perfect == {"pr_auc": 1.0, "roc_auc": 1.0}
    reversed_scores = evaluate.ranking_metrics(y_true, np.array([0.9, 0.8, 0.7, 0.2, 0.1]))
    assert reversed_scores["roc_auc"] == 0.0
    assert reversed_scores["pr_auc"] < 0.5


def test_threshold_metrics_counts() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.5, 0.8, 0.9])
    metrics = evaluate.threshold_metrics(y_true, y_score, threshold=0.5)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (2, 2, 1, 3)
    assert metrics["precision"] == pytest.approx(3 / 5)
    assert metrics["recall"] == pytest.approx(3 / 4)
    assert metrics["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))
    assert metrics["threshold"] == 0.5


def test_threshold_is_inclusive_and_changes_the_result() -> None:
    y_true = np.array([0, 1])
    y_score = np.array([0.2, 0.5])
    assert evaluate.predict_at_threshold(y_score, 0.5).tolist() == [0, 1]
    assert evaluate.threshold_metrics(y_true, y_score, 0.6)["recall"] == 0.0
    assert evaluate.threshold_metrics(y_true, y_score, 0.1)["fp"] == 1


def test_threshold_metrics_without_any_flagged_transaction() -> None:
    metrics = evaluate.threshold_metrics(np.array([0, 1]), np.array([0.1, 0.2]), 0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_bootstrap_ci_is_reproducible(scored: tuple[np.ndarray, np.ndarray]) -> None:
    metrics = {"pr_auc": average_precision_score}
    first = evaluate.bootstrap_cis(*scored, metrics, n_boot=200, seed=1)
    second = evaluate.bootstrap_cis(*scored, metrics, n_boot=200, seed=1)
    other_seed = evaluate.bootstrap_cis(*scored, metrics, n_boot=200, seed=2)
    assert first == second
    assert first != other_seed


def test_bootstrap_ci_contains_the_point_estimate(scored: tuple[np.ndarray, np.ndarray]) -> None:
    result = evaluate.evaluate_scores(*scored, n_boot=300)
    assert result["pr_auc_ci_low"] < result["pr_auc"] < result["pr_auc_ci_high"]
    assert result["recall_ci_low"] <= result["recall"] <= result["recall_ci_high"]
    assert result["pr_auc_ci_low"] >= 0 and result["pr_auc_ci_high"] <= 1


def test_bootstrap_ci_widens_with_the_confidence_level(
    scored: tuple[np.ndarray, np.ndarray],
) -> None:
    metrics = {"pr_auc": average_precision_score}
    wide = evaluate.bootstrap_cis(*scored, metrics, n_boot=300, confidence=0.95)["pr_auc"]
    narrow = evaluate.bootstrap_cis(*scored, metrics, n_boot=300, confidence=0.5)["pr_auc"]
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_bootstrap_is_stratified(scored: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_score = scored
    fraud_counts: list[int] = []

    def count_fraud(labels: np.ndarray, scores: np.ndarray) -> float:
        fraud_counts.append(int(labels.sum()))
        assert len(labels) == len(y_true)
        return 0.0

    evaluate.bootstrap_cis(y_true, y_score, {"count": count_fraud}, n_boot=50)
    assert set(fraud_counts) == {int(y_true.sum())}


def test_bootstrap_requires_both_classes() -> None:
    with pytest.raises(ValueError, match="Both classes"):
        evaluate.bootstrap_cis(np.zeros(10), np.zeros(10), {"pr_auc": average_precision_score})


def test_evaluate_scores_is_reproducible(scored: tuple[np.ndarray, np.ndarray]) -> None:
    assert evaluate.evaluate_scores(*scored, n_boot=100) == evaluate.evaluate_scores(
        *scored, n_boot=100
    )


def test_total_cost_follows_the_cost_model() -> None:
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0])
    amount = np.array([100.0, 250.0, 40.0, 999.0, 0.0])
    # TP: one review. FN: 250 lost. FP: one review. TN: free. FN with zero amount: free.
    assert evaluate.total_cost(y_true, y_pred, amount) == pytest.approx(250 + 2 * REVIEW_COST_EUR)
    assert evaluate.total_cost(y_true, y_pred, amount, review_cost=1.0) == pytest.approx(252.0)


def test_no_model_cost_is_the_sum_of_fraud_amounts() -> None:
    y_true = np.array([1, 1, 0, 0])
    amount = np.array([100.0, 250.0, 40.0, 999.0])
    assert evaluate.no_model_cost(y_true, amount) == pytest.approx(350.0)
    assert evaluate.total_cost(y_true, y_true, amount) == pytest.approx(2 * REVIEW_COST_EUR)
