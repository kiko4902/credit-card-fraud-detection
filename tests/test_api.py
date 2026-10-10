"""API and serving tests.

They use the committed model (``models/``) and the committed sample of explained
transactions (``app/data/``): no Kaggle data is needed.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest
from app.api import LATENCY_HEADER, app
from fastapi.testclient import TestClient

from fraud.config import APP_SAMPLE_CSV, MAX_BATCH_SIZE, PCA_COLUMNS
from fraud.explain import SHAP_PREFIX
from fraud.features import FEATURE_LABELS, MODEL_FEATURES
from fraud.serving import INPUT_COLUMNS, FraudModel, get_model, model_inputs


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:  # the context manager runs the startup code
        yield test_client


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return pd.read_csv(APP_SAMPLE_CSV)


def payload(row: pd.Series) -> dict[str, float]:
    body = {column: float(row[column]) for column in PCA_COLUMNS}
    return {**body, "Amount": float(row["Amount"]), "seconds_of_day": float(row["Time"])}


@pytest.fixture(scope="module")
def valid(sample: pd.DataFrame) -> dict[str, float]:
    return payload(sample.iloc[0])


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": get_model().version,
    }


def test_model_info_reports_version_threshold_cost_model_and_metrics(client: TestClient) -> None:
    info = client.get("/model-info").json()
    model = get_model()
    assert info["model_version"] == model.version
    assert info["threshold"] == model.threshold
    assert info["cost_model"]["review_cost_per_flagged_transaction"] == 5.0
    assert info["metrics"]["test_at_threshold"]["pr_auc"]["value"] == pytest.approx(0.758, abs=1e-3)
    assert info["input_features"] == INPUT_COLUMNS
    assert info["max_batch_size"] == MAX_BATCH_SIZE
    assert "not a calibrated probability" in info["output"]


def test_predict_returns_score_decision_and_top_features(
    client: TestClient, valid: dict[str, float]
) -> None:
    response = client.post("/predict", json=valid)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"fraud_score", "flagged", "threshold", "top_features", "model_version"}
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["flagged"] == (body["fraud_score"] >= body["threshold"])
    assert body["threshold"] == get_model().threshold
    assert len(body["top_features"]) == 5
    strengths = [abs(feature["contribution"]) for feature in body["top_features"]]
    assert strengths == sorted(strengths, reverse=True)
    for feature in body["top_features"]:
        assert feature["feature"] in MODEL_FEATURES
        assert feature["label"] == FEATURE_LABELS[feature["feature"]]
        expected = "increases" if feature["contribution"] > 0 else "decreases"
        assert feature["direction"] == expected


def test_known_sample_rows_give_the_expected_scores(
    client: TestClient, sample: pd.DataFrame
) -> None:
    # One row of every outcome; the scores were saved by the final evaluation.
    for _, row in sample.groupby("outcome").head(1).iterrows():
        body = client.post("/predict", json=payload(row)).json()
        assert body["fraud_score"] == pytest.approx(row["score"], abs=1e-4)
        assert body["flagged"] == bool(row["flagged"])
        assert body["flagged"] == (row["outcome"] in ("caught_fraud", "false_alarm"))


def test_contributions_equal_the_shap_values_of_the_final_evaluation(
    sample: pd.DataFrame,
) -> None:
    # serving uses XGBoost's own TreeSHAP; the sample holds shap.TreeExplainer values.
    model = get_model()
    values, base = model.contributions(sample)
    expected = sample[[f"{SHAP_PREFIX}{name}" for name in MODEL_FEATURES]].to_numpy()
    np.testing.assert_allclose(values, expected, atol=2e-4)
    assert base == pytest.approx(sample["base_value"].iloc[0], abs=1e-4)
    scores = model.scores(sample)
    # The booster path used for serving gives the scores of the sklearn pipeline.
    pipeline_scores = model.pipeline.predict_proba(model.to_frame(sample))[:, 1]
    np.testing.assert_allclose(scores, pipeline_scores, atol=1e-7)
    inputs = model_inputs(model.pipeline, model.to_frame(sample))
    assert list(inputs.columns) == MODEL_FEATURES
    np.testing.assert_allclose(scores, sample["score"], atol=1e-4)
    np.testing.assert_array_equal(model.flag(scores), sample["flagged"].astype(bool))


def test_latency_is_reported_in_a_header(client: TestClient, valid: dict[str, float]) -> None:
    for response in (client.get("/health"), client.post("/predict", json=valid)):
        assert float(response.headers[LATENCY_HEADER]) > 0


@pytest.mark.parametrize(
    "change",
    [
        {"V1": None},  # required field missing (removed below)
        {"V7": "abc"},
        {"Amount": -1.0},
        {"Amount": 2_000_000.0},
        {"seconds_of_day": 86_400.0},
        {"seconds_of_day": -5.0},
        {"V3": 5_000.0},
        {"unknown_field": 1.0},
    ],
)
def test_invalid_input_returns_422(
    client: TestClient, valid: dict[str, float], change: dict[str, Any]
) -> None:
    body: dict[str, Any] = {**valid, **change}
    if change == {"V1": None}:
        del body["V1"]
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert next(iter(change)) in str(response.json()["detail"])


def test_non_finite_numbers_are_rejected(client: TestClient, valid: dict[str, float]) -> None:
    text = "{" + ", ".join(f'"{k}": {v}' for k, v in valid.items()) + "}"
    broken = text.replace(f'"Amount": {valid["Amount"]}', '"Amount": NaN')
    response = client.post("/predict", content=broken, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_batch_scores_every_transaction_like_single_requests(
    client: TestClient, sample: pd.DataFrame
) -> None:
    rows = sample.head(20)
    response = client.post(
        "/predict/batch", json={"transactions": [payload(row) for _, row in rows.iterrows()]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 20 and len(body["predictions"]) == 20
    scores = [prediction["fraud_score"] for prediction in body["predictions"]]
    np.testing.assert_allclose(scores, rows["score"], atol=1e-4)
    assert body["flagged_count"] == int(rows["flagged"].sum())
    single = client.post("/predict", json=payload(rows.iloc[3])).json()
    assert single["fraud_score"] == pytest.approx(scores[3], abs=1e-6)
    assert single["top_features"] == body["predictions"][3]["top_features"]


def test_batch_limit(client: TestClient, valid: dict[str, float]) -> None:
    at_limit = client.post("/predict/batch", json={"transactions": [valid] * MAX_BATCH_SIZE})
    assert at_limit.status_code == 200 and at_limit.json()["count"] == MAX_BATCH_SIZE
    too_many = client.post("/predict/batch", json={"transactions": [valid] * (MAX_BATCH_SIZE + 1)})
    assert too_many.status_code == 422
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_time_of_day_wraps_and_time_column_is_accepted(valid: dict[str, float]) -> None:
    model = get_model()
    as_time = {key: value for key, value in valid.items() if key != "seconds_of_day"}
    as_time["Time"] = valid["seconds_of_day"] + 86_400  # the same second, one day later
    assert model.scores([as_time])[0] == pytest.approx(model.scores([valid])[0], abs=1e-9)
    with pytest.raises(ValueError, match="Missing input columns"):
        model.scores([{"Amount": 1.0}])


def test_model_without_a_threshold_is_refused() -> None:
    model = get_model()
    with pytest.raises(ValueError, match="no threshold"):
        FraudModel(model.pipeline, {**model.config, "threshold": None})


def test_serving_does_not_import_training_only_packages() -> None:
    # The API image installs neither plotting nor shap nor lightgbm.
    import subprocess
    import sys

    code = (
        "import sys, app.api, fraud.serving; "
        "bad = [m for m in ('shap', 'matplotlib', 'lightgbm', 'seaborn', 'kaggle', 'streamlit') "
        "if m in sys.modules]; print(bad)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
