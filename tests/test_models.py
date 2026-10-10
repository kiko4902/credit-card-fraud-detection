from typing import Any

import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from fraud.config import TARGET_COLUMN
from fraud.data import split_xy, time_based_split
from fraud.features import MODEL_FEATURES
from fraud.models import (
    K_NEIGHBORS,
    MODELS,
    STRATEGIES,
    AutoWeightXGBClassifier,
    build_pipeline,
    is_supported,
)

# Small, fast variants of the ensemble models for tests.
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 10, "n_jobs": 1},
    "extra_trees": {"n_estimators": 10, "n_jobs": 1},
    "xgboost": {"n_estimators": 10, "n_jobs": 1},
    "lightgbm": {"n_estimators": 10, "n_jobs": 1},
    "mlp": {"max_iter": 30},
}
SAMPLING_STRATEGIES = [name for name, spec in STRATEGIES.items() if spec.sampler_factory]
NO_CLASS_WEIGHT = {"gaussian_nb", "knn", "mlp"}


def fast(model: str) -> dict[str, Any]:
    return FAST_PARAMS.get(model, {})


@pytest.fixture
def train_val(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = time_based_split(transactions)
    return splits["train"], splits["val"]


def test_registry_contents() -> None:
    assert set(MODELS) == {
        "logreg",
        "decision_tree",
        "gaussian_nb",
        "knn",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "mlp",
    }
    assert set(STRATEGIES) == {
        "none",
        "class_weight",
        "smote",
        "smote_tomek",
        "smote_enn",
        "safe_level_smote",
        "polynom_fit_smote",
    }


def test_excluded_methods_are_not_registered() -> None:
    assert not {"undersample", "borderline_smote", "adasyn"} & set(STRATEGIES)
    assert "svm" not in MODELS


def test_class_weight_support() -> None:
    for model in MODELS:
        assert is_supported(model, "class_weight") == (model not in NO_CLASS_WEIGHT)
        assert is_supported(model, "none") and is_supported(model, "smote")
    with pytest.raises(ValueError, match="does not support class weights"):
        build_pipeline("knn", "class_weight")


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_supported_combination_builds_in_the_right_order(model: str, strategy: str) -> None:
    if not is_supported(model, strategy):
        pytest.skip("class weights not supported by this classifier")
    pipeline = build_pipeline(model, strategy)
    names = [name for name, _ in pipeline.steps]
    assert isinstance(pipeline, Pipeline)
    assert names[0] == "features"
    assert names[-1] == "model"
    assert ("sampler" in names) == (strategy in SAMPLING_STRATEGIES)
    assert names == [n for n in ("features", "scale", "sampler", "model") if n in names]


def test_scaling_only_where_needed() -> None:
    for model in ("logreg", "knn", "mlp"):
        assert "scale" in build_pipeline(model).named_steps
    for model in ("decision_tree", "gaussian_nb", "random_forest", "extra_trees", "xgboost"):
        assert "scale" not in build_pipeline(model).named_steps
    assert "scale" not in build_pipeline("lightgbm", "class_weight").named_steps
    # Every sampler gets scaled features, also in front of a tree model.
    for strategy in SAMPLING_STRATEGIES:
        assert "scale" in build_pipeline("lightgbm", strategy).named_steps


def test_class_weight_strategy_sets_model_weights() -> None:
    for model in sorted(set(MODELS) - NO_CLASS_WEIGHT - {"xgboost"}):
        assert build_pipeline(model).named_steps["model"].get_params()["class_weight"] is None
        weighted = build_pipeline(model, "class_weight").named_steps["model"]
        assert weighted.get_params()["class_weight"] in {"balanced", "balanced_subsample"}


def test_xgboost_class_weight_is_computed_from_the_fitted_data(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, _ = train_val
    features, target = split_xy(train)
    plain = build_pipeline("xgboost", model_params=fast("xgboost")).fit(features, target)
    assert not isinstance(plain.named_steps["model"], AutoWeightXGBClassifier)
    assert plain.named_steps["model"].get_params()["scale_pos_weight"] is None

    pipeline = build_pipeline("xgboost", "class_weight", model_params=fast("xgboost"))
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] is None  # not yet known
    pipeline.fit(features, target)
    expected = (target == 0).sum() / (target == 1).sum()
    assert pipeline.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(expected)

    # Fitting on other rows recomputes the weight from those rows only.
    half = len(features) // 2
    pipeline.fit(features.iloc[:half], target.iloc[:half])
    expected_half = (target.iloc[:half] == 0).sum() / (target.iloc[:half] == 1).sum()
    refit = pipeline.named_steps["model"].get_params()["scale_pos_weight"]
    assert refit == pytest.approx(expected_half)


@pytest.mark.parametrize("strategy", SAMPLING_STRATEGIES)
def test_sampling_ratio_and_k_are_passed_to_the_sampler(strategy: str) -> None:
    def ratio_of(sampler: Any) -> float:
        params = sampler.get_params()
        return params.get("sampling_ratio", params.get("sampling_strategy"))

    assert ratio_of(build_pipeline("logreg", strategy, 0.1).named_steps["sampler"]) == 0.1
    assert ratio_of(build_pipeline("logreg", strategy).named_steps["sampler"]) == 1.0
    params = build_pipeline("logreg", strategy, 0.1).named_steps["sampler"].get_params()
    neighbours = [value for name, value in params.items() if name.endswith("k_neighbors")]
    assert all(value == K_NEIGHBORS == 5 for value in neighbours)
    if strategy != "polynom_fit_smote":
        assert neighbours


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm")
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_pipeline("logreg", "adasyn")
    with pytest.raises(ValueError, match="does not resample"):
        build_pipeline("logreg", "none", sampling_ratio=0.5)
    with pytest.raises(ValueError, match="does not resample"):
        build_pipeline("logreg", "class_weight", sampling_ratio=0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        build_pipeline("logreg", "smote", sampling_ratio=1.5)


def test_random_seed_is_set_on_every_random_step() -> None:
    for model in MODELS:
        for strategy in STRATEGIES:
            if not is_supported(model, strategy):
                continue
            pipeline = build_pipeline(model, strategy, random_state=7)
            model_params = pipeline.named_steps["model"].get_params()
            if "random_state" in model_params:  # Gaussian NB and k-NN are deterministic
                assert model_params["random_state"] == 7
            if "sampler" in pipeline.named_steps:
                assert pipeline.named_steps["sampler"].get_params()["random_state"] == 7
    assert {m for m in MODELS if "random_state" not in MODELS[m].factory(0).get_params()} == {
        "gaussian_nb",
        "knn",
    }


@pytest.mark.parametrize("model", sorted(MODELS))
def test_time_is_not_a_model_input(
    model: str, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, _ = train_val
    features, target = split_xy(train)
    assert "Time" in features.columns  # the pipeline receives it ...
    pipeline = build_pipeline(model, model_params=fast(model)).fit(features, target)
    seen_by_model = list(pipeline.named_steps["model"].feature_names_in_)
    assert seen_by_model == MODEL_FEATURES  # ... but the model does not
    assert "Time" not in seen_by_model
    assert "Amount" not in seen_by_model
    assert TARGET_COLUMN not in seen_by_model


@pytest.mark.parametrize("model", sorted(MODELS))
def test_same_seed_gives_identical_predictions(
    model: str, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, val = train_val
    features, target = split_xy(train)
    val_features, _ = split_xy(val)
    scores = [
        build_pipeline(model, "smote", 0.5, model_params=fast(model))
        .fit(features, target)
        .predict_proba(val_features)[:, 1]
        for _ in range(2)
    ]
    np.testing.assert_array_equal(scores[0], scores[1])
    assert scores[0].shape == (len(val),)
    assert ((scores[0] >= 0) & (scores[0] <= 1)).all()


@pytest.mark.parametrize("strategy", SAMPLING_STRATEGIES)
def test_predictions_have_one_row_per_input_row(
    strategy: str, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, val = train_val
    features, target = split_xy(train)
    val_features, _ = split_xy(val)
    pipeline = build_pipeline("logreg", strategy, 0.5).fit(features, target)
    assert pipeline.predict_proba(val_features).shape == (len(val), 2)
    assert len(pipeline.predict(val_features)) == len(val)


def test_sampler_runs_only_during_fit_and_only_on_training_data(
    train_val: tuple[pd.DataFrame, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    train, val = train_val
    features, target = split_xy(train)
    val_features, _ = split_xy(val)
    calls: list[dict[str, Any]] = []
    original = SMOTE.fit_resample

    def spy(self: SMOTE, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        X_resampled, y_resampled = original(self, X, y)
        calls.append({"index": X.index, "rows_out": len(X_resampled), "y_out": y_resampled})
        return X_resampled, y_resampled

    monkeypatch.setattr(SMOTE, "fit_resample", spy)
    pipeline = build_pipeline("logreg", "smote", 0.5).fit(features, target)

    # Fit: the sampler saw exactly the training rows and changed their class balance.
    assert len(calls) == 1
    assert calls[0]["index"].equals(train.index)
    assert not set(calls[0]["index"]) & set(val.index)
    assert calls[0]["rows_out"] > len(train)
    balance_before = target.sum() / (target == 0).sum()
    balance_after = calls[0]["y_out"].sum() / (calls[0]["y_out"] == 0).sum()
    assert balance_before < 0.05
    assert balance_after == pytest.approx(0.5, abs=0.01)

    # Predict: one output row per validation row, and the sampler is not called again.
    assert len(pipeline.predict_proba(val_features)) == len(val)
    assert len(pipeline.predict(val_features)) == len(val)
    assert len(calls) == 1


def test_scaler_is_fitted_on_training_data_only(
    train_val: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, val = train_val
    features, target = split_xy(train)
    pipeline = build_pipeline("logreg").fit(features, target)
    assert pipeline.named_steps["scale"].n_samples_seen_ == len(train)
    transformed = pipeline[:-1].transform(features)
    assert transformed["V1"].mean() == pytest.approx(0.0, abs=1e-9)
    # Validation data is transformed with the training statistics, so it is not exactly centred.
    val_transformed = pipeline[:-1].transform(split_xy(val)[0])
    assert len(val_transformed) == len(val)
    assert abs(val_transformed["V1"].mean()) > 1e-6
