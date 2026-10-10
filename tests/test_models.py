from typing import Any

import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from fraud.config import TARGET_COLUMN
from fraud.data import split_xy, time_based_split
from fraud.features import MODEL_FEATURES
from fraud.models import MODELS, STRATEGIES, build_pipeline

# Small, fast variants of the tree models for tests.
FAST_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {},
    "random_forest": {"n_estimators": 10, "n_jobs": 1},
    "lightgbm": {"n_estimators": 10, "n_jobs": 1},
}
SAMPLING_STRATEGIES = [name for name, spec in STRATEGIES.items() if spec.sampler_factory]


@pytest.fixture
def train_val(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = time_based_split(transactions)
    return splits["train"], splits["val"]


def test_registry_contents() -> None:
    assert set(MODELS) == {"logreg", "random_forest", "lightgbm"}
    assert set(STRATEGIES) == {
        "none",
        "class_weight",
        "undersample",
        "smote",
        "borderline_smote",
        "adasyn",
        "smote_tomek",
    }


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("strategy", sorted(STRATEGIES))
def test_every_combination_builds_in_the_right_order(model: str, strategy: str) -> None:
    pipeline = build_pipeline(model, strategy)
    names = [name for name, _ in pipeline.steps]
    assert isinstance(pipeline, Pipeline)
    assert names[0] == "features"
    assert names[-1] == "model"
    assert ("sampler" in names) == (strategy in SAMPLING_STRATEGIES)
    expected = [n for n in ("features", "scale", "sampler", "model") if n in names]
    assert names == expected


def test_scaling_only_where_needed() -> None:
    assert "scale" in build_pipeline("logreg").named_steps
    assert "scale" not in build_pipeline("random_forest").named_steps
    assert "scale" not in build_pipeline("lightgbm", "undersample").named_steps
    # Distance-based samplers need a common scale even in front of a tree model.
    assert "scale" in build_pipeline("lightgbm", "smote").named_steps


def test_class_weight_strategy_sets_model_weights() -> None:
    for model in MODELS:
        assert build_pipeline(model).named_steps["model"].get_params()["class_weight"] is None
        weighted = build_pipeline(model, "class_weight").named_steps["model"]
        assert weighted.get_params()["class_weight"] in {"balanced", "balanced_subsample"}


@pytest.mark.parametrize("strategy", SAMPLING_STRATEGIES)
def test_sampling_ratio_is_passed_to_the_sampler(strategy: str) -> None:
    assert build_pipeline("logreg", strategy, 0.1).named_steps["sampler"].sampling_strategy == 0.1
    assert build_pipeline("logreg", strategy).named_steps["sampler"].sampling_strategy == "auto"


def test_invalid_configurations_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm")
    with pytest.raises(ValueError, match="Unknown strategy"):
        build_pipeline("logreg", "magic")
    with pytest.raises(ValueError, match="does not resample"):
        build_pipeline("logreg", "none", sampling_ratio=0.5)
    with pytest.raises(ValueError, match="does not resample"):
        build_pipeline("logreg", "class_weight", sampling_ratio=0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        build_pipeline("logreg", "smote", sampling_ratio=1.5)


def test_random_seed_is_set_on_every_random_step() -> None:
    for model in MODELS:
        for strategy in STRATEGIES:
            pipeline = build_pipeline(model, strategy, random_state=7)
            assert pipeline.named_steps["model"].get_params()["random_state"] == 7
            if "sampler" in pipeline.named_steps:
                assert pipeline.named_steps["sampler"].get_params()["random_state"] == 7


@pytest.mark.parametrize("model", sorted(MODELS))
def test_time_is_not_a_model_input(
    model: str, train_val: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, _ = train_val
    features, target = split_xy(train)
    assert "Time" in features.columns  # the pipeline receives it ...
    pipeline = build_pipeline(model, model_params=FAST_PARAMS[model]).fit(features, target)
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
        build_pipeline(model, "smote", 0.5, model_params=FAST_PARAMS[model])
        .fit(features, target)
        .predict_proba(val_features)[:, 1]
        for _ in range(2)
    ]
    assert (scores[0] == scores[1]).all()


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
