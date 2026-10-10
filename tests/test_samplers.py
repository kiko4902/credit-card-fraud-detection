import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fraud.data import split_xy, time_based_split
from fraud.models import STRATEGIES, build_pipeline
from fraud.samplers import PolynomFitSMOTE, SafeLevelSMOTE

NEW_SAMPLERS = [SafeLevelSMOTE, PolynomFitSMOTE]
RATIOS = [0.1, 0.5, 1.0]


@pytest.fixture
def scaled(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Training rows as the sampler sees them inside the pipeline: features, scaled."""
    features, target = split_xy(time_based_split(transactions)["train"])
    return build_pipeline("logreg", "smote", 0.5)[:2].fit_transform(features, target), target


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
@pytest.mark.parametrize("ratio", RATIOS)
def test_output_shape_and_minority_count_match_the_ratio(
    sampler_class: type, ratio: float, scaled: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = scaled
    n_majority, n_minority = int((y == 0).sum()), int((y == 1).sum())
    X_out, y_out = sampler_class(sampling_ratio=ratio).fit_resample(X, y)

    expected_minority = int(ratio * n_majority)
    assert int((y_out == 1).sum()) == expected_minority
    assert int((y_out == 0).sum()) == n_majority
    assert X_out.shape == (n_majority + expected_minority, X.shape[1])
    assert len(y_out) == len(X_out)
    assert expected_minority > n_minority
    assert isinstance(X_out, pd.DataFrame) and isinstance(y_out, pd.Series)
    assert list(X_out.columns) == list(X.columns)
    assert y_out.name == y.name
    assert not X_out.isna().any().any()


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
def test_original_rows_are_unchanged(
    sampler_class: type, scaled: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = scaled
    before = X.copy()
    X_out, y_out = sampler_class(sampling_ratio=0.5).fit_resample(X, y)
    pd.testing.assert_frame_equal(X, before)  # input not mutated
    # The original rows, majority and minority, come first and are identical.
    np.testing.assert_array_equal(X_out.iloc[: len(X)].to_numpy(), X.to_numpy())
    np.testing.assert_array_equal(y_out.iloc[: len(y)].to_numpy(), y.to_numpy())
    # Everything appended is synthetic minority.
    assert (y_out.iloc[len(y) :] == 1).all()
    majority_out = X_out[y_out == 0].to_numpy()
    np.testing.assert_array_equal(majority_out, X[y == 0].to_numpy())


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
def test_reproducible_with_a_fixed_seed(
    sampler_class: type, scaled: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = scaled
    first, _ = sampler_class(sampling_ratio=0.5, random_state=7).fit_resample(X, y)
    second, _ = sampler_class(sampling_ratio=0.5, random_state=7).fit_resample(X, y)
    other, _ = sampler_class(sampling_ratio=0.5, random_state=8).fit_resample(X, y)
    pd.testing.assert_frame_equal(first, second)
    assert first.shape == other.shape
    assert not np.array_equal(first.to_numpy(), other.to_numpy())


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
def test_synthetic_rows_are_interpolations_of_real_rows(
    sampler_class: type, scaled: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = scaled
    X_out, _ = sampler_class(sampling_ratio=0.5).fit_resample(X, y)
    synthetic = X_out.iloc[len(X) :]
    assert (synthetic.min() >= X.min() - 1e-9).all()
    assert (synthetic.max() <= X.max() + 1e-9).all()


def test_polynom_fit_stays_inside_the_minority_class(
    scaled: tuple[pd.DataFrame, pd.Series],
) -> None:
    # Star topology: every synthetic point lies between a minority row and the minority mean.
    X, y = scaled
    X_out, _ = PolynomFitSMOTE(sampling_ratio=0.5).fit_resample(X, y)
    minority, synthetic = X[y == 1], X_out.iloc[len(X) :]
    assert (synthetic.min() >= minority.min() - 1e-9).all()
    assert (synthetic.max() <= minority.max() + 1e-9).all()


def test_safe_level_smote_can_interpolate_towards_majority_rows(
    scaled: tuple[pd.DataFrame, pd.Series],
) -> None:
    # Documents a property of the smote-variants implementation (see the class docstring):
    # neighbours are searched in the whole data, so synthetic rows can leave the minority range.
    X, y = scaled
    X_out, _ = SafeLevelSMOTE(sampling_ratio=0.5).fit_resample(X, y)
    minority, synthetic = X[y == 1], X_out.iloc[len(X) :]
    outside = (synthetic < minority.min() - 1e-9) | (synthetic > minority.max() + 1e-9)
    assert outside.any().any()


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
def test_ratio_below_the_existing_balance_is_rejected(
    sampler_class: type, scaled: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = scaled
    with pytest.raises(ValueError, match="already has"):
        sampler_class(sampling_ratio=0.001).fit_resample(X, y)


@pytest.mark.parametrize("sampler_class", NEW_SAMPLERS)
def test_samplers_are_cloneable_and_picklable(sampler_class: type, tmp_path) -> None:
    sampler = sampler_class(sampling_ratio=0.3, random_state=5)
    assert clone(sampler).get_params() == sampler.get_params()
    joblib.dump(sampler, tmp_path / "sampler.joblib")
    assert joblib.load(tmp_path / "sampler.joblib").get_params() == sampler.get_params()


def test_safe_level_smote_uses_k_neighbors() -> None:
    assert SafeLevelSMOTE(k_neighbors=7).get_params()["k_neighbors"] == 7
    assert "k_neighbors" not in PolynomFitSMOTE().get_params()  # no neighbour search


@pytest.mark.parametrize("strategy", ["safe_level_smote", "polynom_fit_smote"])
def test_new_samplers_work_inside_the_pipeline(strategy: str, transactions: pd.DataFrame) -> None:
    splits = time_based_split(transactions)
    features, target = split_xy(splits["train"])
    val_features, _ = split_xy(splits["val"])
    pipeline = build_pipeline("logreg", strategy, 0.5).fit(features, target)
    assert pipeline.predict_proba(val_features).shape == (len(val_features), 2)


def test_every_registered_sampler_hits_the_same_minority_count(
    scaled: tuple[pd.DataFrame, pd.Series],
) -> None:
    # SMOTE-Tomek and SMOTE-ENN remove rows after oversampling, so they are the exceptions.
    X, y = scaled
    counts = {}
    for name, spec in STRATEGIES.items():
        if spec.sampler_factory is None or name in ("smote_tomek", "smote_enn"):
            continue
        _, y_out = spec.sampler_factory(0.5, 42).fit_resample(X, y)
        counts[name] = int((y_out == 1).sum())
    assert len(set(counts.values())) == 1, counts
