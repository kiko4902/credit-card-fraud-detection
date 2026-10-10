import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fraud.data import split_xy, time_based_split
from fraud.models import STRATEGIES, build_pipeline
from fraud.samplers import (
    PolynomFitSMOTE,
    SafeLevelSMOTE,
    neighbours_without_self,
    safe_level_gap_bounds,
)

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


def test_safe_level_smote_stays_inside_the_minority_class(
    scaled: tuple[pd.DataFrame, pd.Series],
) -> None:
    # The neighbour n is always a minority row, so no synthetic row leaves the minority range.
    X, y = scaled
    X_out, _ = SafeLevelSMOTE(sampling_ratio=0.5).fit_resample(X, y)
    minority, synthetic = X[y == 1], X_out.iloc[len(X) :]
    assert (synthetic.min() >= minority.min() - 1e-9).all()
    assert (synthetic.max() <= minority.max() + 1e-9).all()


def test_safe_level_gap_bounds_follow_the_five_cases_of_the_paper() -> None:
    sl_p = np.array([0, 3, 2, 4, 1, 0])
    sl_n = np.array([0, 0, 2, 1, 4, 3])
    low, high = safe_level_gap_bounds(sl_p, sl_n)
    assert np.isnan(low[0]) and np.isnan(high[0])  # 1: both noise, nothing generated
    assert (low[1], high[1]) == (0.0, 0.0)  # 2: n is noise, p is duplicated
    assert (low[2], high[2]) == (0.0, 1.0)  # 3: equal safe levels, plain SMOTE
    assert (low[3], high[3]) == (0.0, 0.25)  # 4: p safer, gap in [0, sl_n / sl_p]
    assert (low[4], high[4]) == (0.75, 1.0)  # 5: n safer, gap in [1 - sl_p / sl_n, 1]
    assert (low[5], high[5]) == (1.0, 1.0)  # 5 with sl_p = 0: n is duplicated


def test_neighbours_never_include_the_row_itself_even_with_exact_copies() -> None:
    points = np.array([[0.0], [0.0], [0.0], [1.0], [5.0]])
    neighbours = neighbours_without_self(points, np.arange(5), k=2)
    assert all(row not in found for row, found in enumerate(neighbours))
    assert set(neighbours[0]) == {1, 2}
    assert set(neighbours[4]) <= {0, 1, 2, 3} and 3 in neighbours[4]


def noisy_minority_data() -> tuple[pd.DataFrame, pd.Series]:
    """A tight minority cluster, plus two minority rows deep inside the majority class."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.1, size=(12, 2))
    noise = np.array([[10.0, 10.0], [-10.0, 10.0]])
    majority = np.vstack(
        [rng.normal(centre, 0.3, size=(60, 2)) for centre in ([10, 10], [-10, 10], [0, 6])]
    )
    X = pd.DataFrame(np.vstack([majority, cluster, noise]), columns=["a", "b"])
    y = pd.Series([0] * len(majority) + [1] * 14, name="Class")
    return X, y


def test_safe_level_smote_generates_nothing_around_noise_rows() -> None:
    # The two noise rows have safe level 0 and every minority neighbour of theirs is in the
    # cluster, so a synthetic row started from them is a copy of a cluster row (case 5 with
    # sl_p = 0). Plain SMOTE would place rows on the segments towards the noise rows.
    X, y = noisy_minority_data()
    X_out, _ = SafeLevelSMOTE(sampling_ratio=0.5).fit_resample(X, y)
    cluster, synthetic = X.iloc[180:192], X_out.iloc[len(X) :]
    assert len(synthetic) == 90 - 14
    assert (synthetic.min() >= cluster.min() - 1e-12).all()
    assert (synthetic.max() <= cluster.max() + 1e-12).all()


def test_safe_level_smote_fails_clearly_when_every_minority_row_is_noise() -> None:
    rng = np.random.default_rng(0)
    majority = rng.normal(0.0, 0.1, size=(200, 2)) + np.repeat([[0, 0], [50, 50]], 100, axis=0)
    X = pd.DataFrame(np.vstack([majority, [[0.0, 0.0], [50.0, 50.0]]]), columns=["a", "b"])
    y = pd.Series([0] * 200 + [1] * 2, name="Class")
    with pytest.raises(ValueError, match="safe level 0"):
        SafeLevelSMOTE(sampling_ratio=0.5).fit_resample(X, y)


def test_polynom_fit_places_evenly_spaced_rows_on_the_segments_to_the_mean() -> None:
    minority = np.array([[0.0, 0.0], [4.0, 0.0], [2.0, 6.0]])  # mean (2, 2)
    majority = np.arange(40, dtype=float).reshape(20, 2) + 100
    X = pd.DataFrame(np.vstack([majority, minority]), columns=["a", "b"])
    y = pd.Series([0] * 20 + [1] * 3, name="Class")
    # 9 minority rows asked for: 6 new, 2 per minority row at 1/3 and 2/3 of the segment.
    X_out, _ = PolynomFitSMOTE(sampling_ratio=0.45).fit_resample(X, y)
    expected = np.array([[2, 2], [4, 4], [10, 2], [8, 4], [6, 14], [6, 10]]) / 3
    np.testing.assert_allclose(X_out.iloc[len(X) :].to_numpy(), expected)


def test_polynom_fit_spreads_the_remainder_over_different_minority_rows(
    scaled: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = scaled
    minority = X[y == 1].to_numpy()
    X_out, _ = PolynomFitSMOTE(sampling_ratio=0.5).fit_resample(X, y)
    synthetic = X_out.iloc[len(X) :].to_numpy()
    centre = minority.mean(axis=0)
    # Each synthetic row is x + t * (mean - x) for one minority row x and one t in (0, 1).
    counts = np.zeros(len(minority), dtype=int)
    direction = centre - minority
    for row in synthetic:
        fraction = ((row - minority) * direction).sum(axis=1) / (direction**2).sum(axis=1)
        residual = np.abs(minority + fraction[:, None] * direction - row).max(axis=1)
        origin = int(residual.argmin())
        assert residual[origin] < 1e-9 and 0 < fraction[origin] < 1
        counts[origin] += 1
    assert counts.max() - counts.min() <= 1


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
