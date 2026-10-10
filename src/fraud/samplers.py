"""Safe-Level-SMOTE and Polynom-Fit SMOTE as imblearn-compatible samplers.

imbalanced-learn does not ship these two methods, so they are implemented here from the
descriptions in the original papers:

- Bunkhumpornpat, Sinapiromsaran & Lursinsap (2009), "Safe-Level-SMOTE: Safe-Level-Synthetic
  Minority Over-Sampling TEchnique for Handling the Class Imbalanced Problem", PAKDD 2009.
- Gazzah & Amara (2008), "New Oversampling Approaches Based on Polynomial Fitting for
  Imbalanced Data Sets", DAS 2008 (star topology only).

Both take the same ``sampling_ratio`` (minority:majority after resampling) as the
imbalanced-learn samplers and return exactly ``int(sampling_ratio * n_majority)`` minority
rows. The original rows are kept untouched and in place; the synthetic rows are appended.
Where a paper leaves a detail open, the choice made is stated in the class docstring.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

from fraud.config import RANDOM_SEED

MINORITY: int = 1
MAJORITY: int = 0


def neighbours_without_self(
    points: np.ndarray, query_index: np.ndarray, k: int, n_jobs: int | None = None
) -> np.ndarray:
    """Indices of the ``k`` nearest rows of ``points`` for the rows ``query_index`` of it.

    A row is never its own neighbour, also when ``points`` holds exact copies of it.
    """
    search = NearestNeighbors(n_neighbors=k + 1, n_jobs=n_jobs).fit(points)
    found = search.kneighbors(points[query_index], return_distance=False)
    neighbours = np.empty((len(query_index), k), dtype=int)
    for row, (own, candidates) in enumerate(zip(query_index, found, strict=True)):
        neighbours[row] = candidates[candidates != own][:k]
    return neighbours


def safe_level_gap_bounds(sl_p: np.ndarray, sl_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bound of the interpolation gap for the five cases of the paper.

    ``sl_p`` and ``sl_n`` are the safe levels of a minority row ``p`` and of its selected
    neighbour ``n``; the synthetic row is ``p + gap * (n - p)``. The bounds are NaN where no
    synthetic row is generated.

    1. ``sl_p = 0`` and ``sl_n = 0``: both are noise, nothing is generated.
    2. ``sl_p > 0`` and ``sl_n = 0``: ``n`` is noise, ``gap = 0`` (``p`` is duplicated).
    3. ``sl_p = sl_n``: ``gap`` in ``[0, 1]``, as in SMOTE.
    4. ``sl_p > sl_n``: ``gap`` in ``[0, sl_n / sl_p]``, closer to ``p``.
    5. ``sl_p < sl_n``: ``gap`` in ``[1 - sl_p / sl_n, 1]``, closer to ``n``.
    """
    sl_p = np.asarray(sl_p, dtype=float)
    sl_n = np.asarray(sl_n, dtype=float)
    low = np.zeros_like(sl_p)
    high = np.ones_like(sl_p)
    closer_to_p = sl_p > sl_n
    closer_to_n = sl_p < sl_n
    # sl_n = 0 gives an upper bound of 0, which is case 2.
    high[closer_to_p] = sl_n[closer_to_p] / sl_p[closer_to_p]
    low[closer_to_n] = 1 - sl_p[closer_to_n] / sl_n[closer_to_n]
    nothing = (sl_p == 0) & (sl_n == 0)
    low[nothing] = np.nan
    high[nothing] = np.nan
    return low, high


class MinorityOversampler(BaseEstimator):
    """Base class: append synthetic minority rows until ``sampling_ratio`` is reached."""

    def __init__(self, sampling_ratio: float = 1.0, random_state: int = RANDOM_SEED) -> None:
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state

    def _generate(
        self, features: np.ndarray, target: np.ndarray, n_new: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Return ``n_new`` synthetic minority rows."""
        raise NotImplementedError

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``X`` and ``y`` with synthetic minority rows appended."""
        features = np.asarray(X, dtype=float)
        target = np.asarray(y)
        n_minority = int((target == MINORITY).sum())
        n_majority = int((target == MAJORITY).sum())
        # Same rounding as imbalanced-learn, so every sampler yields the same minority count.
        n_new = int(self.sampling_ratio * n_majority) - n_minority
        if n_minority == 0 or n_new <= 0:
            raise ValueError(
                f"sampling_ratio={self.sampling_ratio} asks for {n_new + n_minority} minority "
                f"rows but the data already has {n_minority}."
            )
        rng = np.random.default_rng(self.random_state)
        generated = self._generate(features, target, n_new, rng)

        columns = getattr(X, "columns", None)
        synthetic = pd.DataFrame(generated, columns=columns)
        X_out = pd.concat([pd.DataFrame(X, columns=columns), synthetic], ignore_index=True)
        y_out = pd.concat(
            [pd.Series(target), pd.Series(np.full(n_new, MINORITY, dtype=target.dtype))],
            ignore_index=True,
        ).rename(getattr(y, "name", None))
        return X_out, y_out


class SafeLevelSMOTE(MinorityOversampler):
    """Safe-Level-SMOTE (Bunkhumpornpat et al., 2009).

    The safe level of a row is the number of minority rows among its ``k_neighbors`` nearest
    neighbours in the whole training data. For a minority row ``p`` and a randomly selected
    neighbour ``n``, the synthetic row is placed closer to the one with the higher safe
    level (see :func:`safe_level_gap_bounds`). As in the paper's pseudo-code, the gap is
    drawn separately for every attribute, so a synthetic row lies in the box spanned by
    ``p`` and ``n`` rather than exactly on the segment between them.

    Choices where the paper is open:

    - ``n`` is one of the ``k_neighbors`` nearest **minority** rows of ``p`` (the paper's
      input is the set of positive instances, and it treats ``n`` as a positive instance),
      while safe levels are counted over all rows. Synthetic rows therefore never lie
      between a minority and a majority row.
    - The paper makes one pass over the minority rows, which gives at most one synthetic row
      per minority row. To reach the requested ratio the pass is repeated, each time over
      the minority rows in a new random order with newly selected neighbours.
    """

    def __init__(
        self, sampling_ratio: float = 1.0, k_neighbors: int = 5, random_state: int = RANDOM_SEED
    ) -> None:
        super().__init__(sampling_ratio=sampling_ratio, random_state=random_state)
        self.k_neighbors = k_neighbors

    def _generate(
        self, features: np.ndarray, target: np.ndarray, n_new: int, rng: np.random.Generator
    ) -> np.ndarray:
        minority_index = np.flatnonzero(target == MINORITY)
        minority = features[minority_index]
        if len(minority) < 2:
            raise ValueError("Safe-Level-SMOTE needs at least 2 minority rows.")
        k_all = min(self.k_neighbors, len(features) - 1)
        k_minority = min(self.k_neighbors, len(minority) - 1)

        # Safe levels: the search covers every row, so it uses all cores.
        around = neighbours_without_self(features, minority_index, k_all, n_jobs=-1)
        safe_level = (target[around] == MINORITY).sum(axis=1)
        # Candidate neighbours n: positions within ``minority``.
        candidates = neighbours_without_self(minority, np.arange(len(minority)), k_minority)
        if not ((safe_level[:, None] > 0) | (safe_level[candidates] > 0)).any():
            raise ValueError(
                "Safe-Level-SMOTE cannot generate anything: every minority row and its "
                "minority neighbours have safe level 0."
            )

        rows: list[np.ndarray] = []
        n_generated = 0
        while n_generated < n_new:
            p = rng.permutation(len(minority))
            n = candidates[p, rng.integers(k_minority, size=len(p))]
            low, high = safe_level_gap_bounds(safe_level[p], safe_level[n])
            keep = ~np.isnan(low)
            p, n, low, high = p[keep], n[keep], low[keep], high[keep]
            gap = rng.uniform(low[:, None], high[:, None], size=(len(p), features.shape[1]))
            rows.append(minority[p] + gap * (minority[n] - minority[p]))
            n_generated += len(p)
        return np.vstack(rows)[:n_new]


class PolynomFitSMOTE(MinorityOversampler):
    """Polynom-Fit SMOTE, star topology (Gazzah & Amara, 2008).

    Synthetic rows are placed at evenly spaced positions on the straight segments between
    each minority row and the mean of the minority class. The method uses no
    nearest-neighbour search (so it has no ``k``) and never looks at the majority class.

    The paper also describes bus, mesh and polynomial-curve topologies; only the star
    topology is implemented. To reach the requested ratio exactly, every minority row gets
    ``n_new // n_minority`` synthetic rows and a seeded random subset of the minority rows
    gets one more; this is the only random part.
    """

    def _generate(
        self, features: np.ndarray, target: np.ndarray, n_new: int, rng: np.random.Generator
    ) -> np.ndarray:
        minority = features[target == MINORITY]
        centre = minority.mean(axis=0)
        per_row = np.full(len(minority), n_new // len(minority))
        extra = rng.choice(len(minority), size=n_new % len(minority), replace=False)
        per_row[extra] += 1

        origin = np.repeat(np.arange(len(minority)), per_row)
        # Position i of m on a segment is i / (m + 1): the ends themselves are never used.
        first_of_row = np.repeat(np.cumsum(per_row) - per_row, per_row)
        step = np.arange(n_new) - first_of_row + 1
        fraction = step / (per_row[origin] + 1)
        return minority[origin] + fraction[:, None] * (centre - minority[origin])
