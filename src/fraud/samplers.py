"""Safe-Level-SMOTE and Polynom-Fit SMOTE as imblearn-compatible samplers.

imbalanced-learn does not ship these two methods. The generation itself is done by the
``smote-variants`` package; the classes here only adapt it to this project:

- they take the same ``sampling_ratio`` (minority:majority after resampling) as the
  imbalanced-learn samplers, instead of smote-variants' ``proportion``;
- they return exactly that many minority rows (smote-variants rounds, and the star topology
  of Polynom-Fit SMOTE generates a whole number of points per minority row), by generating
  slightly more than needed and keeping a seeded random subset;
- they keep the original rows untouched and in place, and return pandas objects.

``smote_variants`` is imported only inside ``fit_resample``, so a fitted pipeline can be
loaded and used for prediction without the package being installed.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from fraud.config import RANDOM_SEED

MINORITY: int = 1
MAJORITY: int = 0


class SmoteVariantsSampler(BaseEstimator):
    """Base class: oversample the minority class to ``sampling_ratio`` with smote-variants."""

    # Name of the smote_variants class that generates the synthetic rows.
    _variant: str = ""

    def __init__(self, sampling_ratio: float = 1.0, random_state: int = RANDOM_SEED) -> None:
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state

    def _variant_params(self) -> dict[str, Any]:
        return {}

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        """Return ``X`` and ``y`` with synthetic minority rows appended."""
        import smote_variants

        logging.getLogger("smote_variants").setLevel(logging.WARNING)
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

        # Ask for one extra synthetic row per minority row, then trim to the exact number.
        proportion = (n_new + n_minority) / (n_majority - n_minority)
        generator = getattr(smote_variants, self._variant)(
            proportion=proportion, random_state=self.random_state, **self._variant_params()
        )
        resampled, labels = generator.sample(features, target)
        generated = resampled[len(features) :]
        if not np.array_equal(resampled[: len(features)], features):
            raise RuntimeError(f"{self._variant} changed the original rows.")
        if not (labels[len(features) :] == MINORITY).all() or len(generated) < n_new:
            raise RuntimeError(
                f"{self._variant} generated {len(generated)} rows, {n_new} minority rows needed."
            )
        rng = np.random.default_rng(self.random_state)
        keep = np.sort(rng.choice(len(generated), size=n_new, replace=False))

        columns = getattr(X, "columns", None)
        synthetic = pd.DataFrame(generated[keep], columns=columns)
        X_out = pd.concat([pd.DataFrame(X, columns=columns), synthetic], ignore_index=True)
        y_out = pd.concat(
            [pd.Series(target), pd.Series(np.full(n_new, MINORITY, dtype=target.dtype))],
            ignore_index=True,
        ).rename(getattr(y, "name", None))
        return X_out, y_out


class SafeLevelSMOTE(SmoteVariantsSampler):
    """Safe-Level-SMOTE (Bunkhumpornpat et al., 2009).

    Like SMOTE, but each synthetic point is placed closer to whichever end of the segment
    has more minority rows among its k nearest neighbours (the higher "safe level"), and a
    minority row whose neighbourhood is entirely majority is not interpolated from.

    Note on the smote-variants implementation used here: the k nearest neighbours are
    searched in the **whole** training data, so the second end of a segment can be a
    majority row (if that row has minority neighbours itself). In the original paper the
    neighbour is another minority row. At 540:1 this matters: most neighbours of a fraud row
    are legitimate rows, so part of the synthetic fraud is placed between a fraud row and a
    legitimate one.
    """

    _variant = "Safe_Level_SMOTE"

    def __init__(
        self, sampling_ratio: float = 1.0, k_neighbors: int = 5, random_state: int = RANDOM_SEED
    ) -> None:
        super().__init__(sampling_ratio=sampling_ratio, random_state=random_state)
        self.k_neighbors = k_neighbors

    def _variant_params(self) -> dict[str, Any]:
        # The neighbour search covers every row, so it uses all cores.
        return {"n_neighbors": self.k_neighbors, "n_jobs": os.cpu_count() or 1}


class PolynomFitSMOTE(SmoteVariantsSampler):
    """Polynom-Fit SMOTE, star topology (Gazzah & Amara, 2008).

    Synthetic points are placed on the straight segments between each minority row and the
    mean of the minority class. It uses no nearest-neighbour search, so it has no ``k``.
    """

    _variant = "polynom_fit_SMOTE_star"
