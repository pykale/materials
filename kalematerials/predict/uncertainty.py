"""Prediction intervals from forest spread and split-conformal calibration.

rf_std: Gaussian intervals using the standard deviation of per-tree predictions.
conformal: Constant width calibrated on absolute residuals.
conformal_norm: Width scales with sigma, calibrated on residuals divided by sigma.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

# Floors keep sigma positive when normalizing residuals.
_SIGMA_FLOOR_FRACTION = 1e-3
_MIN_SIGMA = 1e-12


def _sigma_floor(sigma: np.ndarray) -> float:
    """Return a fraction of mean sigma, or _MIN_SIGMA when the mean is zero."""
    mean = float(np.mean(sigma))
    return _SIGMA_FLOOR_FRACTION * mean if mean > 0 else _MIN_SIGMA


RF_STD = "rf_std"
CONFORMAL = "conformal"
CONFORMAL_NORM = "conformal_norm"


def rf_tree_std(model, X) -> np.ndarray:
    """Return the standard deviation of predictions across a fitted forest's `estimators_`."""
    if not hasattr(model, "estimators_"):
        raise TypeError(f"{type(model).__name__} has no estimators_; rf_std needs a forest.")

    values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    return np.stack([tree.predict(values) for tree in model.estimators_]).std(axis=0)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the calibration-score quantile at min(1, ceil((n + 1) * (1 - alpha)) / n)."""
    n = scores.size
    if n == 0:
        raise ValueError("Conformal calibration needs at least one residual.")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


@dataclass(frozen=True)
class ConformalCalibrator:
    """Calibrated split-conformal interval width.

    `quantile` is the half-width, or a sigma multiplier when `normalized` is True.
    Normalized intervals use `sigma_floor`, fixed during calibration, as the minimum sigma.
    """

    quantile: float
    normalized: bool
    sigma_floor: float = 0.0

    @classmethod
    def fit(
        cls,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        alpha: float = 0.05,
        sigma_cal: Optional[np.ndarray] = None,
    ) -> "ConformalCalibrator":
        """Calibrate residuals on held-out data at nominal coverage 1 - alpha.

        Supplying per-sample `sigma_cal` selects sigma-normalized intervals.
        """
        residuals = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float))
        normalized = sigma_cal is not None
        floor = 0.0
        if normalized:
            sigma_cal = np.asarray(sigma_cal, dtype=float)
            floor = _sigma_floor(sigma_cal)
            residuals = residuals / np.maximum(sigma_cal, floor)

        return cls(quantile=_conformal_quantile(residuals, alpha), normalized=normalized, sigma_floor=floor)

    def half_width(self, sigma: np.ndarray) -> np.ndarray:
        """Interval half-width per test sample, given its sigma."""
        sigma = np.asarray(sigma, dtype=float)
        if self.normalized:
            return self.quantile * np.maximum(sigma, self.sigma_floor)
        return np.full(sigma.shape, self.quantile, dtype=float)


def gaussian_half_width(sigma: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Two-sided Gaussian interval half-width at miscoverage `alpha`."""
    z = norm.ppf(1.0 - alpha / 2.0)
    return z * np.asarray(sigma, dtype=float)
