"""Calibration metrics for prediction intervals, pooled over samples."""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

SAMPLE_COLUMNS = ("y_true", "y_pred", "sigma", "half_width")


def compute_calibration_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    half_width: Sequence[float],
    sigma: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Score prediction intervals y_pred ± half_width at nominal coverage 1 - alpha.

    Return sample counts (`n`, `n_zero_sigma`), point errors (`mae`, `rmse`), `mean_sigma`, `mean_width`,
    `coverage` and `coverage_error` (coverage minus nominal). `mean_abs_z` and `rms_z` summarize
    |error| / sigma, excluding zero-sigma samples; calibrated Gaussian sigma gives rms_z = 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    half_width = np.asarray(half_width, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    error = np.abs(y_pred - y_true)
    coverage = float(np.mean(error <= half_width))

    scored = sigma > 0
    z = error[scored] / sigma[scored]

    return {
        "n": int(error.size),
        "n_zero_sigma": int((~scored).sum()),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_sigma": float(np.mean(sigma)),
        "coverage": coverage,
        "coverage_error": coverage - (1.0 - alpha),
        "mean_width": float(np.mean(2.0 * half_width)),
        "mean_abs_z": float(np.mean(z)) if z.size else np.nan,
        "rms_z": float(np.sqrt(np.mean(z**2))) if z.size else np.nan,
    }


def summarize_calibration(samples: pd.DataFrame, by: Sequence[str], alpha: float = 0.05) -> pd.DataFrame:
    """Compute calibration metrics per group of `by` from a frame with SAMPLE_COLUMNS."""
    missing = [c for c in (*SAMPLE_COLUMNS, *by) if c not in samples.columns]
    if missing:
        raise ValueError(f"Missing columns in per-sample frame: {missing}")

    rows: List[Dict] = []
    for keys, group in samples.groupby(list(by), sort=True):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            compute_calibration_metrics(
                group["y_true"],
                group["y_pred"],
                group["half_width"],
                group["sigma"],
                alpha,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_across_seeds(
    per_seed: pd.DataFrame,
    by: Sequence[str],
    metrics: Sequence[str] = ("mae", "coverage", "mean_sigma", "mean_width", "rms_z"),
) -> pd.DataFrame:
    """Return metric means and standard deviations across seeds, grouped by `by`.

    `per_seed` comes from summarize_calibration grouped with `seed`; `by` excludes `seed`.
    """
    if per_seed.empty:
        return pd.DataFrame()

    available = [m for m in metrics if m in per_seed.columns]
    summary = per_seed.groupby(list(by), sort=True)[available].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_seeds"] = per_seed.groupby(list(by), sort=True).size()

    return summary.reset_index().fillna({f"{m}_std": 0.0 for m in available})
