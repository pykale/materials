"""Evaluate prediction-interval calibration on ID, OOD and size-matched control splits.

Fit one model per split, apply the three interval methods in predict.uncertainty and score pooled samples.
"""

from __future__ import annotations

import warnings
from typing import List, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kalematerials.evaluate.calibration import summarize_across_seeds, summarize_calibration
from kalematerials.loaddata.splits import (
    build_kfold_splits,
    build_size_matched_controls,
    ID_KFOLD,
    ID_RANDOM,
    OOD,
    Split,
)
from kalematerials.predict.uncertainty import (
    CONFORMAL,
    CONFORMAL_NORM,
    ConformalCalibrator,
    gaussian_half_width,
    RF_STD,
    rf_tree_std,
)
from kalematerials.utils.registry import ModelSpec
from kalematerials.utils.reporting import Logger, SILENT

# Models with per-tree predictions for rf_tree_std.
ENSEMBLE_STD_MODELS = frozenset({"rf"})

# Metrics from evaluate.calibration.
UNCERTAINTY_METRICS = ("mae", "rmse", "mean_sigma", "coverage", "coverage_error", "mean_width", "mean_abs_z", "rms_z")


def _fit_and_predict(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    train_model,
    *,
    calibration_fraction: float,
    alpha: float,
    seed: int,
    model_random_state: int,
) -> Optional[pd.DataFrame]:
    """Fit and calibrate on separate parts of `train_idx`, then predict `test_idx`.

    Return one row per test sample and interval method, or None if the training set is too small.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(train_idx))
    n_calibration = int(round(len(shuffled) * calibration_fraction))
    if n_calibration < 2 or len(shuffled) - n_calibration < 2:
        return None

    cal_idx, fit_idx = shuffled[:n_calibration], shuffled[n_calibration:]

    model = train_model(X.iloc[fit_idx], y.iloc[fit_idx], hyperparams=None, random_state=model_random_state)

    sigma_cal = rf_tree_std(model, X.iloc[cal_idx])
    y_cal = y.iloc[cal_idx].to_numpy(dtype=float)
    pred_cal = model.predict(X.iloc[cal_idx])

    sigma = rf_tree_std(model, X.iloc[test_idx])
    y_true = y.iloc[test_idx].to_numpy(dtype=float)
    y_pred = model.predict(X.iloc[test_idx])

    half_widths = {
        RF_STD: gaussian_half_width(sigma, alpha),
        CONFORMAL: ConformalCalibrator.fit(y_cal, pred_cal, alpha).half_width(sigma),
        CONFORMAL_NORM: ConformalCalibrator.fit(y_cal, pred_cal, alpha, sigma_cal).half_width(sigma),
    }

    return pd.concat(
        [
            pd.DataFrame(
                {"method": method, "y_true": y_true, "y_pred": y_pred, "sigma": sigma, "half_width": half_width}
            )
            for method, half_width in half_widths.items()
        ],
        ignore_index=True,
    )


def _collect_samples(
    X: pd.DataFrame,
    y: pd.Series,
    scenario_splits: Sequence[Tuple[str, str, List[Split]]],
    train_model,
    *,
    seed: int,
    calibration_fraction: float,
    alpha: float,
    model_random_state: int,
) -> pd.DataFrame:
    """Predict every split of every scenario for one seed and stack the per-sample rows."""
    frames = []
    for split_type, scenario, splits in scenario_splits:
        for split_id, train_idx, test_idx in splits:
            samples = _fit_and_predict(
                X,
                y,
                train_idx,
                test_idx,
                train_model,
                calibration_fraction=calibration_fraction,
                alpha=alpha,
                seed=seed,
                model_random_state=model_random_state,
            )
            if samples is None:
                warnings.warn(f"Skipping {scenario} {split_id}: too few training rows to calibrate.")
                continue
            samples.insert(0, "n_train", len(train_idx))
            samples.insert(0, "split_id", split_id)
            samples.insert(0, "scenario", scenario)
            samples.insert(0, "split_type", split_type)
            samples.insert(0, "seed", seed)
            frames.append(samples)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve_uncertainty_model(registry: Mapping[str, ModelSpec], model_key: str) -> ModelSpec:
    """Resolve a model key, requiring membership in ENSEMBLE_STD_MODELS."""
    if model_key not in registry:
        raise ValueError(f"Unknown model {model_key!r}. Available: {sorted(registry)}")

    spec = registry[model_key]
    if model_key not in ENSEMBLE_STD_MODELS:
        usable = sorted(ENSEMBLE_STD_MODELS & set(registry))
        raise ValueError(
            f"Model {model_key!r} ({spec.name}) exposes no ensemble spread, which the uncertainty estimators need. "
            f"Models that do: {usable}"
        )
    return spec


class UncertaintyTables(NamedTuple):
    """Calibration tables at three aggregation levels.

    `by_split` groups by seed, split and interval method. `pooled_by_seed` pools samples per seed,
    split type and method. `across_seeds` gives means and standard deviations of `pooled_by_seed`.
    """

    by_split: pd.DataFrame
    pooled_by_seed: pd.DataFrame
    across_seeds: pd.DataFrame


def run_uncertainty_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ModelSpec,
    scenarios: Sequence[Tuple[str, List[Split]]],
    *,
    seeds: Sequence[int] = (0,),
    alpha: float = 0.05,
    calibration_fraction: float = 0.25,
    inner_folds: int = 5,
    inner_shuffle: bool = True,
    size_matched_control: bool = True,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> Optional[UncertaintyTables]:
    """Evaluate interval calibration for an ENSEMBLE_STD_MODELS spec across ID and OOD splits.

    `scenarios` contains (scenario_name, splits) pairs; each seed runs one repeat. `alpha` is nominal
    miscoverage (0.05 gives 95% intervals). Reserve `calibration_fraction` of training rows for calibration.
    `inner_folds` and `inner_shuffle` define the ID reference; `size_matched_control` adds random controls.
    Return UncertaintyTables, or None if no split has enough training rows.
    """
    if not scenarios:
        raise ValueError("No scenario has a split; nothing to score.")

    logger.write(
        f"\n[INFO] Uncertainty run: {spec.name}, {len(scenarios)} OOD scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"{calibration_fraction:.0%} of each training set held out for conformal calibration.\n"
    )

    per_seed_frames = []
    for seed in seeds:
        scenario_splits: List[Tuple[str, str, List[Split]]] = [
            (ID_KFOLD, ID_KFOLD, build_kfold_splits(len(X), inner_folds, inner_shuffle, seed))
        ]
        for scenario, splits in scenarios:
            scenario_splits.append((OOD, scenario, splits))
            if size_matched_control:
                controls = build_size_matched_controls(splits, len(X), seed, scenario)
                scenario_splits.append((ID_RANDOM, scenario, list(controls.values())))

        per_seed_frames.append(
            _collect_samples(
                X,
                y,
                scenario_splits,
                spec.train,
                seed=seed,
                calibration_fraction=calibration_fraction,
                alpha=alpha,
                model_random_state=model_random_state,
            )
        )

    frames = [frame for frame in per_seed_frames if not frame.empty]
    if not frames:
        warnings.warn("No uncertainty samples produced; nothing to report.")
        return None
    samples = pd.concat(frames, ignore_index=True)

    by_split = summarize_calibration(samples, ("seed", "split_type", "scenario", "split_id", "method"), alpha)
    pooled_by_seed = summarize_calibration(samples, ("seed", "split_type", "method"), alpha)
    return UncertaintyTables(
        by_split=by_split,
        pooled_by_seed=pooled_by_seed,
        across_seeds=summarize_across_seeds(pooled_by_seed, ("split_type", "method")),
    )
