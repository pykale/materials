"""Paired significance tests for models evaluated on the same splits."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

from kalematerials.evaluate.metrics import FoldScores

MIN_PAIRS_FOR_TEST = 6


@dataclass(frozen=True)
class SignificanceResult:
    """Paired model comparison with a corrected t-test and confidence interval.

    `mean_difference` is mean(a) - mean(b); negative favors model_a for lower-is-better metrics.
    `ci_low` and `ci_high` use the same corrected variance as the test. Unavailable test values are NaN;
    `note` explains skipped tests or limitations.
    """

    metric: str
    model_a: str
    model_b: str
    n_pairs: int
    t_stat: float
    t_pvalue: float
    mean_difference: float
    ci_low: float = np.nan
    ci_high: float = np.nan
    note: Optional[str] = None


def _paired_scores(results: FoldScores, model_a: str, model_b: str, metric: str):
    """Return the models' paired score arrays; raise ValueError for missing or unequal-length scores."""
    for name in (model_a, model_b):
        if name not in results:
            raise ValueError(f"Model {name!r} not in results: {sorted(results)}")
        if metric not in results[name]:
            raise ValueError(f"Metric {metric!r} not recorded for {name!r}.")

    a = np.asarray(results[model_a][metric], dtype=float)
    b = np.asarray(results[model_b][metric], dtype=float)
    if len(a) != len(b):
        raise ValueError(f"Paired count mismatch: {model_a} has {len(a)}, {model_b} has {len(b)}")
    return a, b


def _corrected_ttest(a: np.ndarray, b: np.ndarray, test_train_ratio: float, confidence: float = 0.95):
    """Paired t-test with the Nadeau-Bengio (2003) correction for shared training data.

    Scale the sample variance by 1/n + test_train_ratio; ratio 0 gives the ordinary paired t-test.
    `a` and `b` are paired scores, `test_train_ratio` is n_test / n_train, and `confidence` is two-sided.
    Return (t_stat, p_value, ci_low, ci_high) with n - 1 degrees of freedom; all NaN for zero variance.
    """
    differences = a - b
    n = len(differences)
    variance = float(np.var(differences, ddof=1))
    mean_difference = float(np.mean(differences))
    if variance == 0.0:
        return np.nan, np.nan, np.nan, np.nan

    standard_error = np.sqrt((1.0 / n + test_train_ratio) * variance)
    t_stat = mean_difference / standard_error
    p_value = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    half_width = stats.t.ppf(0.5 + confidence / 2.0, df=n - 1) * standard_error
    return t_stat, float(p_value), mean_difference - half_width, mean_difference + half_width


def compare_models_significance(
    results: FoldScores,
    model_a: str,
    model_b: str,
    metric: str = "mse",
    min_pairs: int = MIN_PAIRS_FOR_TEST,
    test_train_ratio: float = 0.0,
) -> SignificanceResult:
    """Compare paired model scores with a mean difference, confidence interval and corrected t-test.

    `results` maps model names to metrics to score lists, with one observation per distinct test set.
    Use lower-is-better metrics; tests below `min_pairs` are skipped.
    `test_train_ratio` is n_test / n_train: 1 / (K - 1) for K-fold or (1 - train_size) / train_size
    for random splits. Use 0 when observations share no training data.
    """
    a, b = _paired_scores(results, model_a, model_b, metric)
    mean_difference = float(np.mean(a) - np.mean(b))

    def _result(t_stat, t_p, ci=(np.nan, np.nan), note=None):
        return SignificanceResult(
            metric=metric,
            model_a=model_a,
            model_b=model_b,
            n_pairs=len(a),
            t_stat=float(t_stat),
            t_pvalue=float(t_p),
            mean_difference=mean_difference,
            ci_low=float(ci[0]),
            ci_high=float(ci[1]),
            note=note,
        )

    if len(a) < min_pairs:
        return _result(
            np.nan,
            np.nan,
            note=f"only {len(a)} paired observation(s); fewer than {min_pairs}",
        )

    t_stat, t_p, ci_low, ci_high = _corrected_ttest(a, b, test_train_ratio)
    ci = (ci_low, ci_high)

    if not np.any(a != b):
        return _result(t_stat, t_p, ci=ci, note="all differences are zero")

    note = "variance corrected for observations that share training data" if test_train_ratio > 0 else None
    return _result(t_stat, t_p, ci=ci, note=note)
