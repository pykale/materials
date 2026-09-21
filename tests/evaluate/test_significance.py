import numpy as np
import pytest
from scipy import stats

from kalematerials.evaluate.significance import _corrected_ttest, compare_models_significance

A, B = [1.0, 2.0, 3.0, 4.0, 5.0, 6.5], [1.5, 2.5, 3.0, 4.5, 5.5, 7.0]


def test_uncorrected_matches_scipy():
    t, p, low, high = _corrected_ttest(np.array(A), np.array(B), test_train_ratio=0.0)
    expected = stats.ttest_rel(A, B)
    assert (t, p) == pytest.approx((expected.statistic, expected.pvalue)) and low < np.mean(A) - np.mean(B) < high
    assert _corrected_ttest(np.array(A), np.array(B), test_train_ratio=0.5)[1] > p  # Correction increases the p-value.
    assert all(np.isnan(_corrected_ttest(np.ones(3), np.ones(3), 0.0)))


def test_compare_models_significance():
    result = compare_models_significance({"a": {"mse": A}, "b": {"mse": B}}, "a", "b", test_train_ratio=0.25)
    assert result.n_pairs == 6 and result.mean_difference < 0 and "corrected" in result.note
    few = compare_models_significance({"a": {"mse": A[:3]}, "b": {"mse": B[:3]}}, "a", "b")
    assert np.isnan(few.t_pvalue) and "fewer than" in few.note
    same = compare_models_significance({"a": {"mse": A}, "b": {"mse": A}}, "a", "b")
    assert same.note == "all differences are zero"
    for bad in ({"a": {"mse": A}}, {"a": {"mae": A}, "b": {"mse": B}}, {"a": {"mse": A}, "b": {"mse": B[:5]}}):
        with pytest.raises(ValueError):
            compare_models_significance(bad, "a", "b")
