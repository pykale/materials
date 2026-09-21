import numpy as np
import pandas as pd
import pytest

from kalematerials.evaluate import calibration as c


def test_compute_calibration_metrics():
    out = c.compute_calibration_metrics(
        y_true=[0, 0, 0, 0], y_pred=[1, 1, 1, 1], half_width=[2, 2, 0.5, 0.5], sigma=[1, 1, 0, 2], alpha=0.5
    )
    assert out["n"] == 4 and out["n_zero_sigma"] == 1 and out["mae"] == 1 and out["rmse"] == 1
    assert out["coverage"] == 0.5 and out["coverage_error"] == 0.0 and out["mean_width"] == 2.5
    assert out["mean_abs_z"] == pytest.approx((1 + 1 + 0.5) / 3)


def test_summaries():
    samples = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "method": ["a"] * 4,
            "y_true": [0.0] * 4,
            "y_pred": [1.0, 1.0, 1.0, 3.0],
            "sigma": [1.0] * 4,
            "half_width": [2.0] * 4,
        }
    )
    by_seed = c.summarize_calibration(samples, ("seed", "method"))
    assert by_seed["mae"].tolist() == [1.0, 2.0]
    across = c.summarize_across_seeds(by_seed, ("method",))
    assert across.loc[0, ["mae_mean", "mae_std", "n_seeds"]].tolist() == [1.5, pytest.approx(np.std([1, 2], ddof=1)), 2]
    assert c.summarize_across_seeds(by_seed.iloc[:0], ("method",)).empty
    with pytest.raises(ValueError):
        c.summarize_calibration(samples.drop(columns="sigma"), ("seed",))
