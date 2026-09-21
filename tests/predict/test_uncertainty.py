import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from kalematerials.predict import uncertainty as u


def test_rf_tree_std(toy_xy):
    X, y = toy_xy
    sigma = u.rf_tree_std(RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y), X)
    assert sigma.shape == (len(X),) and (sigma >= 0).all()
    with pytest.raises(TypeError):
        u.rf_tree_std(LinearRegression().fit(X, y), X)


def test_conformal_quantile_and_calibrator():
    residuals = np.arange(1.0, 11.0)
    assert u._conformal_quantile(residuals, alpha=0.2) == 10.0  # level ceil(11 * 0.8) / 10 = 0.9, method 'higher'
    with pytest.raises(ValueError):
        u._conformal_quantile(np.array([]), 0.1)
    y_cal, pred = np.zeros(10), residuals
    fixed = u.ConformalCalibrator.fit(y_cal, pred, alpha=0.2)
    assert fixed.half_width(np.ones(3)).tolist() == [10.0] * 3
    adaptive = u.ConformalCalibrator.fit(y_cal, pred, alpha=0.2, sigma_cal=residuals)
    assert adaptive.normalized and adaptive.half_width(np.array([2.0])).tolist() == [2.0]  # residual/sigma = 1
    assert u.ConformalCalibrator.fit(y_cal, pred, sigma_cal=np.zeros(10)).sigma_floor == u._MIN_SIGMA


def test_gaussian_half_width():
    assert u.gaussian_half_width(np.array([1.0]), alpha=0.05)[0] == pytest.approx(1.959964, abs=1e-6)
