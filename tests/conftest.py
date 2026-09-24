"""Shared reference tables, regression data and lightweight model fixtures."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from kalematerials.predict.sklearn_models import MODEL_REGISTRY
from kalematerials.utils.registry import ModelSpec

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBRARY_ROOT = REPO_ROOT / "kalematerials"
EXAMPLE_ROOT = REPO_ROOT / "examples" / "saturation_magnetism_prediction"
sys.path.insert(0, str(EXAMPLE_ROOT))

FORMULA, TARGET = "chemical formula", "saturation magnetization"


@pytest.fixture
def periodic_table():
    """Fe and Ni with group numbers, plus Nd with an f-block label and no group number."""
    return pd.DataFrame(
        {
            "symbol": ["Fe", "Ni", "Nd"],
            "atomic_weight": [55.845, 58.6934, 144.242],
            "period": [4, 4, 6],
            "group_block": ["group 8, d-block", "group 10, d-block", "group n/a, f-block"],
            "melting_point": [1811, 1728, 1297],
            "valence": [8, 10, 3],
            "electronegativity": ["1.83", "1.91", "1.14"],
        }
    ).set_index("symbol", drop=False)


@pytest.fixture
def miedema():
    return pd.DataFrame([[0.0, -2.0], [-2.0, 0.0]], index=["Fe", "Ni"], columns=["Fe", "Ni"])


@pytest.fixture
def toy_xy():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] + 0.1 * rng.normal(size=40), name=TARGET)
    return X, y


def _train_small_forest(X, y, hyperparams=None, random_state=0):
    return RandomForestRegressor(n_estimators=5, random_state=random_state).fit(X, y)


SMALL_RF = ModelSpec("rf", "Random Forest", _train_small_forest)


@pytest.fixture
def specs():
    return (MODEL_REGISTRY["linear"], SMALL_RF)


def formulas(*names):
    return pd.DataFrame({FORMULA: list(names)})
