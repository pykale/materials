import pytest

from kalematerials.loaddata.splits import ID_KFOLD, ID_RANDOM, OOD
from kalematerials.pipeline import uncertainty as u
from kalematerials.pipeline.ood import build_scenarios
from kalematerials.predict.sklearn_models import MODEL_REGISTRY

from ..conftest import SMALL_RF


def test_resolve_uncertainty_model():
    assert u.resolve_uncertainty_model(MODEL_REGISTRY, "rf").key == "rf"
    with pytest.raises(ValueError, match="Unknown"):
        u.resolve_uncertainty_model(MODEL_REGISTRY, "nope")
    with pytest.raises(ValueError, match="ensemble spread"):
        u.resolve_uncertainty_model(MODEL_REGISTRY, "xgb")


def test_run_uncertainty_evaluation(toy_xy):
    X, y = toy_xy
    elements = [["Fe", "Co"] if i % 2 else ["Fe", "Ni"] for i in range(len(X))]
    scenarios = build_scenarios(X, y, elements, {}, {}, scenarios=("element",), elements=["Co"])
    tables = u.run_uncertainty_evaluation(X, y, SMALL_RF, scenarios, seeds=(0, 1), inner_folds=2, alpha=0.2)
    assert set(tables.by_split["split_type"]) == {ID_KFOLD, OOD, ID_RANDOM}
    assert set(tables.by_split["method"]) == {"rf_std", "conformal", "conformal_norm"}
    assert tables.across_seeds["n_seeds"].tolist() == [2] * 9 and (tables.pooled_by_seed["coverage"] <= 1).all()
    with pytest.raises(ValueError):
        u.run_uncertainty_evaluation(X, y, SMALL_RF, [])
    with pytest.warns(UserWarning, match="too few"):
        assert u.run_uncertainty_evaluation(X, y, SMALL_RF, scenarios, calibration_fraction=0.01, inner_folds=2) is None
