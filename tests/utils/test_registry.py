import pytest

from kalematerials.predict.sklearn_models import MODEL_REGISTRY
from kalematerials.utils.registry import resolve_models, resolve_trained_models


def test_resolve():
    assert [s.name for s in resolve_models(MODEL_REGISTRY, ["xgb", "rf"])] == ["XGBoost", "Random Forest"]
    assert resolve_trained_models(MODEL_REGISTRY, {"rf": object()}, ["rf"])[0].key == "rf"
    with pytest.raises(ValueError, match="Unknown model key"):
        resolve_models(MODEL_REGISTRY, ["rf", "nope"])
    with pytest.raises(ValueError, match="fitted"):
        resolve_trained_models(MODEL_REGISTRY, {"rf": object()}, ["xgb"])
