import numpy as np
import pytest

from kalematerials.embed.composition_descriptors import add_engineered_features, ENGINEERED_FEATURE_COLUMNS
from kalematerials.loaddata.element_tables import load_element_properties
from kalematerials.pipeline.prediction import check_bundle_matches, fit_bundle, predict_formulas, PREDICTION_COLUMN
from kalematerials.predict.sklearn_models import MODEL_REGISTRY
from kalematerials.utils.persistence import TRAINING_FINGERPRINT

from ..conftest import formulas


def test_fit_bundle_and_predict():
    tables = load_element_properties()
    data = add_engineered_features(formulas("Fe", "Co", "FeCo", "Ni"), *tables)
    X, y = data[list(ENGINEERED_FEATURE_COLUMNS)], data["mean_valence"]
    bundle = fit_bundle(X, y, MODEL_REGISTRY["linear"], target_column="valence", dataset="toy")
    assert bundle.feature_columns == ENGINEERED_FEATURE_COLUMNS and bundle.n_train == 4
    predictions, skipped = predict_formulas(bundle, formulas("FeNi", "??", "Xx"), *tables)
    assert len(predictions) == 1 and np.isfinite(predictions[PREDICTION_COLUMN]).all() and skipped == ["??", "Xx"]


def test_check_bundle_matches(toy_xy):
    X, y = toy_xy
    bundle = fit_bundle(X, y, MODEL_REGISTRY["linear"], target_column="t", dataset="toy")
    check_bundle_matches(bundle, X, y)
    for other in (X.iloc[:-1], X + 1, X.rename(columns={"a": "z"}), X[["b", "a", "c"]]):
        with pytest.raises(ValueError):
            check_bundle_matches(bundle, other, y.iloc[: len(other)])
    bundle.metadata.pop(TRAINING_FINGERPRINT)
    with pytest.warns(UserWarning):
        check_bundle_matches(bundle, X, y)
