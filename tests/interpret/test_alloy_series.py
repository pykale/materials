import pytest
from sklearn.linear_model import LinearRegression

from kalematerials.embed.composition_descriptors import add_engineered_features, ENGINEERED_FEATURE_COLUMNS
from kalematerials.interpret.alloy_series import AlloySeries, build_series_features, build_series_panels
from kalematerials.loaddata.element_tables import load_element_properties

from ..conftest import FORMULA, formulas

SERIES = AlloySeries("FeCo", "Co", ["Fe90Co10", "Fe50Co50"], {0.1: 2.2, 0.5: 2.4})


def test_series_features_and_panels():
    periodic_table, miedema = load_element_properties()
    features, stoich = build_series_features(SERIES.formulas, periodic_table, miedema)
    assert list(features[FORMULA]) == SERIES.formulas and stoich.loc[1, "Co"] == 50
    training = add_engineered_features(formulas("Fe", "Co", "FeCo"), periodic_table, miedema)
    model = LinearRegression().fit(training[list(ENGINEERED_FEATURE_COLUMNS)], [2.2, 1.8, 2.4])
    (panel,) = build_series_panels([SERIES], ENGINEERED_FEATURE_COLUMNS, {"Linear": model}, periodic_table, miedema)
    assert panel.title == "FeCo Case Study" and list(panel.x) == [0.1, 0.5] and len(panel.predictions["Linear"]) == 2
    with pytest.raises(ValueError):
        build_series_panels([SERIES], ENGINEERED_FEATURE_COLUMNS, {}, periodic_table, miedema)
