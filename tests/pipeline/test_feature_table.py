import pandas as pd
import pytest

from kalematerials.embed.composition_descriptors import ENGINEERED_FEATURE_COLUMNS
from kalematerials.loaddata.feature_table import standardize_records
from kalematerials.pipeline.feature_table import build_feature_table, filter_samples

from ..conftest import FORMULA, TARGET


def test_build_feature_table(periodic_table, miedema):
    raw = pd.DataFrame(
        {
            FORMULA: ["FeNi", "Fe2Ni2", "NiFe", "Fe0.5Ni0.5", "Fe2Ni"],
            TARGET: [0.05, 0.10, 1.5, 2.5, 0.8],
            "sample_id": list("abcde"),
        }
    )
    original = raw.copy()
    table, features = build_feature_table(raw, periodic_table, miedema)
    assert features == list(ENGINEERED_FEATURE_COLUMNS) and list(table.columns) == [
        TARGET,
        *features,
        "n_records",
        "sample_id",
    ]
    assert list(table.index) == ["Fe2Ni", "FeNi"] and table.index.name == FORMULA
    assert table.loc["FeNi", TARGET] == 2.0  # Median after min_target filtering.
    assert table.loc["FeNi", "n_records"] == 2 and pd.isna(table.loc["FeNi", "sample_id"])
    assert table.loc["Fe2Ni", "sample_id"] == "e" and table.loc["FeNi", "mixing_enthalpy"] == -2.0
    pd.testing.assert_frame_equal(raw, original)


def test_incomplete_features_are_dropped_with_a_warning(periodic_table, miedema):
    raw = pd.DataFrame({FORMULA: ["FeNi", "FeNd", "??"], TARGET: [1.5, 1.5, 1.5]})
    with pytest.warns(UserWarning, match="mixing_enthalpy"):
        table, _ = build_feature_table(raw, periodic_table, miedema)
    assert list(table.index) == ["FeNi"]
    empty, features = build_feature_table(raw.iloc[:0], periodic_table, miedema)
    assert empty.empty and list(empty.columns) == [TARGET, *features, "n_records"]


def test_filter_samples():
    raw = standardize_records(
        pd.DataFrame(
            {
                FORMULA: ["FeNi", "Fe", "NdFe", "UFe", "Ni", None],
                TARGET: [0.18, 1, 1, 1, 1, 1],
                "is_magnetic": [True, False, True, True, None, True],
            }
        )
    )
    assert filter_samples(raw, excluded_elements=["Nd", "U"], magnetic_only=True)[FORMULA].tolist() == ["FeNi"]
    assert len(filter_samples(raw, min_target=None)) == 5 and len(filter_samples(raw, min_target=0.5)) == 4
    with pytest.raises(ValueError, match="is_magnetic"):
        filter_samples(raw.drop(columns="is_magnetic"), magnetic_only=True)
    with pytest.raises(ValueError, match="boolean"):
        filter_samples(raw.assign(is_magnetic=[1, 0, 1, 1, 0, 1]), magnetic_only=True)
