import numpy as np
import pytest

from kalematerials.embed import composition_descriptors as d
from kalematerials.loaddata.element_tables import load_element_properties
from kalematerials.prepdata.composition import get_stoich_array

from ..conftest import FORMULA, formulas

# Pinned against the packaged reference tables, in ENGINEERED_FEATURE_COLUMNS order.
EXPECTED = {
    "Fe": (1, 0, 55.8452, 4, 8, 1811, 0, 6, 1.83),
    "FeNi": (2, 0.6931471805599453, 57.26932, 4, 9, 1769.5, -2, 5, 1.87),
    "Nd2Fe14B": (3, 0.5783252866601273, 63.5957294117647, 4.117647058823529, 7.705882352941176,
                 1782.1764705882351, -6.006920415224914, 5.470588235294118, 1.7611764705882353),
    "AlCo2Cr": (3, 1.0397207708399179, 49.2110218425, 3.75, 9.25, 1662.3675, -14, 4.75, 1.7575),
}  # fmt: skip


def test_pinned_descriptor_values():
    out = d.add_engineered_features(formulas(*EXPECTED), *load_element_properties()).set_index(FORMULA)
    for formula, expected in EXPECTED.items():
        np.testing.assert_allclose(
            out.loc[formula, list(d.ENGINEERED_FEATURE_COLUMNS)].to_numpy(float), expected, rtol=1e-12
        )


def test_scale_invariance_and_unparseable(periodic_table, miedema):
    with pytest.warns(UserWarning):
        out = d.add_engineered_features(formulas("FeNi", "Fe2Ni2", "??"), periodic_table, miedema)
    features = out[list(d.ENGINEERED_FEATURE_COLUMNS)]
    assert features.iloc[0].equals(features.iloc[1]) and features.iloc[2].isna().all()
    with pytest.raises(ValueError):
        d.add_engineered_features(out.drop(columns=FORMULA), periodic_table, miedema)


def test_group_renormalizes_without_f_block(periodic_table):
    stoich = get_stoich_array(formulas("Fe3Nd", "Fe3Ni", "Nd"), periodic_table)
    assert d.get_weighted_group(periodic_table, stoich).tolist()[:2] == [8.0, 0.75 * 8 + 0.25 * 10]
    assert np.isnan(d.get_weighted_group(periodic_table, stoich).iloc[2])


def test_mixing_enthalpy(periodic_table, miedema):
    stoich = get_stoich_array(formulas("FeNi", "FeNd", "Nd", "Fe2"), periodic_table)
    values = d.get_weighted_mixing_enthalpy(miedema, stoich)
    assert values.iloc[0] == -2.0 and np.isnan(values.iloc[1]) and values.tolist()[2:] == [0.0, 0.0]


def test_annotated_property_columns(periodic_table):
    periodic_table.loc["Nd", "melting_point"] = "912±3 K (639±3 °C)"
    periodic_table.loc["Nd", "electronegativity"] = "Pauling scale: no data"
    stoich = get_stoich_array(formulas("NdFe", "FeNi"), periodic_table)
    assert d.get_weighted_melting_point(periodic_table, stoich).iloc[0] == 0.5 * 912 + 0.5 * 1811
    assert np.isnan(d.get_weighted_electronegativity(periodic_table, stoich).iloc[0])
    assert d.get_weighted_electronegativity(periodic_table, stoich).iloc[1] == 0.5 * 1.83 + 0.5 * 1.91
    periodic_table["melting_point"] = [1e-7, 3e-7, 1.0]  # numeric columns must not go through text extraction
    assert d.get_weighted_melting_point(periodic_table, stoich).iloc[1] == 0.5 * 1e-7 + 0.5 * 3e-7
