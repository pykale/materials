import numpy as np
import pandas as pd
import pytest

from kalematerials.prepdata import composition as c

from ..conftest import formulas


def test_composition_key():
    assert {c.get_composition_key(f) for f in ["FeNi", "NiFe", "Fe2Ni2", "Fe0.5Ni0.5", "(FeNi)2"]} == {"FeNi"}
    assert c.get_composition_key("Fe2Ni") == c.get_composition_key("Ni2Fe4") != "FeNi"
    assert c.get_composition_key("NaN") == "NNa"  # Hill order, so CSV readers do not see a missing value
    assert c.get_composition_key(None) is None and c.get_composition_key(np.nan) is None
    with pytest.warns(UserWarning, match="Could not normalize"):
        assert [c.get_composition_key(f) for f in ["invalid", "Fe1e400", "Fe0"]] == [None] * 3


def test_elements():
    assert c.get_elements("Nd2Fe14B") == ["Nd", "Fe", "B"] and c.get_elements(None) == []
    frame = formulas("FeSi", "FeS", None)
    assert c.get_elements_per_row(frame) == [["Fe", "Si"], ["Fe", "S"], []]
    assert c.formula_contains_elements(frame, ["S"]).tolist() == [False, True, False]
    with pytest.warns(UserWarning, match="could not be parsed"):
        assert c.get_elements_per_row(pd.Series(["??"])) == [[]]
    with pytest.raises(ValueError):
        c.get_elements_per_row(pd.DataFrame({"x": []}))


def test_group_period_maps(periodic_table):
    group, period = c.get_group_period_maps(periodic_table)
    assert group == {"Fe": 8, "Ni": 10} and period["Nd"] == 6
    with pytest.raises(ValueError, match="period"):
        c.get_group_period_maps(periodic_table.drop(columns="period"))


def test_stoich_array(periodic_table):
    stoich = c.get_stoich_array(formulas("FeNi", "Fe0.5Ni0.5", "Fe3Ni", None), periodic_table)
    assert stoich[["Fe", "Ni"]].values.tolist() == [[1, 1], [0.5, 0.5], [3, 1], [0, 0]]
    assert c.get_stoich_array("Fe3Ni", periodic_table).loc[0, "Fe"] == 3
    with pytest.warns(UserWarning, match="not in the periodic table"):
        assert c.get_stoich_array("FeCo", periodic_table).loc[0, "Fe"] == 1
    with pytest.warns(UserWarning, match="Could not parse"):
        assert (c.get_stoich_array("!!", periodic_table).iloc[0] == 0).all()


def test_fractions_and_primitives(periodic_table):
    stoich = c.get_stoich_array(formulas("Fe3Ni", "FeNi", "Nd"), periodic_table)
    assert c.get_atomic_fraction(stoich.iloc[0]).to_dict() == {"Fe": 0.75, "Ni": 0.25}
    assert c.get_atomic_fraction(stoich.iloc[0] * 0).empty
    fractions = c.get_atomic_fraction_array(stoich)
    assert fractions.loc[0, "Fe"] == 0.75 and np.isnan(fractions.loc[0, "Nd"])
    weighted = c.get_weighted_property(periodic_table["atomic_weight"], stoich)
    assert weighted.iloc[0] == 0.75 * 55.845 + 0.25 * 58.6934
    assert np.isnan(c.get_weighted_property(periodic_table["atomic_weight"].drop("Nd"), stoich).iloc[2])
    assert np.isnan(c.get_weighted_property(periodic_table["atomic_weight"], stoich * 0).iloc[0])
    entropy = c.get_mixing_entropy(stoich)
    assert entropy.iloc[1] == pytest.approx(np.log(2)) and entropy.iloc[2] == 0
    assert c.get_compound_radix(formulas("Fe", "FeNi", None)).tolist()[:2] == [1, 2]
