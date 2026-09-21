import numpy as np
import pandas as pd
import pytest

from kalematerials.loaddata.splits import ID_RANDOM, ID_REFERENCE, OOD
from kalematerials.pipeline import ood

from ..conftest import FORMULA


def scenario_inputs(toy_xy):
    X, y = toy_xy
    elements = [["Fe", "Co"] if i % 2 else ["Fe", "Ni"] for i in range(len(X))]
    return X, y, elements, {"Fe": 8, "Co": 9, "Ni": 10}, {"Fe": 4, "Co": 5, "Ni": 4}


def test_build_scenarios(toy_xy):
    inputs = scenario_inputs(toy_xy)
    names = [name for name, _ in ood.build_scenarios(*inputs, min_train=5, min_test=5)]
    assert names == ["LOEO", "LOPO", "LOGO", "LOCO(k=5)", "SparseX", "SparseY"]
    built = dict(
        ood.build_scenarios(
            *inputs, scenarios=("element", "system"), systems=(("Fe", "Co"),), elements=["Co"], max_splits=1
        )
    )
    assert [s[0] for s in built["LOEO"]] == ["E=Co"] and [s[0] for s in built["LOSO"]] == ["S=Fe-Co"]
    assert ood.build_scenarios(*inputs, scenarios=()) == []
    with pytest.warns(UserWarning, match="LOEO: no split passed"):
        assert ood.build_scenarios(*inputs, scenarios=("element",), min_test=100) == []
    with pytest.warns(UserWarning, match=r"LOSO: \['S=Co-Ni'\] did not pass"):
        ood.build_scenarios(*inputs, scenarios=("system",), systems=(("Fe", "Co"), ("Co", "Ni")))
    assert ood.heldout_label("E=Fe") == "Fe" and ood.heldout_label("SparseX_top10pct") == ""


def test_resolve_split_elements(periodic_table):
    elements, group, period = ood.resolve_split_elements(
        pd.DataFrame({FORMULA: ["FeNi", "Nd"]}), FORMULA, periodic_table
    )
    assert elements == [["Fe", "Ni"], ["Nd"]] and group == {"Fe": 8, "Ni": 10} and period["Nd"] == 6
    with pytest.raises(ValueError, match=FORMULA):
        ood.resolve_split_elements(pd.DataFrame({"x": [1]}), FORMULA, periodic_table)


def test_run_ood_evaluation(toy_xy, specs):
    X, y, elements, group, period = scenario_inputs(toy_xy)
    scenarios = ood.build_scenarios(X, y, elements, group, period, scenarios=("element",), elements=["Co", "Ni"])
    splits, results = ood.run_ood_evaluation(X, y, specs, scenarios, seeds=(0,), inner_folds=2)
    assert splits["split_id"].tolist() == ["E=Co", "E=Ni"] and splits["heldout_label"].tolist() == ["Co", "Ni"]
    assert set(results["split_type"]) == {OOD, ID_REFERENCE, ID_RANDOM} and len(results) == 2 * 3 * 2 * 4
    _, no_control = ood.run_ood_evaluation(X, y, specs, scenarios, inner_folds=2, size_matched_control=False)
    assert ID_RANDOM not in set(no_control["split_type"])
    with pytest.raises(ValueError):
        ood.run_ood_evaluation(X, y, specs, [])
    with pytest.warns(UserWarning, match="no inner folds"):
        assert ood.run_ood_evaluation(X, y, specs, [("tiny", [("t", np.arange(1), np.arange(1, 5))])], inner_folds=2)[
            1
        ].empty
