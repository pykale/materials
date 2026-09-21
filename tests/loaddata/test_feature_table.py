import numpy as np
import pandas as pd
import pytest

from kalematerials.loaddata.feature_table import load_feature_table, resolve_feature_columns, standardize_records

from ..conftest import FORMULA, TARGET


def test_standardize_records():
    raw = pd.DataFrame({FORMULA: [" FeNi ", "Fe"], TARGET: ["1.2", "inf"], "is_magnetic": ["TRUE", "0"], "id": [1, 2]})
    data = standardize_records(raw)
    assert data[FORMULA].tolist() == ["FeNi", "Fe"]
    assert data[TARGET].iloc[0] == 1.2 and np.isnan(data[TARGET].iloc[1])
    assert data["is_magnetic"].tolist() == [True, False]
    assert list(data.columns) == [FORMULA, TARGET, "is_magnetic", "id"]


def test_standardize_records_rejects_bad_input():
    with pytest.raises(ValueError, match="Missing column"):
        standardize_records(pd.DataFrame({FORMULA: ["Fe"]}))
    with pytest.raises(ValueError, match="Unknown is_magnetic"):
        standardize_records(pd.DataFrame({FORMULA: ["Fe"], TARGET: [1], "is_magnetic": ["maybe"]}))


def test_resolve_feature_columns():
    data = pd.DataFrame({"a": [1.0], "b": [2], "note": ["x"]})
    assert resolve_feature_columns(data, ["b", "a"]) == ["b", "a"]
    for bad in ([], ["nope"], ["a", "note"]):
        with pytest.raises(ValueError):
            resolve_feature_columns(data, bad)


def test_load_feature_table(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({FORMULA: ["Fe", "Ni"], TARGET: [1.0, 2.0], "f": [0.1, 0.2], "sample_id": ["a", "b"]}).to_csv(path)
    X, y, metadata = load_feature_table(path, ["f"], target_column=TARGET)
    assert list(X.columns) == ["f"] and y.tolist() == [1.0, 2.0]
    assert metadata[FORMULA].tolist() == ["Fe", "Ni"] and "sample_id" in metadata
    with pytest.raises(ValueError, match="target"):
        load_feature_table(path, ["f"], target_column="missing")
