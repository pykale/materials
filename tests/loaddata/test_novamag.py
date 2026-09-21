import json

import pytest

from kalematerials.loaddata.novamag import load_novamag, read_novamag_records


def record(formula, ms):
    return json.dumps(
        {
            "properties": {
                "chemistry": {"chemical formula": {"value": formula}},
                "crystal": {"compound space group": {"value": 1}},
                "magnetics": {"saturation magnetization": {"value": ms}},
            }
        }
    )


def test_load_novamag(tmp_path):
    (tmp_path / "a.json").write_text(record("Fe2Ni2", "0.10"))
    (tmp_path / "b.json").write_text("not json")
    with pytest.warns(UserWarning, match="Skipped 1"):
        data = load_novamag(tmp_path, formula_column="formula", target_column="target")
    assert data["formula"].tolist() == ["Fe2Ni2"] and data["target"].tolist() == [0.1]
    assert data["sample_id"].tolist() == ["a.json"] and data["compound space group"].tolist() == [1]


def test_novamag_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_novamag_records(tmp_path / "missing")
    with pytest.raises(ValueError, match="No readable"):
        read_novamag_records(tmp_path)
