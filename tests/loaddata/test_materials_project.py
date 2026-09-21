import pandas as pd
import pytest

from kalematerials.loaddata.materials_project import load_materials_project, TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM


def test_load_materials_project(tmp_path):
    path = tmp_path / "mp.csv"
    pd.DataFrame(
        {
            "composition": ["FeNi", "NdFe"],
            "material_id": ["mp-1", "mp-2"],
            "total_magnetization_normalized_vol": [0.1, 0.2],
            "is_magnetic": [False, True],
        }
    ).to_csv(path, index=False)
    data = load_materials_project(path, formula_column="formula", target_column="target")
    assert data["formula"].tolist() == ["FeNi", "NdFe"]
    assert data["sample_id"].tolist() == ["mp-1", "mp-2"]
    assert data["target"].iloc[0] == pytest.approx(0.1 * TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM)
    assert data["is_magnetic"].tolist() == [False, True]
