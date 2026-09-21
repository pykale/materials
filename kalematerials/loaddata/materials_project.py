"""Load computed magnetic properties from a Materials Project CSV export."""

import numpy as np
import pandas as pd

from kalematerials.loaddata.feature_table import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, standardize_records

# Convert magnetization from μB/Å³ to μ_0 M in tesla.
BOHR_MAGNETON = 9.274e-24  # A·m^2
CUBIC_ANGSTROM = 1e-30  # m^3
VACUUM_PERMEABILITY = 4 * np.pi * 1e-7  # T·m/A
TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM = (BOHR_MAGNETON / CUBIC_ANGSTROM) * VACUUM_PERMEABILITY


def read_materials_project_records(csv_path: str, formula_column: str = DEFAULT_FORMULA_COLUMN) -> pd.DataFrame:
    """Read a Materials Project CSV and rename its composition column to the shared formula column."""
    return pd.read_csv(csv_path).rename(columns={"composition": formula_column})


def load_materials_project(
    csv_path: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load standardized Materials Project records with saturation magnetization in tesla."""
    data = read_materials_project_records(csv_path, formula_column=formula_column)

    volumetric_moment = pd.to_numeric(data["total_magnetization_normalized_vol"], errors="coerce")
    data[target_column] = volumetric_moment * TESLA_PER_BOHR_MAGNETON_PER_CUBIC_ANGSTROM

    data["source"] = "materials_project"
    if "material_id" in data.columns:
        data["sample_id"] = data["material_id"]
    return standardize_records(data, target_column, formula_column)
