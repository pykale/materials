"""Composition descriptors: a fixed-length feature vector from a chemical formula.

Adapted from https://github.com/rich970/ML-alloy-design/blob/master/alloys.py.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from kalematerials.prepdata.composition import (
    get_atomic_fraction_array,
    get_compound_radix,
    get_group_period_maps,
    get_mixing_entropy,
    get_stoich_array,
    get_weighted_property,
)

# mean_* features use atomic-fraction weights.
ENGINEERED_FEATURE_COLUMNS: Tuple[str, ...] = (
    "n_elements",
    "mixing_entropy",
    "mean_atomic_weight",
    "mean_period",
    "mean_group",
    "mean_melting_point",
    "mixing_enthalpy",
    "mean_valence",
    "mean_electronegativity",
)


def _numeric_property(periodic_table, column):
    """Read property values as floats, extracting the first number from text entries."""
    values = periodic_table[column]
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float)
    return pd.to_numeric(values.astype(str).str.extract(r"(-?\d*\.?\d+)")[0], errors="coerce")


def get_weighted_electronegativity(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean Pauling electronegativity."""
    return get_weighted_property(_numeric_property(periodic_table, "electronegativity"), stoich_array)


def get_weighted_atomic_weight(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean atomic weight."""
    return get_weighted_property(periodic_table["atomic_weight"], stoich_array)


def get_weighted_period(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean period number."""
    return get_weighted_property(periodic_table["period"], stoich_array)


def get_weighted_melting_point(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean melting point in K."""
    return get_weighted_property(_numeric_property(periodic_table, "melting_point"), stoich_array)


def get_weighted_valence(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean of the highest oxidation state."""
    return get_weighted_property(periodic_table["valence"], stoich_array)


def get_weighted_group(periodic_table, stoich_array):
    """Atomic-fraction-weighted mean group number, excluding elements without one."""
    element_to_group, _ = get_group_period_maps(periodic_table)
    fractions = get_atomic_fraction_array(stoich_array)
    group = pd.Series(element_to_group, dtype=float).reindex(fractions.columns)
    weights = np.where(group.notna().to_numpy(), fractions.fillna(0.0).to_numpy(), 0.0)
    total = weights.sum(axis=1)
    values = (weights @ group.fillna(0.0).to_numpy()) / np.where(total > 0, total, np.nan)
    return pd.Series(values, index=stoich_array.index)


def get_weighted_mixing_enthalpy(miedema, stoich_array):
    """Mixing enthalpy: 4 * sum_{a<b} f_a f_b H_ab from the Miedema matrix.

    Pure elements give 0; empty compositions or elements missing from the matrix give NaN.
    """
    fractions = get_atomic_fraction_array(stoich_array)
    present = fractions.notna()
    covered = fractions.columns.isin(miedema.index)
    weights = fractions.fillna(0.0).to_numpy()
    enthalpy = miedema.reindex(index=fractions.columns, columns=fractions.columns).fillna(0.0).to_numpy()
    np.fill_diagonal(enthalpy, 0.0)
    values = 2.0 * np.einsum("ia,ab,ib->i", weights, enthalpy, weights)  # ordered pairs, so 2 rather than 4
    n_present = present.sum(axis=1)
    invalid = (n_present == 0) | ((n_present > 1) & (present & ~covered).any(axis=1))
    return pd.Series(np.where(invalid, np.nan, values), index=stoich_array.index)


def add_engineered_features(
    raw_data: pd.DataFrame,
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Return a copy of `raw_data` with ENGINEERED_FEATURE_COLUMNS added.

    Read formulas from `formula_column`; invalid formulas give NaN features. `periodic_table` is indexed by
    element symbol, and `miedema` is a symmetric element-pair matrix.
    """
    if formula_column not in raw_data.columns:
        raise ValueError(f"Missing column {formula_column!r} in the input data.")

    data = raw_data.copy()
    stoich_array = get_stoich_array(data, periodic_table, formula_column=formula_column)

    data["n_elements"] = get_compound_radix(data, formula_column=formula_column)
    data["mixing_entropy"] = get_mixing_entropy(stoich_array)
    data["mean_atomic_weight"] = get_weighted_atomic_weight(periodic_table, stoich_array)
    data["mean_period"] = get_weighted_period(periodic_table, stoich_array)
    data["mean_group"] = get_weighted_group(periodic_table, stoich_array)
    data["mean_melting_point"] = get_weighted_melting_point(periodic_table, stoich_array)
    data["mixing_enthalpy"] = get_weighted_mixing_enthalpy(miedema, stoich_array)
    data["mean_valence"] = get_weighted_valence(periodic_table, stoich_array)
    data["mean_electronegativity"] = get_weighted_electronegativity(periodic_table, stoich_array)

    return data
