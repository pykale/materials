"""Formula parsing, stoichiometry and atomic-fraction descriptors."""

import re
import warnings
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition

# --- Formula parsing ---


def get_composition_key(formula: str) -> Optional[str]:
    """Return a canonical Hill formula, or None if invalid. FeNi, NiFe and Fe2Ni2 all become "FeNi"."""
    if pd.isna(formula):
        return None
    try:
        composition = Composition(str(formula), strict=True)
        amounts = composition.get_el_amt_dict()
        if not amounts or any(not np.isfinite(amount) or amount <= 0 for amount in amounts.values()):
            raise ValueError("Element amounts must be finite and positive")
        total = sum(amounts.values())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("Total element amount must be finite and positive")
        integer_formula, _ = composition.get_integer_formula_and_factor()
        return Composition(integer_formula).hill_formula.replace(" ", "")
    except (ValueError, TypeError) as exc:
        warnings.warn(f"Could not normalize chemical formula {formula!r}; dropping the row ({exc}).")
        return None


def get_elements(formula: str) -> List[str]:
    """Return element symbols in formula order, or [] for a missing or invalid formula."""
    if pd.isna(formula):
        return []
    try:
        return [str(element) for element in Composition(str(formula)).elements]
    except Exception as exc:
        warnings.warn(f"Could not parse chemical formula {formula!r}; treating it as containing no elements ({exc}).")
        return []


def get_elements_per_row(x, formula_column: str = "chemical formula") -> List[List[str]]:
    """Return element lists from a formula Series or a DataFrame with `formula_column`."""
    if isinstance(x, pd.DataFrame):
        if formula_column not in x.columns:
            raise ValueError(f"Missing column: {formula_column}")
        formulas = x[formula_column].tolist()
    else:
        formulas = list(x)
    elements_per_row = [get_elements(formula) for formula in formulas]

    n_unparsed = sum(
        1 for formula, elements in zip(formulas, elements_per_row) if not elements and not pd.isna(formula)
    )
    if n_unparsed:
        warnings.warn(f"{n_unparsed} of {len(formulas)} formulas in '{formula_column}' could not be parsed.")

    return elements_per_row


def formula_contains_elements(
    df: pd.DataFrame,
    elements: Iterable[str],
    formula_column: str = "chemical formula",
) -> np.ndarray:
    """Boolean mask over `df` rows whose formula contains any of `elements`."""
    wanted = set(elements)
    return np.array(
        [bool(wanted.intersection(row)) for row in get_elements_per_row(df, formula_column=formula_column)],
        dtype=bool,
    )


# --- Periodic table ---


def get_group_period_maps(
    periodic_table: pd.DataFrame,
    element_col: str = "symbol",
    period_col: str = "period",
    group_block_col: str = "group_block",
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (element_to_group, element_to_period) maps.

    Read group numbers from labels such as "group 1, s-block"; omit elements without a group number.
    """
    for column in (element_col, period_col, group_block_col):
        if column not in periodic_table.columns:
            raise ValueError(f"Missing column '{column}' in periodic table file. Found: {list(periodic_table.columns)}")

    def group_number(group_block):
        match = re.search(r"group\s*(\d+)", str(group_block), flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    symbols = periodic_table[element_col].astype(str)
    element_to_period = dict(zip(symbols, periodic_table[period_col].astype(int)))
    groups = (group_number(group_block) for group_block in periodic_table[group_block_col])
    element_to_group = {symbol: group for symbol, group in zip(symbols, groups) if group is not None}
    return element_to_group, element_to_period


# --- Stoichiometry ---


def get_stoich_array(
    x: Union[pd.DataFrame, str],
    periodic_table: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Return element amounts: one row per formula, one column per periodic-table symbol, zero where absent.

    Accept a formula string or a DataFrame with `formula_column`. Unparseable formulas produce all-zero rows.
    """
    if isinstance(x, pd.DataFrame):
        formulas, index = x[formula_column], x.index
    else:
        formulas = pd.Series(x)
        index = formulas.index

    # Preserve longest-symbol-first order for reproducible floating-point sums.
    symbols = periodic_table["symbol"].astype(str)
    symbols = symbols.reindex(symbols.str.len().sort_values(ascending=False).index)

    rows = []
    for formula in formulas:
        amounts = {}
        if not pd.isna(formula):
            try:
                amounts = Composition(str(formula)).get_el_amt_dict()
            except Exception as exc:
                warnings.warn(f"Could not parse chemical formula {formula!r}; dropping the row ({exc}).")
            unknown = [element for element in amounts if element not in symbols.values]
            if unknown:
                warnings.warn(f"Formula {formula!r}: element(s) {unknown} are not in the periodic table; dropped.")
        rows.append(amounts)

    return pd.DataFrame(rows, index=index, columns=symbols, dtype=float).fillna(0.0)


# --- Atomic fractions ---


def get_atomic_fraction(compound: pd.Series) -> pd.Series:
    """Atomic fractions of the elements present in one stoichiometry row; empty if the row has none."""
    present = compound[compound != 0]
    if present.empty or present.sum() == 0:
        return pd.Series(dtype=float)
    return present / present.sum()


def get_atomic_fraction_array(stoich_array: pd.DataFrame) -> pd.DataFrame:
    """Atomic fractions with the shape of `stoich_array`; NaN where an element is absent."""
    present = stoich_array != 0
    return stoich_array.div(stoich_array.sum(axis=1), axis=0).where(present)


# --- Descriptor primitives ---


def get_weighted_property(values: pd.Series, stoich_array: pd.DataFrame) -> pd.Series:
    """Return the atomic-fraction-weighted mean per composition.

    `values` is indexed by element symbol. Empty compositions or missing element properties give NaN.
    """
    fractions = get_atomic_fraction_array(stoich_array)
    weights = fractions.to_numpy(dtype=float)
    props = values.reindex(fractions.columns).to_numpy(dtype=float)
    present = ~np.isnan(weights)
    weighted = np.where(present, weights * props, 0.0).sum(axis=1)
    weighted[~present.any(axis=1)] = np.nan
    return pd.Series(weighted, index=stoich_array.index)


def get_mixing_entropy(stoich_array: pd.DataFrame) -> pd.Series:
    """Ideal mixing entropy, -sum(f ln f) over each composition's atomic fractions."""
    fractions = get_atomic_fraction_array(stoich_array)
    return -(fractions * np.log(fractions)).sum(axis=1, min_count=1)


def get_compound_radix(x: Union[pd.DataFrame, str], formula_column: str = "chemical formula") -> pd.Series:
    """Number of distinct elements per formula, NaN where the formula could not be parsed."""
    formulas = x[formula_column] if isinstance(x, pd.DataFrame) else pd.Series(x)
    return formulas.apply(lambda formula: len(get_elements(formula)) or np.nan)
