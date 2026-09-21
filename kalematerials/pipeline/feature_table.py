"""Select records, merge equivalent compositions and compute features."""

import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from kalematerials.embed.composition_descriptors import add_engineered_features, ENGINEERED_FEATURE_COLUMNS
from kalematerials.prepdata.composition import formula_contains_elements, get_composition_key

# Optional element exclusions.
RARE_EARTH_ELEMENTS = (
    "La",
    "Sc",
    "Dy",
    "Sm",
    "Lu",
    "Er",
    "Y",
    "Pr",
    "Nd",
    "Gd",
    "Tm",
    "Pm",
    "Ce",
    "Tb",
    "Eu",
    "Ho",
    "Yb",
)
NON_COMMERCIAL_ELEMENTS = (
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
)

# Source metadata, excluded from model inputs.
PROVENANCE_COLUMNS = ("n_records", "sample_id")


def _describe_sources(data: pd.DataFrame, targets: pd.Series, target_column: str) -> pd.DataFrame:
    """Count source rows per composition and record the sample_id matching the median target, if any."""
    grouped = data.groupby("composition_key", sort=True)
    provenance = pd.DataFrame({"n_records": grouped.size()})
    provenance.index.name = targets.index.name

    if "sample_id" in data.columns:
        median_of = data["composition_key"].map(targets)
        exact = data.loc[data[target_column] == median_of, ["composition_key", "sample_id"]]
        first = exact.groupby("composition_key")["sample_id"].first()
        provenance["sample_id"] = first.reindex(provenance.index)

    return provenance


def filter_samples(
    data: pd.DataFrame,
    min_target: Optional[float] = 0.18,
    excluded_elements: Optional[Iterable[str]] = None,
    magnetic_only: bool = False,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
) -> pd.DataFrame:
    """Select records with a formula and a finite target.

    `min_target` is inclusive; None disables the bound. Drop formulas containing `excluded_elements`.
    With `magnetic_only`, keep rows whose boolean `is_magnetic` is True.
    """
    required = [formula_column, target_column] + (["is_magnetic"] if magnetic_only else [])
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) for sample selection: {missing}")
    selected = data.copy()
    selected[target_column] = pd.to_numeric(selected[target_column], errors="coerce")
    selected = selected.loc[np.isfinite(selected[target_column]) & selected[formula_column].notna()].copy()
    if min_target is not None:
        selected = selected.loc[selected[target_column] >= min_target]
    if magnetic_only:
        if not pd.api.types.is_bool_dtype(selected["is_magnetic"]):
            raise ValueError("is_magnetic must contain standardized boolean values")
        selected = selected.loc[selected["is_magnetic"].fillna(False)]
    excluded = tuple(excluded_elements) if excluded_elements is not None else ()
    if excluded:
        selected = selected.loc[~formula_contains_elements(selected, excluded, formula_column=formula_column)]
    return selected.reset_index(drop=True)


def build_feature_table(
    raw_data: pd.DataFrame,
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    min_target: Optional[float] = 0.18,
    target_column: str = "saturation magnetization",
    formula_column: str = "chemical formula",
    *,
    excluded_elements: Optional[Iterable[str]] = None,
    magnetic_only: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build one sample per composition using the median target and engineered features.

    Equivalent formulas such as FeNi and Fe2Ni2 share a composition key. Selection follows `filter_samples`.
    Return (table, feature_columns), with the table indexed by composition key and containing the target,
    features and PROVENANCE_COLUMNS. Warn and drop rows with incomplete features.
    """
    data = filter_samples(
        raw_data,
        min_target=min_target,
        excluded_elements=excluded_elements,
        magnetic_only=magnetic_only,
        target_column=target_column,
        formula_column=formula_column,
    )
    keys = {formula: get_composition_key(formula) for formula in data[formula_column].unique()}
    data["composition_key"] = data[formula_column].map(keys)
    data = data.dropna(subset=["composition_key"])
    grouped = data.groupby("composition_key", sort=True)
    targets = grouped[target_column].median()
    targets.index.name = formula_column
    provenance = _describe_sources(data, targets, target_column)
    data = add_engineered_features(targets.reset_index(), periodic_table, miedema, formula_column=formula_column)
    data = data.join(provenance, on=formula_column)

    feature_columns = list(ENGINEERED_FEATURE_COLUMNS)
    data = data.replace([np.inf, -np.inf], np.nan)
    incomplete = data[feature_columns].isna().any(axis=1)
    if incomplete.any():
        unusable = sorted(data.loc[incomplete, feature_columns].isna().any().loc[lambda s: s].index)
        warnings.warn(f"Dropped {int(incomplete.sum())} of {len(data)} compositions with NaN in {unusable}.")
    kept = data.loc[~incomplete].set_index(formula_column)
    carried = [column for column in PROVENANCE_COLUMNS if column in kept.columns]
    return kept[[target_column] + feature_columns + carried], feature_columns
