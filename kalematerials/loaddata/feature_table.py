"""Standardize records and load prepared feature tables."""

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdtypes

DEFAULT_TARGET_COLUMN = "saturation magnetization"
DEFAULT_FORMULA_COLUMN = "chemical formula"


def standardize_records(
    data: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Copy records with trimmed formulas, numeric targets and optional nullable `is_magnetic` booleans.

    Invalid targets become NaN. Missing required columns or unknown magnetic labels raise ValueError.
    """
    required = [formula_column, target_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} in the loaded data. Available: {sorted(data.columns)}")
    data = data[required + [column for column in data.columns if column not in required]].copy()
    data[formula_column] = data[formula_column].astype("string").str.strip().replace("", pd.NA)
    data[target_column] = pd.to_numeric(data[target_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "is_magnetic" in data.columns:
        labels = data["is_magnetic"].astype("string").str.strip().str.lower()
        labels = labels.replace({"": pd.NA, "none": pd.NA, "nan": pd.NA})
        mapping = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
        invalid = labels.notna() & ~labels.isin(mapping)
        if invalid.any():
            raise ValueError(f"Unknown is_magnetic label(s): {labels[invalid].unique().tolist()}")
        data["is_magnetic"] = labels.map(mapping).astype("boolean")
    return data.reset_index(drop=True)


def resolve_feature_columns(data: pd.DataFrame, feature_columns: Sequence[str]) -> List[str]:
    """Return a nonempty list of existing numeric feature columns, in the requested order."""
    resolved = list(feature_columns)
    if not resolved:
        raise ValueError("feature_columns is empty.")
    missing = [c for c in resolved if c not in data.columns]
    if missing:
        raise ValueError(f"feature_columns {missing} are not in the data. Available: {sorted(data.columns)}")
    non_numeric = [c for c in resolved if not pdtypes.is_numeric_dtype(data[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature column(s): {non_numeric}")
    return resolved


def load_feature_table(
    path,
    feature_columns: Sequence[str],
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read a CSV as row-aligned (X, y, metadata).

    X follows `feature_columns` order; metadata contains all columns outside the features and target.
    """
    data = pd.read_csv(path).reset_index(drop=True)
    if target_column not in data.columns:
        raise ValueError(f"Missing target column {target_column!r} in {path}. Available: {sorted(data.columns)}")

    resolved = resolve_feature_columns(data, feature_columns)
    metadata = [column for column in data.columns if column not in resolved and column != target_column]
    return data[resolved].copy(), data[target_column].copy(), data[metadata].copy()
