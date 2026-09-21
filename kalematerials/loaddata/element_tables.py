"""Packaged periodic table and Miedema mixing-enthalpy matrix."""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

RESOURCES_DIR = Path(__file__).resolve().parent.parent / "resources"
DEFAULT_PERIODIC_TABLE_PATH = RESOURCES_DIR / "periodic_table.xlsx"
DEFAULT_MIEDEMA_PATH = RESOURCES_DIR / "miedema_model.xlsx"


def load_periodic_table(path: Optional[str] = None) -> pd.DataFrame:
    """Read a periodic table indexed by element symbol; None uses the packaged table."""
    periodic_table = pd.read_excel(path or DEFAULT_PERIODIC_TABLE_PATH)
    periodic_table.index = periodic_table["symbol"]
    return periodic_table


def load_miedema(path: Optional[str] = None) -> pd.DataFrame:
    """Read a symmetric Miedema element-pair matrix; None uses the packaged table."""
    miedema = pd.read_excel(
        path or DEFAULT_MIEDEMA_PATH,
        header=1,
        index_col=73,
        usecols=range(0, 74),
        nrows=73,
    )
    # Replace element-symbol labels on the diagonal with zeros.
    miedema = miedema.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return miedema + miedema.transpose()


def load_element_properties(
    periodic_table_path: Optional[str] = None,
    miedema_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (periodic_table, miedema), using packaged tables for omitted paths."""
    return load_periodic_table(periodic_table_path), load_miedema(miedema_path)
