"""Load Novamag JSON records, one per computed structure."""

import os
import warnings

import pandas as pd

from kalematerials.loaddata.feature_table import DEFAULT_FORMULA_COLUMN, DEFAULT_TARGET_COLUMN, standardize_records


def _flatten(x):
    """Unwrap Novamag's nested {'value': ...} quantities."""
    if isinstance(x, dict) and "value" in x and len(x) == 1:
        return x["value"]
    return x


def read_novamag_records(root_dir: str) -> pd.DataFrame:
    """Read Novamag JSON files recursively, skipping unreadable files with a warning.

    Return chemistry, crystal and magnetics fields plus `sample_id` and `source`, one row per record.
    Raise FileNotFoundError for a missing directory or ValueError if no records can be read.
    """
    rows, failed = [], []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Novamag directory does not exist: {root_dir}")
    for dir_name, sub_dirs, file_names in os.walk(root_dir):
        sub_dirs.sort()
        for file_name in sorted(file_names):
            if not file_name.endswith(".json"):
                continue
            path = os.path.join(dir_name, file_name)
            try:
                record = pd.read_json(path, encoding="Latin")
                rows.append(
                    {
                        **record.properties.chemistry,
                        **record.properties.crystal,
                        **record.properties.magnetics,
                        "sample_id": os.path.relpath(path, root_dir),
                        "source": "novamag",
                    }
                )
            except ValueError:
                failed.append(path)

    if failed:
        warnings.warn(f"Skipped {len(failed)} unreadable Novamag file(s), the first being {failed[0]}.")

    if not rows:
        raise ValueError(f"No readable Novamag records found in {root_dir}")
    data = pd.DataFrame(rows).apply(lambda column: column.map(_flatten))
    data = data.mask(data.isna() | data.isin(["none", "None"]))
    return data.infer_objects(copy=False)


def load_novamag(
    root_dir: str,
    target_column: str = DEFAULT_TARGET_COLUMN,
    formula_column: str = DEFAULT_FORMULA_COLUMN,
) -> pd.DataFrame:
    """Load standardized Novamag records with saturation magnetization in tesla."""
    data = read_novamag_records(root_dir).rename(
        columns={
            DEFAULT_FORMULA_COLUMN: formula_column,
            DEFAULT_TARGET_COLUMN: target_column,
        }
    )
    return standardize_records(data, target_column, formula_column)
