"""Build feature tables for main.py; rerun after changes to raw data, features or selection."""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from kalematerials.loaddata.element_tables import load_element_properties
from kalematerials.loaddata.materials_project import load_materials_project
from kalematerials.loaddata.novamag import load_novamag
from kalematerials.pipeline.feature_table import build_feature_table, NON_COMMERCIAL_ELEMENTS, RARE_EARTH_ELEMENTS


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Prepare composition features from local magnetic-material datasets")
    parser.add_argument("--dataset", choices=("all", "novamag", "materials_project"), default="all")
    parser.add_argument(
        "--novamag-dir",
        default="./data/novamag/Novamag_Data_Files/",
        help="Directory containing Novamag JSON files",
    )
    parser.add_argument(
        "--materials-project-csv",
        default="./data/materials_project/mp-data.csv",
        help="Materials Project CSV export",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--min-target",
        type=float,
        default=0.18,
        help="Inclusive record threshold in tesla, before median",
    )
    parser.add_argument(
        "--exclude-elements",
        nargs="*",
        default=None,
        metavar="ELEMENT",
        help="Override element exclusions for selected datasets; no values disables exclusions. "
        "Default: original rare-earth/actinide exclusions for MP, none for Novamag.",
    )
    parser.add_argument(
        "--materials-project-include-nonmagnetic",
        action="store_true",
        help="Disable MP's is_magnetic label filter; the target threshold still applies",
    )
    return parser.parse_args()


def _save_feature_table(
    data: pd.DataFrame,
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    output_path: Path,
    *,
    min_target: float,
    excluded_elements: Optional[Iterable[str]],
    magnetic_only: bool = False,
) -> None:
    """Build one feature table and write it to `output_path`."""
    table, _ = build_feature_table(
        data,
        periodic_table,
        miedema,
        min_target=min_target,
        excluded_elements=excluded_elements,
        magnetic_only=magnetic_only,
    )
    print(f"  {len(data)} loaded records -> {len(table)} compositions -> {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=True)


def main() -> None:
    """Build the selected feature tables."""
    args = parse_args()
    periodic_table, miedema = load_element_properties()

    if args.dataset in ("all", "novamag"):
        print("Processing Novamag dataset...")
        novamag = load_novamag(args.novamag_dir)
        records_path = args.output_dir / "novamag" / "novamag-raw.csv"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        novamag.to_csv(records_path, index=False)
        _save_feature_table(
            novamag,
            periodic_table,
            miedema,
            args.output_dir / "novamag-magnetism.csv",
            min_target=args.min_target,
            excluded_elements=args.exclude_elements,
        )

    if args.dataset in ("all", "materials_project"):
        print("Processing Materials Project dataset...")
        excluded = args.exclude_elements
        if excluded is None:
            excluded = RARE_EARTH_ELEMENTS + NON_COMMERCIAL_ELEMENTS
        _save_feature_table(
            load_materials_project(args.materials_project_csv),
            periodic_table,
            miedema,
            args.output_dir / "mp-magnetism.csv",
            min_target=args.min_target,
            excluded_elements=excluded,
            magnetic_only=not args.materials_project_include_nonmagnetic,
        )


if __name__ == "__main__":
    main()
