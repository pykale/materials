"""Build the datasets main.py reads, and show what they contain.

Rerun after changes to raw data, features or selection.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from kalematerials.interpret.dataset_distributions import (
    count_compounds_by_radix,
    plot_target_distribution_by_element,
    plot_target_violin_by_element,
)
from kalematerials.loaddata.element_tables import load_element_properties
from kalematerials.loaddata.materials_project import load_materials_project
from kalematerials.loaddata.novamag import load_novamag
from kalematerials.pipeline.feature_table import (
    build_feature_table,
    NON_COMMERCIAL_ELEMENTS,
    RARE_EARTH_ELEMENTS,
)
from kalematerials.utils.reporting import print_compound_counts

TARGET_COLUMN = "saturation magnetization"
FORMULA_COLUMN = "chemical formula"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Build composition datasets from local magnetic-material sources")
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
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"), help="Where to write the figures")
    parser.add_argument("--no-plots", action="store_true", help="Build the tables without describing them")
    parser.add_argument(
        "--distribution-elements",
        nargs="*",
        default=["Fe", "Co", "Cr", "Mn"],
        metavar="ELEMENT",
        help="Elements to overlay on the distribution figures",
    )
    parser.add_argument(
        "--bins",
        nargs=3,
        type=float,
        default=(0.0, 2.6, 0.2),
        metavar=("MIN", "MAX", "STEP"),
        help="Histogram bin edges in tesla",
    )
    return parser.parse_args()


def _build_table(
    data: pd.DataFrame,
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    output_path: Path,
    *,
    min_target: float,
    excluded_elements: Optional[Iterable[str]],
    magnetic_only: bool = False,
) -> pd.DataFrame:
    """Build one feature table, write it to `output_path` and return it."""
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
    return table


def _describe_table(
    table: pd.DataFrame,
    name: str,
    plots_dir: Path,
    *,
    elements: Sequence[str],
    bins: Sequence[float],
) -> None:
    """Save the target-distribution figures for one table and print its compound counts."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    data = table.reset_index()

    plot_target_distribution_by_element(
        data,
        target_column=TARGET_COLUMN,
        elements=elements,
        formula_column=FORMULA_COLUMN,
        bins=np.arange(*bins),
        save_path=plots_dir / f"{name}_ms_distribution_by_tm.png",
    )
    plot_target_violin_by_element(
        data,
        target_column=TARGET_COLUMN,
        elements=elements,
        formula_column=FORMULA_COLUMN,
        target_label="Saturation Magnetization (T)",
        title=f"{name.upper()} Violin Plot",
        save_path=plots_dir / f"{name}_violin_ms_by_tm.png",
    )
    print_compound_counts(count_compounds_by_radix(data, formula_column=FORMULA_COLUMN))


def main() -> None:
    """Build the selected datasets and describe them."""
    args = parse_args()
    periodic_table, miedema = load_element_properties()

    if args.dataset in ("all", "novamag"):
        print("Processing Novamag dataset...")
        novamag = load_novamag(args.novamag_dir)
        records_path = args.output_dir / "novamag" / "novamag-raw.csv"
        records_path.parent.mkdir(parents=True, exist_ok=True)
        novamag.to_csv(records_path, index=False)
        table = _build_table(
            novamag,
            periodic_table,
            miedema,
            args.output_dir / "novamag-magnetism.csv",
            min_target=args.min_target,
            excluded_elements=args.exclude_elements,
        )
        if not args.no_plots:
            _describe_table(table, "novamag", args.plots_dir, elements=args.distribution_elements, bins=args.bins)

    if args.dataset in ("all", "materials_project"):
        print("Processing Materials Project dataset...")
        excluded = args.exclude_elements
        if excluded is None:
            excluded = RARE_EARTH_ELEMENTS + NON_COMMERCIAL_ELEMENTS
        table = _build_table(
            load_materials_project(args.materials_project_csv),
            periodic_table,
            miedema,
            args.output_dir / "mp-magnetism.csv",
            min_target=args.min_target,
            excluded_elements=excluded,
            magnetic_only=not args.materials_project_include_nonmagnetic,
        )
        if not args.no_plots:
            _describe_table(
                table, "materials_project", args.plots_dir, elements=args.distribution_elements, bins=args.bins
            )


if __name__ == "__main__":
    main()
