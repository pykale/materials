"""Build prediction and measurement panels for binary alloy series."""

from typing import Any, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import pandas as pd

from kalematerials.embed.composition_descriptors import add_engineered_features
from kalematerials.interpret.reference_comparison import ComparisonPanel, panel_from_series
from kalematerials.prepdata.composition import get_atomic_fraction_array, get_stoich_array
from kalematerials.utils.persistence import align_features


class AlloySeries(NamedTuple):
    """Binary alloy formulas and reference measurements.

    `element` is the alloying element on the x-axis. `measurements` maps its atomic fraction to measured
    targets; measurement compositions may differ from `formulas`.
    """

    title: str
    element: str
    formulas: List[str]
    measurements: Dict[float, float]


def build_series_features(
    formulas: Sequence[str],
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
    formula_column: str = "chemical formula",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (features, stoich_array), with formulas and engineered descriptors in `features`."""
    frame = pd.DataFrame({formula_column: list(formulas)})
    stoich = get_stoich_array(frame, periodic_table, formula_column=formula_column)
    features = add_engineered_features(frame, periodic_table, miedema, formula_column=formula_column)
    return features, stoich


def build_series_panel(
    series: AlloySeries,
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
) -> ComparisonPanel:
    """Predict an alloy series with models keyed by display name and pair it with measurements.

    `feature_columns` follows training order; `periodic_table` is indexed by element symbol and
    `miedema` is a symmetric element-pair matrix.
    """
    features, stoich_array = build_series_features(series.formulas, periodic_table, miedema)
    X = align_features(features, feature_columns, source=f"the {series.title} case study")
    atomic_fraction = get_atomic_fraction_array(stoich_array)

    return panel_from_series(
        title=f"{series.title} Case Study",
        x_label=f"{series.element} content [atomic fraction]",
        x=atomic_fraction[series.element],
        predictions={name: model.predict(X) for name, model in models.items()},
        measurements=series.measurements,
    )


def build_series_panels(
    all_series: Sequence[AlloySeries],
    feature_columns: Sequence[str],
    models: Mapping[str, Any],
    periodic_table: pd.DataFrame,
    miedema: pd.DataFrame,
) -> List[ComparisonPanel]:
    """Build one prediction-and-measurement panel per alloy series."""
    if not models:
        raise ValueError("A case study needs at least one fitted model.")
    return [build_series_panel(series, feature_columns, models, periodic_table, miedema) for series in all_series]
