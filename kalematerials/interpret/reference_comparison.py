"""Plot predicted series against reference measurements, one panel per series."""

import math
from typing import Mapping, NamedTuple, Sequence, Tuple

import matplotlib.pyplot as plt
import seaborn as sns


class ComparisonPanel(NamedTuple):
    """Predictions and reference measurements for one panel.

    `predictions` maps legend names to values aligned with `x`. Reference points use `measured_x`
    and `measured_y`; `x_label` includes units.
    """

    title: str
    x_label: str
    x: Sequence[float]
    predictions: Mapping[str, Sequence[float]]
    measured_x: Sequence[float]
    measured_y: Sequence[float]


def plot_predictions_against_measurements(
    panels: Sequence[ComparisonPanel],
    *,
    y_label: str,
    measurement_label: str = "literature",
    max_columns: int = 3,
    panel_size: Tuple[float, float] = (6.7, 4.0),
    save_path=None,
) -> None:
    """Overlay predictions and measurements in a panel grid.

    `panel_size` is (width, height) in inches. Save to `save_path` when set; otherwise show the figure.
    """
    if not panels:
        raise ValueError("plot_predictions_against_measurements needs at least one panel.")

    empty = [panel.title for panel in panels if not panel.predictions]
    if empty:
        raise ValueError(f"No predicted series for panel(s): {empty}.")

    columns = min(len(panels), max(1, max_columns))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_size[0] * columns, panel_size[1] * rows),
        squeeze=False,
    )
    flat = [ax for row in axes for ax in row]

    for ax, panel in zip(flat, panels):
        _plot_one_panel(ax, panel, y_label=y_label, measurement_label=measurement_label)
    for ax in flat[len(panels) :]:
        fig.delaxes(ax)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def _plot_one_panel(ax, panel: ComparisonPanel, *, y_label: str, measurement_label: str) -> None:
    """Draw one panel's predicted series and reference points onto `ax`."""
    for values in panel.predictions.values():
        sns.scatterplot(x=panel.x, y=values, ax=ax)
    sns.scatterplot(x=panel.measured_x, y=panel.measured_y, ax=ax)

    ax.set_title(panel.title, fontsize=16)
    ax.set_xlabel(panel.x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    legend = ax.legend([*panel.predictions, measurement_label], loc="upper right", fontsize=12)
    legend.get_frame().set_facecolor("white")


def panel_from_series(
    title: str,
    x_label: str,
    x: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    measurements: Mapping[float, float],
) -> ComparisonPanel:
    """Build a panel from measurements given as {x: measured value}."""
    return ComparisonPanel(
        title=title,
        x_label=x_label,
        x=x,
        predictions=predictions,
        measured_x=list(measurements.keys()),
        measured_y=list(measurements.values()),
    )
