import pytest

from kalematerials.interpret.reference_comparison import (
    ComparisonPanel,
    panel_from_series,
    plot_predictions_against_measurements,
)


def panel(title="FeCo", predictions=None):
    return panel_from_series(
        title, "x", [0.0, 0.5], predictions if predictions is not None else {"rf": [1.0, 2.0]}, {0.1: 1.1}
    )


def test_panel_from_series():
    p = panel()
    assert isinstance(p, ComparisonPanel) and p.measured_x == [0.1] and p.measured_y == [1.1]


def test_plot(tmp_path):
    path = tmp_path / "fig.png"
    plot_predictions_against_measurements(
        [panel(), panel("FeAl"), panel("FeCr"), panel("FeNi")], y_label="Ms", max_columns=3, save_path=path
    )
    assert path.stat().st_size > 0
    for bad in ([], [panel(predictions={})]):
        with pytest.raises(ValueError):
            plot_predictions_against_measurements(bad, y_label="Ms")
