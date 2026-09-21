import io

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import build_result_rows, summarize_scores
from kalematerials.evaluate.significance import SignificanceResult
from kalematerials.utils import reporting as r

SCORES = {
    "A": {"mse": [1.0, 2.0], "mae": [1.0, 1.0], "mre": [0.1, 0.1], "r2": [0.5, 0.7]},
    "B": {"mse": [1.0, 2.5], "mae": [1.0, 1.2], "mre": [0.1, 0.2], "r2": [0.5, 0.6]},
}


def test_logger():
    stream = io.StringIO()
    r.Logger(stream).write("x")
    r.SILENT.write("y")
    assert stream.getvalue() == "x"


def test_format_functions():
    assert r.format_mean_std(1.0, 0.5, 2) == "1.00 ± 0.50" and r.format_mean_std(np.nan, 1.0) == "nan"
    assert "MSE: 1.5000 ± 0.7071" in r.format_cross_validation_results(SCORES)
    assert "R2:  1.0000" in r.format_split_results([1.0, 2.0], {"A": np.array([1.0, 2.0])})
    assert "A vs B" in r.format_comparisons(SCORES, "A", "B", metrics=("mse",), test_train_ratio=0.5)
    assert "42 searches" in r.format_search_budget(2, 3, 7)
    text = r.format_significance(SignificanceResult("mse", "A", "B", 3, np.nan, np.nan, -0.1, note="too few"))
    assert "no p-value reported: too few" in text
    table = r.format_metric_table(pd.DataFrame({"MSE_mean": [1.0], "MSE_std": [0.1], "other": [2]}))
    assert table.columns.tolist() == ["other", "MSE"] and table["MSE"][0] == "1.0000 ± 0.1000"
    assert r.format_metric_table(pd.DataFrame()).empty


def test_print_functions(capsys):
    rows = build_result_rows(SCORES, scenario="cv", split_type="ID", split_ids=["f0", "f1"], seed=0)
    r.print_summary(summarize_scores(rows), "Title")
    r.print_summary(rows.iloc[:0], "Empty")
    r.print_compound_counts(pd.Series({2: 3, 9: 1}))
    r.print_ood_tables(pd.DataFrame({"split_id": ["E=Fe"]}), summarize_scores(rows), pd.DataFrame(), pd.DataFrame())
    r.print_uncertainty_report(pd.DataFrame({"coverage_mean": [0.9]}), alpha=0.05)
    r.print_predictions(pd.DataFrame({"predicted": range(3)}), max_rows=2)
    out = capsys.readouterr().out
    for expected in (
        "Title",
        "(no observations)",
        "3 binary compounds",
        "1 9-component",
        "Splits",
        "(empty)",
        "nominal coverage 95%",
        "1 more row(s)",
    ):
        assert expected in out
