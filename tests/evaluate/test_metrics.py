import pandas as pd
import pytest

from kalematerials.evaluate import metrics as m

SCORES = {"A": {"mse": [1.0, 2.0], "r2": [0.5, 0.6]}, "B": {"mse": [3.0, 4.0], "r2": [0.1, 0.2]}}


def test_compute_metrics():
    assert m.compute_metrics([1, 2, 4], [1, 2, 4]) == {"mse": 0, "mae": 0, "mre": 0, "r2": 1}
    assert m.compute_metrics([1, 2], [2, 3])["mre"] == pytest.approx(0.75)


def test_result_rows_roundtrip():
    rows = m.build_result_rows(SCORES, scenario="cv", split_type="ID", split_ids=["f0", "f1"], seed=0)
    assert list(rows.columns) == list(m.RESULT_COLUMNS) and len(rows) == 8
    assert m.results_to_scores(rows) == SCORES
    with pytest.raises(ValueError):
        m.build_result_rows(SCORES, scenario="cv", split_type="ID", split_ids=["f0"], seed=0)
    splits = [("f0", None, None), ("f1", None, None)]
    assert m.scores_to_results(SCORES, splits, scenario="cv", split_type="ID").equals(rows)


def test_summary_and_best_model():
    rows = m.build_result_rows(SCORES, scenario="cv", split_type="ID", split_ids=["f0", "f1"], seed=0)
    summary = m.summarize_scores(rows)
    assert list(summary.columns) == [*m.SUMMARY_GROUPS, "mean", "std", "n"]
    assert summary.set_index(["model", "metric"]).loc[("A", "mse"), ["mean", "n"]].tolist() == [1.5, 2]
    assert m.summarize_scores(rows.iloc[:1])["std"].tolist() == [0.0]
    assert m.summarize_scores(rows.iloc[:0]).empty
    assert m.best_model(summary) == "A" and m.best_model(summary, "r2") == "A"
    with pytest.raises(ValueError):
        m.best_model(summary, "mae")


def test_wide_table_to_results():
    wide = pd.DataFrame({"split_id": ["s0"], "seed": [0], "cov": [0.9], "method": ["conformal"]})
    long = m.wide_table_to_results(
        wide,
        {"cov": "coverage", "absent": "x"},
        identifiers={
            "scenario": "LOEO",
            "split_type": "OOD",
            "split_id": "split_id",
            "seed": "seed",
            "model": "method",
        },
    )
    assert long.iloc[0].tolist() == ["LOEO", "OOD", "s0", 0, "conformal", "coverage", 0.9]
    assert m.wide_table_to_results(wide.iloc[:0], {"cov": "coverage"}, identifiers={}).empty
    assert m.wide_table_to_results(wide, {"none": "x"}, identifiers={}).empty
