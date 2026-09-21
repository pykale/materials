import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import build_result_rows
from kalematerials.evaluate.ood_summaries import summarize_generalization_gap, summarize_model_comparison
from kalematerials.loaddata.splits import ID_RANDOM, ID_REFERENCE, OOD


def rows(split_type, a, b):
    return build_result_rows(
        {"A": {"mse": a}, "B": {"mse": b}},
        scenario="LOEO",
        split_type=split_type,
        split_ids=[f"E={i}" for i in range(len(a))],
        seed=0,
    )


def test_model_comparison():
    results = pd.concat(
        [rows(OOD, [1, 2, 3, 4, 5, 6.5], [1.4, 2.5, 3.6, 4.5, 5.7, 7]), rows(ID_REFERENCE, [1] * 6, [1] * 6)]
    )
    table = summarize_model_comparison(results, "A", "B")
    assert table.iloc[0][["scenario", "metric", "n_splits"]].tolist() == ["LOEO", "MSE", 6]
    assert table.iloc[0]["mean_difference"] < 0 and bool(table.iloc[0]["significant"])
    assert summarize_model_comparison(results, "A", "C").empty


def test_generalization_gap():
    summary = pd.DataFrame(
        {
            "scenario": ["LOEO"] * 3,
            "split_type": [OOD, ID_REFERENCE, ID_RANDOM],
            "model": ["A"] * 3,
            "metric": ["mse"] * 3,
            "mean": [3.0, 2.0, 1.5],
        }
    )
    gap = summarize_generalization_gap(summary)
    assert gap.iloc[0][["shift_gap", "train_pool_gap"]].tolist() == [1.0, 0.5]
    assert np.isnan(summarize_generalization_gap(summary.iloc[:1]).iloc[0]["shift_gap"])
    assert summarize_generalization_gap(summary, "mae").empty
