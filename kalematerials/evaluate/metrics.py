"""Regression metrics and the long-form result table: one row per split, model and metric."""

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

from kalematerials.loaddata.splits import ID_KFOLD, Split

METRICS = ("mse", "mae", "mre", "r2")
METRIC_DECIMALS = {"mse": 4, "mae": 4, "mre": 6, "r2": 4}

# Paired tests use lower-is-better metrics.
HIGHER_IS_BETTER = ("r2",)
COMPARABLE_METRICS = tuple(metric for metric in METRICS if metric not in HIGHER_IS_BETTER)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """MSE, MAE, MRE (mean absolute percentage error as a fraction) and R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mre": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# {model: {metric: [score per split]}}
FoldScores = Dict[str, Dict[str, List[float]]]

# `scenario` names the split family; `split_type` uses loaddata.splits constants; `split_id` identifies a split.
RESULT_COLUMNS = ("scenario", "split_type", "split_id", "seed", "model", "metric", "value")

SUMMARY_GROUPS = ("scenario", "split_type", "model", "metric")


def wide_table_to_results(
    table: pd.DataFrame,
    value_columns: Dict[str, str],
    *,
    identifiers: Dict[str, str],
) -> pd.DataFrame:
    """Reshape per-split metric columns into RESULT_COLUMNS.

    `value_columns` maps input columns to metric names. `identifiers` maps output identifiers to input
    columns, or to constant values when no matching column exists.
    """
    if table.empty:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))

    frame = pd.DataFrame(index=table.index)
    for target, source in identifiers.items():
        frame[target] = table[source] if source in table.columns else source

    melted = []
    for column, metric in value_columns.items():
        if column not in table.columns:
            continue
        part = frame.copy()
        part["metric"] = metric
        part["value"] = pd.to_numeric(table[column], errors="coerce")
        melted.append(part)

    if not melted:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.concat(melted, ignore_index=True)[list(RESULT_COLUMNS)]


def build_result_rows(
    scores: Dict[str, Dict[str, Sequence[float]]],
    *,
    scenario: str,
    split_type: str,
    split_ids: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Convert {model: {metric: [scores]}} to RESULT_COLUMNS rows.

    Each score list follows `split_ids` and has the same length. `scenario` names the split family;
    `split_type` uses the constants in loaddata.splits.
    """
    rows = []
    for model, per_metric in scores.items():
        for metric, values in per_metric.items():
            if len(values) != len(split_ids):
                raise ValueError(
                    f"{model}/{metric} has {len(values)} value(s) but {len(split_ids)} split id(s) were given."
                )
            for split_id, value in zip(split_ids, values):
                rows.append((scenario, split_type, split_id, seed, model, metric, float(value)))
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def results_to_scores(results: pd.DataFrame) -> Dict[str, Dict[str, List[float]]]:
    """Return {model: {metric: [values in table order]}} from result rows.

    Use one scenario and seed, with matching split order across models.
    """
    return {
        model: {
            metric: [float(value) for value in per_metric["value"]]
            for metric, per_metric in per_model.groupby("metric", sort=False)
        }
        for model, per_model in results.groupby("model", sort=False)
    }


def best_model(summary: pd.DataFrame, metric: str = "mse") -> str:
    """Return the model with the best mean `metric`; raise ValueError if the metric has no rows."""
    rows = summary[summary["metric"] == metric]
    if rows.empty:
        raise ValueError(f"No {metric!r} rows in the summary; metrics present: {sorted(summary['metric'].unique())}")
    ranked = rows.sort_values("mean", ascending=metric not in HIGHER_IS_BETTER)
    return str(ranked.iloc[0]["model"])


def scores_to_results(
    scores: FoldScores,
    splits: Sequence[Split],
    *,
    scenario: str,
    split_type: str = ID_KFOLD,
    seed: int = 0,
) -> pd.DataFrame:
    """Convert scores to result rows using the IDs in `splits`."""
    return build_result_rows(
        scores,
        scenario=scenario,
        split_type=split_type,
        split_ids=[split_id for split_id, _, _ in splits],
        seed=seed,
    )


def summarize_scores(results: pd.DataFrame, by: Sequence[str] = SUMMARY_GROUPS) -> pd.DataFrame:
    """Mean, sample std (0.0 for a single observation) and count of `value` per group of `by`."""
    if results.empty:
        return pd.DataFrame(columns=[*by, "mean", "std", "n"])

    grouped = results.groupby(list(by), sort=True)["value"]
    summary = grouped.agg(mean="mean", std=lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n="size")
    return summary.reset_index()
