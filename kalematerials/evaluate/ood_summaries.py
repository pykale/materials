"""Model comparisons and generalization gaps for out-of-distribution results."""

from typing import Sequence

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import COMPARABLE_METRICS
from kalematerials.evaluate.significance import compare_models_significance, MIN_PAIRS_FOR_TEST
from kalematerials.loaddata.splits import ID_RANDOM, ID_REFERENCE, OOD


def summarize_model_comparison(
    results: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    split_type: str = OOD,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    min_pairs: int = MIN_PAIRS_FOR_TEST,
) -> pd.DataFrame:
    """Compare two models per scenario and metric, averaging seeds before pairing splits.

    Use rows of `split_type` and lower-is-better metrics. Skip tests below `min_pairs`.
    Return one row per (scenario, metric), or an empty table if the models or split type are absent.
    """
    subset = results[(results["split_type"] == split_type) & results["model"].isin([model_a, model_b])]
    if subset.empty or not {model_a, model_b}.issubset(set(subset["model"])):
        return pd.DataFrame()

    rows = []
    for scenario, group in subset.groupby("scenario", sort=True):
        wide = group.pivot_table(index="split_id", columns=["metric", "model"], values="value", aggfunc="mean")
        for metric in metrics:
            if (metric, model_a) not in wide.columns or (metric, model_b) not in wide.columns:
                continue
            pair = wide[[(metric, model_a), (metric, model_b)]].dropna()
            if pair.empty:
                continue

            paired = {
                model_a: {metric: pair[(metric, model_a)].tolist()},
                model_b: {metric: pair[(metric, model_b)].tolist()},
            }
            result = compare_models_significance(paired, model_a, model_b, metric=metric, min_pairs=min_pairs)
            rows.append(
                dict(
                    scenario=scenario,
                    metric=metric.upper(),
                    model_a=model_a,
                    model_b=model_b,
                    n_splits=result.n_pairs,
                    mean_difference=result.mean_difference,
                    ci_low=result.ci_low,
                    ci_high=result.ci_high,
                    t_pvalue=result.t_pvalue,
                    significant=bool(np.isfinite(result.ci_low) and (result.ci_low > 0 or result.ci_high < 0)),
                    note=result.note or "",
                )
            )

    return pd.DataFrame(rows)


def summarize_generalization_gap(summary: pd.DataFrame, metric: str = "mse") -> pd.DataFrame:
    """Return shift_gap = OOD - ID-reference and train_pool_gap = ID-reference - ID-random per (scenario, model).

    Use a lower-is-better `metric` from the summary table. Return an empty table if the metric is absent.
    """
    rows = summary[summary["metric"] == metric]
    if rows.empty:
        return pd.DataFrame()

    wide = rows.pivot_table(index=["scenario", "model"], columns="split_type", values="mean").reset_index()
    for split_type in (OOD, ID_REFERENCE, ID_RANDOM):
        if split_type not in wide.columns:
            wide[split_type] = np.nan
    wide["shift_gap"] = wide[OOD] - wide[ID_REFERENCE]
    wide["train_pool_gap"] = wide[ID_REFERENCE] - wide[ID_RANDOM]
    renamed = {split_type: f"{metric.upper()}_{split_type}" for split_type in (OOD, ID_REFERENCE, ID_RANDOM)}
    return wide.rename(columns=renamed)[["scenario", "model", *renamed.values(), "shift_gap", "train_pool_gap"]]
