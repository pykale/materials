"""Progress logging and result formatting for console output."""

import sys
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import COMPARABLE_METRICS, compute_metrics, METRIC_DECIMALS, METRICS
from kalematerials.evaluate.significance import compare_models_significance, SignificanceResult


class Logger:
    """Write text to a stream; None silences output."""

    def __init__(self, stream=sys.stdout):
        self.stream = stream

    def write(self, message: str) -> None:
        """Write `message` verbatim, without adding a newline."""
        if self.stream is not None:
            self.stream.write(message)


SILENT = Logger(None)


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Format a "mean ± std" string, or "nan" if either value is missing."""
    if mean is None or std is None or np.isnan(mean) or np.isnan(std):
        return "nan"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def format_split_results(y_true, predictions: Mapping[str, np.ndarray]) -> str:
    """Format the regression metrics of one split, per model."""
    lines = ["Regression Metrics:"]
    for name, y_pred in predictions.items():
        metrics = compute_metrics(y_true, y_pred)
        lines.append(f"\n{name}:")
        lines.append(f"MSE: {metrics['mse']:.4f}")
        lines.append(f"MAE: {metrics['mae']:.4f}")
        lines.append(f"MRE: {metrics['mre']:.6f}")
        lines.append(f"R2:  {metrics['r2']:.4f}")
    return "\n".join(lines) + "\n"


def format_cross_validation_results(
    results: Mapping[str, Mapping[str, Sequence[float]]],
    title: str = "Cross-Validation Metrics (mean ± std):",
) -> str:
    """Format mean ± std of each metric over the repeats, per model."""

    def _std(values: Sequence[float]) -> float:
        return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    lines = [title]
    for name, scores in results.items():
        lines.append(f"\n{name}:")
        for metric in METRICS:
            values = list(scores[metric])
            label = "R2" if metric == "r2" else metric.upper()
            summary = format_mean_std(float(np.mean(values)), _std(values), METRIC_DECIMALS[metric])
            lines.append(f"{label + ':':<5}{summary}")
    return "\n".join(lines) + "\n"


def format_comparisons(
    results,
    model_a: str,
    model_b: str,
    metrics: Sequence[str] = COMPARABLE_METRICS,
    test_train_ratio: float = 0.0,
) -> str:
    """Format paired significance tests for two models on lower-is-better metrics.

    Scores and `test_train_ratio` follow evaluate.significance.compare_models_significance.
    """
    return "".join(
        format_significance(
            compare_models_significance(
                results,
                model_a,
                model_b,
                metric=metric,
                test_train_ratio=test_train_ratio,
            )
        )
        for metric in metrics
    )


RADIX_NAMES = {2: "binary", 3: "ternary", 4: "quaternary", 5: "quinary", 6: "senary", 7: "septenary"}


def print_compound_counts(counts: pd.Series) -> None:
    """Print the number of compounds per number of distinct elements."""
    for radix, count in counts.items():
        name = RADIX_NAMES.get(int(radix), f"{int(radix)}-component")
        print(f"We have {count} {name} compounds")


def format_search_budget(n_seeds: int, n_folds: int, n_tunable: int) -> str:
    """Format the number of nested hyperparameter searches a run will make."""
    return (
        f"\n[INFO] Nested hyperparameter search: {n_seeds} seed(s) x {n_folds} folds x "
        f"{n_tunable} tunable model(s) = {n_seeds * n_folds * n_tunable} searches.\n"
    )


def print_summary(summary: pd.DataFrame, title: str) -> None:
    """Print mean ± std per model and metric from a summary table."""
    if summary.empty:
        print(f"\n{title}\n  (no observations)")
        return

    print(f"\n{title}")
    for model, group in summary.groupby("model", sort=True):
        print(f"\n{model}:")
        for metric in METRICS:
            row = group[group["metric"] == metric]
            if row.empty:
                continue
            mean, std, n = float(row["mean"].iloc[0]), float(row["std"].iloc[0]), int(row["n"].iloc[0])
            label = "R2" if metric == "r2" else metric.upper()
            print(f"{label + ':':<5}{format_mean_std(mean, std, METRIC_DECIMALS[metric])}  (n={n})")


def format_significance(result: SignificanceResult) -> str:
    """Format one paired model comparison."""
    lines = [
        f"\n{result.model_a} vs {result.model_b} — metric={result.metric.upper()}",
        f"  paired observations: {result.n_pairs}",
    ]

    if np.isnan(result.ci_low):
        lines.append(f"  difference (a - b): {result.mean_difference:+.6g}")
    else:
        lines.append(
            f"  difference (a - b): {result.mean_difference:+.6g}"
            f"  95% CI [{result.ci_low:+.6g}, {result.ci_high:+.6g}]"
        )

    if np.isnan(result.t_pvalue):
        lines.append(f"  no p-value reported: {result.note}")
    else:
        lines.append(f"  paired t-test: t={result.t_stat:.4f}, p={result.t_pvalue:.6g}")
        if result.note:
            lines.append(f"  note: {result.note}")

    return "\n".join(lines) + "\n"


def format_metric_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse `<metric>_mean`/`<metric>_std` column pairs into "mean ± std" strings."""
    if df.empty:
        return df

    out = df.copy()
    for metric in METRICS:
        mean_col, std_col = f"{metric.upper()}_mean", f"{metric.upper()}_std"
        if mean_col not in out.columns or std_col not in out.columns:
            continue
        decimals = METRIC_DECIMALS[metric]
        out[metric.upper()] = [format_mean_std(mean, std, decimals) for mean, std in zip(out[mean_col], out[std_col])]
        out = out.drop(columns=[mean_col, std_col])

    return out


def _print_frame(title: str, df: pd.DataFrame, note: Optional[str] = None) -> None:
    """Print one named result table, or a placeholder when it is empty."""
    print(f"\n{title}")
    if note:
        print(f"({note})")
    print(df.to_string(index=False) if not df.empty else "  (empty)")


def print_ood_tables(
    splits: pd.DataFrame,
    summary: pd.DataFrame,
    significance: pd.DataFrame,
    generalization_gap: pd.DataFrame,
) -> None:
    """Print the OOD result tables, with the summary pivoted wide."""
    _print_frame("Splits", splits)
    wide = summary.pivot_table(index=["scenario", "split_type", "model"], columns="metric", values="mean")
    _print_frame("Summary (mean over splits and seeds)", wide.reset_index())
    _print_frame(
        "Model comparison significance",
        significance,
        note="paired across OOD splits — one observation per split, not per inner fold",
    )
    _print_frame("Generalization gap (MSE), shift vs training-pool", generalization_gap)


def print_uncertainty_report(across_seeds: pd.DataFrame, alpha: float) -> None:
    """Print the calibration table across seeds."""
    nominal = 1.0 - alpha
    print(f"\nCalibration by split type and method (nominal coverage {nominal:.0%}, mean ± std over seeds)")
    print(across_seeds.to_string(index=False))


def print_predictions(predictions: pd.DataFrame, max_rows: int = 50) -> None:
    """Print a prediction table, truncating a long one."""
    shown = predictions.head(max_rows)
    print(shown.to_string(index=False))
    if len(predictions) > max_rows:
        print(f"... {len(predictions) - max_rows} more row(s) not shown")
