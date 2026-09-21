"""Build out-of-distribution scenarios and score them alongside in-distribution references."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import build_result_rows, compute_metrics, FoldScores, METRICS
from kalematerials.loaddata.splits import (
    build_group_splits,
    build_kfold_splits,
    build_kmeans_cluster_splits,
    build_loeo_splits,
    build_period_splits,
    build_size_matched_controls,
    build_sparsex_splits,
    build_sparsey_splits,
    build_system_splits,
    ID_RANDOM,
    ID_REFERENCE,
    OOD,
    Split,
)
from kalematerials.predict.fitting import fit_models
from kalematerials.prepdata.composition import get_elements_per_row, get_group_period_maps
from kalematerials.utils.registry import ModelSpec
from kalematerials.utils.reporting import Logger, SILENT


def _counts_by_attr(
    elements_per_row: Sequence[Sequence[str]],
    element_to_attr: Optional[Dict[str, int]] = None,
) -> Dict[Any, int]:
    """Count rows containing each element (or each element attribute)."""
    counts: Dict[Any, int] = {}
    for els in elements_per_row:
        if element_to_attr is None:
            keys = set(els)
        else:
            keys = {element_to_attr[e] for e in set(els) if e in element_to_attr}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _ranked_targets(counts: Dict[Any, int], max_n: Optional[int]) -> List[Any]:
    """Return targets by decreasing frequency, limited to `max_n` when set."""
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ordered = ordered if max_n is None else ordered[:max_n]
    return [key for key, _ in ordered]


def _capped(splits: List[Split], max_splits: Optional[int]) -> List[Split]:
    return splits if max_splits is None else splits[:max_splits]


def _warn_unbuilt(scenario: str, splits: List[Split], named_ids: Sequence[str] = ()) -> None:
    """Warn about scenarios or requested targets with no split meeting the size limits."""
    built_ids = {split_id for split_id, _, _ in splits}
    missing = [split_id for split_id in named_ids if split_id not in built_ids]
    if not splits:
        warnings.warn(f"{scenario}: no split passed min_train/min_test; scenario dropped.")
    elif missing:
        warnings.warn(f"{scenario}: {missing} did not pass min_train/min_test; dropped.")


def resolve_split_elements(
    metadata: pd.DataFrame,
    formula_column: str,
    periodic_table: pd.DataFrame,
) -> Tuple[List[List[str]], Dict[str, int], Dict[str, int]]:
    """Return (elements_per_row, element_to_group, element_to_period) from metadata and the periodic table."""
    if formula_column not in metadata.columns:
        raise ValueError(f"Missing column {formula_column!r} in metadata. Available: {sorted(metadata.columns)}")
    element_to_group, element_to_period = get_group_period_maps(periodic_table)
    return get_elements_per_row(metadata[formula_column]), element_to_group, element_to_period


def build_scenarios(
    X: pd.DataFrame,
    y: pd.Series,
    elements_per_row: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    element_to_period: Dict[str, int],
    *,
    scenarios: Sequence[str] = ("element", "period", "group", "cluster", "sparsex", "sparsey"),
    min_train: int = 1,
    min_test: int = 1,
    max_splits: Optional[int] = None,
    elements: Optional[Sequence[str]] = None,
    periods: Optional[Sequence[int]] = None,
    groups: Optional[Sequence[int]] = None,
    systems: Sequence[Sequence[str]] = (),
    period_strict: bool = False,
    group_strict: bool = False,
    n_clusters: int = 5,
    cluster_seed: int = 0,
    fractions: Sequence[float] = (0.1, 0.2),
    sparsex_neighbors: int = 5,
    sparsey_center: str = "median",
) -> List[Tuple[str, List[Split]]]:
    """Return (scenario_name, splits) pairs for the selected OOD scenarios, omitting empty scenarios.

    Args:
        X, y: Features for cluster/sparsex splits and targets for sparsey splits.
        elements_per_row: Element symbols per sample.
        element_to_group, element_to_period: Element-to-group and element-to-period maps.
        scenarios: Names from element, period, group, system, cluster, sparsex and sparsey.
        min_train, min_test: Minimum rows on each side of a split.
        max_splits: Maximum splits per scenario, selecting hold-out targets by decreasing frequency.
        elements, periods, groups: Explicit hold-out targets; None selects by frequency.
        systems: Element sets to hold out together, e.g. (("Fe", "Co"),).
        period_strict, group_strict: Require every element in a held-out sample to belong to the period/group.
        n_clusters, cluster_seed: Cluster count and seed for k-means.
        fractions: Hold-out fractions for sparsex and sparsey.
        sparsex_neighbors: Neighbours used to measure feature-space isolation.
        sparsey_center: "median" or "mean".
    """
    sizes = dict(min_train=min_train, min_test=min_test)
    built: List[Tuple[str, List[Split]]] = []

    def selected(name: str) -> bool:
        return name in scenarios

    def targets_for(configured, element_to_attr=None) -> List[Any]:
        """Use explicit targets when given, otherwise select by frequency."""
        if configured is not None:
            return list(configured)
        return _ranked_targets(_counts_by_attr(elements_per_row, element_to_attr), max_splits)

    def named(label: str, configured) -> List[str]:
        """Return requested split IDs for warnings about targets omitted by size limits."""
        return [f"{label}={target}" for target in configured] if configured else []

    if selected("element"):
        built.append(("LOEO", build_loeo_splits(elements_per_row, targets_for(elements), **sizes)))
        _warn_unbuilt("LOEO", built[-1][1], named("E", elements))

    if selected("period"):
        built.append(
            (
                "LOPO",
                build_period_splits(
                    elements_per_row,
                    element_to_period,
                    targets_for(periods, element_to_period),
                    strict=period_strict,
                    **sizes,
                ),
            )
        )
        _warn_unbuilt("LOPO", built[-1][1], named("P", periods))

    if selected("group"):
        built.append(
            (
                "LOGO",
                build_group_splits(
                    elements_per_row,
                    element_to_group,
                    targets_for(groups, element_to_group),
                    strict=group_strict,
                    **sizes,
                ),
            )
        )
        _warn_unbuilt("LOGO", built[-1][1], named("G", groups))

    if selected("system"):
        built.append(("LOSO", build_system_splits(elements_per_row, systems, **sizes)))
        _warn_unbuilt("LOSO", built[-1][1], named("S", ["-".join(system) for system in systems]))

    if selected("cluster"):
        built.append(
            (
                f"LOCO(k={n_clusters})",
                build_kmeans_cluster_splits(X, k=n_clusters, seed=cluster_seed, **sizes),
            )
        )
        _warn_unbuilt(built[-1][0], built[-1][1])

    if selected("sparsex"):
        built.append(
            (
                "SparseX",
                build_sparsex_splits(X, fractions=fractions, n_neighbors=sparsex_neighbors, **sizes),
            )
        )
        _warn_unbuilt("SparseX", built[-1][1])

    if selected("sparsey"):
        built.append(
            (
                "SparseY",
                build_sparsey_splits(y, fractions=fractions, center=sparsey_center, **sizes),
            )
        )
        _warn_unbuilt("SparseY", built[-1][1])

    return [(name, _capped(splits, max_splits)) for name, splits in built if splits]


def _empty_scores(specs: Sequence[ModelSpec]) -> FoldScores:
    return {spec.name: {metric: [] for metric in METRICS} for spec in specs}


def _mean_over_folds(scores: FoldScores) -> FoldScores:
    """Replace each metric's per-fold values by their mean."""
    return {
        model: {metric: [float(np.mean(values))] for metric, values in per_metric.items()}
        for model, per_metric in scores.items()
    }


def heldout_label(split_id: str) -> str:
    """The held-out target in a split id such as "E=Fe" or "P=4"; "" for ids without one."""
    return split_id.split("=", 1)[1] if "=" in split_id else ""


def _score_fold_models(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    specs: Sequence[ModelSpec],
    *,
    inner_splits: Sequence[Split],
    score_inner: bool,
    hyperparameter_tuning: bool,
    model_random_state: int,
    tune_cv_folds: int,
    tune_n_iter: int,
) -> Tuple[FoldScores, Optional[FoldScores]]:
    """Fit on inner folds of `train_idx` and score `test_idx`.

    Return (test_scores, inner_scores); `inner_scores` is None unless `score_inner` is enabled.
    """
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    test_scores = _empty_scores(specs)
    inner_scores = _empty_scores(specs) if score_inner else None

    for _, fit_idx, inner_idx in inner_splits:
        X_fit, y_fit = X_train_full.iloc[fit_idx], y_train_full.iloc[fit_idx]
        X_inner, y_inner = X_train_full.iloc[inner_idx], y_train_full.iloc[inner_idx]

        fitted = fit_models(
            X_fit,
            y_fit,
            specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )

        for spec in specs:
            model = fitted[spec.key]
            for metric, value in compute_metrics(y_test, model.predict(X_test)).items():
                test_scores[spec.name][metric].append(value)
            if score_inner:
                for metric, value in compute_metrics(y_inner, model.predict(X_inner)).items():
                    inner_scores[spec.name][metric].append(value)

    return test_scores, inner_scores


def score_splits_with_inner_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Sequence[Split],
    specs: Sequence[ModelSpec],
    *,
    scenario: str,
    seed: int,
    inner_folds: int,
    inner_shuffle: bool,
    hyperparameter_tuning: bool,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    controls: Optional[Mapping[str, Split]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score one scenario and seed as OOD, ID_REFERENCE and optional ID_RANDOM rows.

    `seed` controls inner K-fold splits; `model_random_state` controls models and tuning, as in `fit_models`.
    `controls` maps split IDs to size-matched random splits.
    Return (splits, results): split labels and sizes, plus long-form scores averaged over inner folds.
    """
    split_rows: List[Dict] = []
    result_frames: List[pd.DataFrame] = []

    def rows(split_id: str, split_type: str, scores: FoldScores) -> pd.DataFrame:
        return build_result_rows(
            _mean_over_folds(scores),
            scenario=scenario,
            split_type=split_type,
            split_ids=[split_id],
            seed=seed,
        )

    for split_id, train_idx, test_idx in splits:
        inner_splits = build_kfold_splits(len(train_idx), inner_folds, inner_shuffle, seed)
        if not inner_splits:
            warnings.warn(f"Skipping split {split_id} in {scenario}: n_train={len(train_idx)} leaves no inner folds")
            continue

        score = partial(
            _score_fold_models,
            X,
            y,
            specs=specs,
            inner_splits=inner_splits,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )

        ood_scores, id_reference_scores = score(train_idx, test_idx, score_inner=True)
        split_rows.append(
            dict(
                scenario=scenario,
                split_id=split_id,
                heldout_label=heldout_label(split_id),
                n_train=len(train_idx),
                n_test=len(test_idx),
                seed=seed,
            )
        )
        result_frames.append(rows(split_id, OOD, ood_scores))
        result_frames.append(rows(split_id, ID_REFERENCE, id_reference_scores))

        control = (controls or {}).get(split_id)
        if control is not None:
            _, control_train_idx, control_test_idx = control
            control_scores, _ = score(control_train_idx, control_test_idx, score_inner=False)
            result_frames.append(rows(split_id, ID_RANDOM, control_scores))

    results = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    return pd.DataFrame(split_rows), results


def run_ood_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    scenarios: Sequence[Tuple[str, List[Split]]],
    *,
    seeds: Sequence[int] = (0,),
    inner_folds: int = 5,
    inner_shuffle: bool = True,
    size_matched_control: bool = True,
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every (scenario_name, splits) pair alongside its in-distribution references.

    Repeat per seed, which controls inner folds and size-matched random controls. Tuning runs within each
    inner training fold, using `fit_models` settings. `size_matched_control` enables random controls.
    Return (splits, results) as in `score_splits_with_inner_kfold`, across all scenarios and seeds.
    """
    if not scenarios:
        raise ValueError("No scenario has a split; nothing to score.")

    logger.write(
        f"\n[INFO] OOD run: {len(scenarios)} scenario(s), "
        f"{sum(len(s) for _, s in scenarios)} split(s), {len(seeds)} seed(s), "
        f"size-matched control {'on' if size_matched_control else 'off'}.\n"
    )
    for name, splits in scenarios:
        logger.write(f"         {name}: {len(splits)} split(s) — {[s[0] for s in splits]}\n")

    split_frames: List[pd.DataFrame] = []
    result_frames: List[pd.DataFrame] = []
    for scenario, splits in scenarios:
        for seed in seeds:
            controls = (
                build_size_matched_controls(splits, len(X), int(seed), scenario) if size_matched_control else None
            )
            split_frame, result_frame = score_splits_with_inner_kfold(
                X,
                y,
                splits,
                specs,
                scenario=scenario,
                seed=int(seed),
                inner_folds=inner_folds,
                inner_shuffle=inner_shuffle,
                hyperparameter_tuning=hyperparameter_tuning,
                model_random_state=model_random_state,
                tune_cv_folds=tune_cv_folds,
                tune_n_iter=tune_n_iter,
                controls=controls,
            )
            split_frames.append(split_frame)
            result_frames.append(result_frame)

    return pd.concat(split_frames, ignore_index=True), pd.concat(result_frames, ignore_index=True)
