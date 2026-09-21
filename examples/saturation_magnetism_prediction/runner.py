"""Translate RunConfig settings into library calls for evaluation, prediction and plotting."""

import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from case_study_references import CASE_STUDIES
from config import RunConfig

from kalematerials.evaluate.metrics import best_model, results_to_scores, wide_table_to_results
from kalematerials.evaluate.ood_summaries import summarize_generalization_gap, summarize_model_comparison
from kalematerials.interpret.alloy_series import build_series_panels
from kalematerials.interpret.dataset_distributions import (
    count_compounds_by_radix,
    plot_target_distribution_by_element,
    plot_target_violin_by_element,
)
from kalematerials.interpret.model_explanations import plot_model_explanations
from kalematerials.interpret.reference_comparison import plot_predictions_against_measurements
from kalematerials.loaddata.element_tables import load_element_properties, load_periodic_table
from kalematerials.loaddata.feature_table import load_feature_table
from kalematerials.loaddata.splits import build_random_split, OOD
from kalematerials.pipeline.cross_validation import cross_validate
from kalematerials.pipeline.ood import build_scenarios, resolve_split_elements, run_ood_evaluation
from kalematerials.pipeline.prediction import check_bundle_matches, fit_bundle, predict_formulas
from kalematerials.pipeline.random_split import evaluate_random_splits
from kalematerials.pipeline.uncertainty import (
    resolve_uncertainty_model,
    run_uncertainty_evaluation,
    UNCERTAINTY_METRICS,
)
from kalematerials.utils.persistence import load_model_bundle, ModelBundle, save_model_bundle, save_results, save_tables
from kalematerials.utils.registry import ModelSpec, resolve_models, resolve_trained_models
from kalematerials.utils.reporting import (
    format_comparisons,
    Logger,
    print_compound_counts,
    print_ood_tables,
    print_predictions,
    print_summary,
    print_uncertainty_report,
    SILENT,
)

# Element subsets and histogram bin edges in tesla.
DISTRIBUTION_ELEMENTS = ("Fe", "Co", "Cr", "Mn")
DISTRIBUTION_BINS = np.arange(0.0, 2.6, 0.2)


def _load_features(cfg: RunConfig) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load (X, y, metadata) from the configured feature table."""
    return load_feature_table(cfg.dataset_path, cfg.feature_columns, target_column=cfg.target_column)


def _logger(cfg: RunConfig) -> Logger:
    """Return a stdout logger when printing is enabled, otherwise SILENT."""
    return Logger() if cfg.print_results else SILENT


def _compared_names(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> Optional[Tuple[str, str]]:
    """Return display names for `compare_models`, or None."""
    if cfg.compare_models is None:
        return None
    name_a, name_b = (spec.name for spec in resolve_models(registry, cfg.compare_models))
    return name_a, name_b


def _save_best_model(cfg: RunConfig, specs, summary, X, y) -> None:
    """Refit the best model by MSE on all rows and save it to `model_dir/<dataset>_<mode>.joblib`."""
    winner = best_model(summary)
    (spec,) = [spec for spec in specs if spec.name == winner]
    bundle = fit_bundle(
        X,
        y,
        spec,
        target_column=cfg.target_column,
        dataset=cfg.dataset,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
    )
    saved = save_model_bundle(Path(cfg.model_dir) / f"{cfg.file_prefix}_{cfg.evaluation_mode}.joblib", bundle)
    if cfg.print_results:
        print(f"\nBest model by MSE: {spec.name}. Refitted on all {len(X)} row(s) and saved to {saved}")


def run_cross_validation(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Repeated K-fold cross-validation; saves the results and the best model."""
    specs = resolve_models(registry, cfg.models)
    X, y, _ = _load_features(cfg)
    compared = _compared_names(cfg, registry)

    results = cross_validate(
        X,
        y,
        specs,
        folds=cfg.kfold.folds,
        shuffle=cfg.kfold.shuffle,
        seeds=cfg.kfold.seeds,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        compare=compared,
        logger=_logger(cfg),
    )

    summary, directory = save_results(results, cfg.output_dir, enabled=cfg.save_results)
    if cfg.print_results and len(cfg.kfold.seeds) > 1:
        print_summary(summary, f"Cross-Validation Metrics across {len(cfg.kfold.seeds)} seed(s) (mean ± std):")
    if directory is not None and cfg.print_results:
        print(f"\nSaved cross-validation results to: {directory.resolve()}")
    if cfg.save_results:
        _save_best_model(cfg, specs, summary, X, y)


def _plot_interpretation(
    cfg: RunConfig,
    registry: Mapping[str, ModelSpec],
    trained: Mapping[str, object],
    feature_columns: Sequence[str],
    X_train,
    X_valid,
    y_valid,
) -> None:
    """Save the requested model-explanation and case-study figures."""
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if cfg.interpret.explain_models:
        specs = resolve_trained_models(registry, trained, cfg.interpret.explain_models)
        plot_model_explanations(
            {spec.name: trained[spec.key] for spec in specs},
            X_train,
            X_valid,
            y_valid,
            save_dir=plots_dir,
            prefix=f"{cfg.file_prefix}_",
            random_state=cfg.model_random_state,
        )

    if cfg.interpret.case_study_models:
        specs = resolve_trained_models(registry, trained, cfg.interpret.case_study_models)
        periodic_table, miedema = load_element_properties(cfg.periodic_table_path, cfg.miedema_path)
        panels = build_series_panels(
            CASE_STUDIES,
            feature_columns,
            {spec.name: trained[spec.key] for spec in specs},
            periodic_table,
            miedema,
        )
        plot_predictions_against_measurements(
            panels,
            y_label=cfg.target_column,
            save_path=plots_dir / f"{cfg.file_prefix}_case_studies.png",
        )


def run_random_split(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Evaluate random splits, save results and the best model, and optionally plot the first seed's models."""
    specs = resolve_models(registry, cfg.models)
    X, y, _ = _load_features(cfg)

    results, first_split_models = evaluate_random_splits(
        X,
        y,
        specs,
        seeds=cfg.random_split.seeds,
        train_size=cfg.random_split.train_size,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        logger=_logger(cfg),
    )

    summary, directory = save_results(results, cfg.output_dir, enabled=cfg.save_results)
    if directory is not None and cfg.print_results:
        print(f"\nSaved random-split results to: {directory.resolve()}")

    if cfg.print_results and len(cfg.random_split.seeds) > 1:
        print_summary(summary, f"Random-Split Metrics across {len(cfg.random_split.seeds)} seed(s) (mean ± std):")
        compared = _compared_names(cfg, registry)
        if compared is not None:
            print(
                format_comparisons(
                    results_to_scores(results),
                    *compared,
                    test_train_ratio=(1.0 - cfg.random_split.train_size) / cfg.random_split.train_size,
                ),
                end="",
            )

    if cfg.save_results:
        _save_best_model(cfg, specs, summary, X, y)

    if not cfg.interpret.enabled:
        return

    _, train_idx, valid_idx = build_random_split(len(X), cfg.random_split.train_size, int(cfg.random_split.seeds[0]))
    _plot_interpretation(
        cfg,
        registry,
        first_split_models,
        list(X.columns),
        X.iloc[train_idx],
        X.iloc[valid_idx],
        y.iloc[valid_idx],
    )


def _read_formulas(cfg: RunConfig) -> pd.DataFrame:
    """Read formulas from the configured CSV or inline list into a one-column frame."""
    if cfg.predict.input_path:
        data = pd.read_csv(cfg.predict.input_path)
        if cfg.formula_column not in data.columns:
            raise ValueError(f"{cfg.predict.input_path} has no {cfg.formula_column!r} column: {sorted(data.columns)}")
        formulas = data[[cfg.formula_column]].copy()
    elif cfg.predict.formulas:
        formulas = pd.DataFrame({cfg.formula_column: list(cfg.predict.formulas)})
    else:
        raise ValueError("Set predict.input_path or predict.formulas.")

    formulas = formulas.dropna(subset=[cfg.formula_column]).reset_index(drop=True)
    if formulas.empty:
        raise ValueError("No usable chemical formulas to predict.")
    return formulas


def _load_training_table(cfg: RunConfig) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Load (X, y) from the configured feature table, or None if the file is absent."""
    if not cfg.dataset_path or not Path(cfg.dataset_path).exists():
        return None
    X, y, _ = _load_features(cfg)
    return X, y


def _resolve_bundle(cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> ModelBundle:
    """Load a bundle and check it against available training data, or fit and save one."""
    path = cfg.predict.model_path
    if path and Path(path).exists() and not cfg.predict.retrain:
        bundle = load_model_bundle(path)
        print(f"Loaded model from {Path(path).resolve()}")
        training = _load_training_table(cfg)
        if training is not None:
            check_bundle_matches(bundle, *training)
        return bundle

    if cfg.predict.model is None:
        raise ValueError(f"No saved model at {path!r}; run an evaluation mode first or set predict.model.")
    (spec,) = resolve_models(registry, [cfg.predict.model])
    training = _load_training_table(cfg)
    if training is None:
        raise ValueError(f"No training data at dataset_path {cfg.dataset_path!r}.")

    X, y = training
    print(f"Fitting {spec.name} on all {len(X)} row(s) of {cfg.dataset_path}")
    bundle = fit_bundle(
        X,
        y,
        spec,
        target_column=cfg.target_column,
        dataset=cfg.dataset,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
    )
    print(f"Saved model to {save_model_bundle(path, bundle)}")
    return bundle


def run_predict(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> pd.DataFrame:
    """Return predictions for the configured compositions, loading or fitting a model as needed."""
    bundle = _resolve_bundle(cfg, registry)
    print(f"Model: {bundle.describe()}")

    formulas = _read_formulas(cfg)
    periodic_table, miedema = load_element_properties(cfg.periodic_table_path, cfg.miedema_path)
    predictions, skipped = predict_formulas(
        bundle,
        formulas,
        periodic_table,
        miedema,
        formula_column=cfg.formula_column,
    )

    if skipped:
        warnings.warn(f"Could not featurize {len(skipped)} formula(s): {skipped}")

    print(f"\nPredicted {bundle.target_column} for {len(predictions)} composition(s):")
    if cfg.print_results:
        print_predictions(predictions)

    directory = save_tables({"predictions.csv": predictions}, cfg.output_dir, enabled=cfg.save_results)
    if directory is not None:
        print(f"\nSaved predictions to: {(directory / 'predictions.csv').resolve()}")

    return predictions


def run_data_visualization(*, cfg: RunConfig) -> None:
    """Save the target-distribution figures and print the compound counts."""
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _, y, metadata = _load_features(cfg)
    data = metadata.assign(**{cfg.target_column: y})

    plot_target_distribution_by_element(
        data,
        target_column=cfg.target_column,
        elements=DISTRIBUTION_ELEMENTS,
        formula_column=cfg.formula_column,
        bins=DISTRIBUTION_BINS,
        save_path=plots_dir / f"{cfg.file_prefix}_ms_distribution_by_tm.png",
    )
    plot_target_violin_by_element(
        data,
        target_column=cfg.target_column,
        elements=DISTRIBUTION_ELEMENTS,
        formula_column=cfg.formula_column,
        target_label="Saturation Magnetization (T)",
        title=f"{cfg.file_prefix.upper()} Violin Plot",
        save_path=plots_dir / f"{cfg.file_prefix}_violin_ms_by_tm.png",
    )
    print_compound_counts(count_compounds_by_radix(data, formula_column=cfg.formula_column))


def _ood_scenarios(cfg: RunConfig, X, y, metadata):
    """Build the configured OOD scenarios."""
    elements_per_row, element_to_group, element_to_period = resolve_split_elements(
        metadata,
        cfg.formula_column,
        load_periodic_table(cfg.periodic_table_path),
    )
    return build_scenarios(
        X,
        y,
        elements_per_row,
        element_to_group,
        element_to_period,
        scenarios=cfg.ood.scenarios,
        min_train=cfg.ood.min_train,
        min_test=cfg.ood.min_test,
        max_splits=cfg.ood.max_splits,
        elements=cfg.ood.elements,
        periods=cfg.ood.periods,
        groups=cfg.ood.groups,
        systems=[tuple(system.split("-")) for system in cfg.ood.systems],
        period_strict=cfg.ood.period_strict,
        group_strict=cfg.ood.group_strict,
        n_clusters=cfg.ood.n_clusters,
        cluster_seed=cfg.ood.cluster_seed,
        fractions=cfg.ood.fractions,
        sparsex_neighbors=cfg.ood.sparsex_neighbors,
        sparsey_center=cfg.ood.sparsey_center,
    )


def run_ood(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Run OOD evaluation and save scores, splits, significance tests and generalization gaps."""
    specs = resolve_models(registry, cfg.models)
    X, y, metadata = _load_features(cfg)

    splits, results = run_ood_evaluation(
        X,
        y,
        specs,
        _ood_scenarios(cfg, X, y, metadata),
        seeds=cfg.ood.seeds,
        inner_folds=cfg.kfold.folds,
        inner_shuffle=cfg.kfold.shuffle,
        size_matched_control=cfg.ood.size_matched_control,
        hyperparameter_tuning=cfg.tuning.enabled,
        tune_cv_folds=cfg.tuning.cv_folds,
        tune_n_iter=cfg.tuning.n_iter,
        model_random_state=cfg.model_random_state,
        logger=_logger(cfg),
    )

    summary, directory = save_results(results, cfg.output_dir, enabled=cfg.save_results)

    compared = _compared_names(cfg, registry)
    significance = (
        summarize_model_comparison(results, *compared, split_type=OOD) if compared is not None else pd.DataFrame()
    )
    generalization_gap = summarize_generalization_gap(summary)
    save_tables(
        {"splits.csv": splits, "significance.csv": significance, "generalization_gap.csv": generalization_gap},
        cfg.output_dir,
        enabled=cfg.save_results,
    )

    if cfg.print_results:
        print_ood_tables(splits, summary, significance, generalization_gap)
        if directory is not None:
            print(f"\nSaved OOD results to: {directory.resolve()}")


def run_uncertainty(*, cfg: RunConfig, registry: Mapping[str, ModelSpec]) -> None:
    """Evaluate uncertainty and save calibration tables per split, per seed and across seeds."""
    spec = resolve_uncertainty_model(registry, cfg.uncertainty.model)
    X, y, metadata = _load_features(cfg)

    tables = run_uncertainty_evaluation(
        X,
        y,
        spec,
        _ood_scenarios(cfg, X, y, metadata),
        seeds=cfg.uncertainty.seeds,
        alpha=cfg.uncertainty.alpha,
        calibration_fraction=cfg.uncertainty.calibration_fraction,
        inner_folds=cfg.kfold.folds,
        inner_shuffle=cfg.kfold.shuffle,
        size_matched_control=cfg.ood.size_matched_control,
        model_random_state=cfg.model_random_state,
        logger=_logger(cfg),
    )
    if tables is None:
        return

    # Here `model` labels the interval method; uncertainty.model selects the estimator.
    results = wide_table_to_results(
        tables.by_split,
        {metric: metric for metric in UNCERTAINTY_METRICS},
        identifiers={
            "scenario": "scenario",
            "split_type": "split_type",
            "split_id": "split_id",
            "seed": "seed",
            "model": "method",
        },
    )
    _, directory = save_results(results, cfg.output_dir, enabled=cfg.save_results)
    save_tables(
        {"pooled_by_seed.csv": tables.pooled_by_seed, "across_seeds.csv": tables.across_seeds},
        cfg.output_dir,
        enabled=cfg.save_results,
    )

    if cfg.print_results:
        print_uncertainty_report(tables.across_seeds, cfg.uncertainty.alpha)
        if directory is not None:
            print(f"\nSaved uncertainty results to: {directory.resolve()}")
