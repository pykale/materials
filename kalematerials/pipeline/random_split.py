"""Repeated random train/validation splits."""

from typing import Dict, Sequence, Tuple

import pandas as pd

from kalematerials.evaluate.metrics import build_result_rows, compute_metrics
from kalematerials.loaddata.splits import build_random_split, ID_RANDOM
from kalematerials.predict.fitting import fit_models
from kalematerials.utils.registry import ModelSpec
from kalematerials.utils.reporting import format_split_results, Logger, SILENT


def evaluate_random_splits(
    X: pd.DataFrame,
    y: pd.Series,
    specs: Sequence[ModelSpec],
    *,
    seeds: Sequence[int] = (0,),
    train_size: float = 0.8,
    hyperparameter_tuning: bool = False,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
    model_random_state: int = 0,
    logger: Logger = SILENT,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Evaluate a random train/validation split per seed.

    `seeds` control splits; `model_random_state` controls models and tuning. Tuning uses only each
    split's training rows, with search settings from `fit_models`.
    Return (results, models): long-form results for all seeds and the first seed's models keyed by registry key.
    """
    collected = []
    first_split_models: Dict[str, object] = {}

    for run_i, seed in enumerate(seeds, start=1):
        logger.write(f"\n=== Random Split Run {run_i}/{len(seeds)} (split seed={seed}) ===\n")

        _, train_idx, valid_idx = build_random_split(len(X), train_size, int(seed))
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        trained = fit_models(
            X_train,
            y_train,
            specs,
            hyperparameter_tuning=hyperparameter_tuning,
            model_random_state=model_random_state,
            tune_cv_folds=tune_cv_folds,
            tune_n_iter=tune_n_iter,
        )
        predictions = {spec.name: trained[spec.key].predict(X_valid) for spec in specs}

        logger.write(format_split_results(y_valid, predictions))

        per_seed = {name: compute_metrics(y_valid, y_pred) for name, y_pred in predictions.items()}
        collected.append(
            build_result_rows(
                {name: {metric: [value] for metric, value in metrics.items()} for name, metrics in per_seed.items()},
                scenario="random_split",
                split_type=ID_RANDOM,
                split_ids=[f"seed{seed}"],
                seed=int(seed),
            )
        )

        if not first_split_models:
            first_split_models = trained

    return pd.concat(collected, ignore_index=True), first_split_models
