"""Fit model specifications on one set of rows."""

from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from kalematerials.utils.registry import ModelSpec


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    specs: Sequence[ModelSpec],
    *,
    hyperparameter_tuning: bool = False,
    best_hyperparams: Optional[Mapping[str, Dict]] = None,
    model_random_state: int = 0,
    tune_cv_folds: int = 3,
    tune_n_iter: int = 20,
) -> Dict[str, object]:
    """Fit each spec on the training rows and return {spec.key: fitted estimator}.

    Use `best_hyperparams` when supplied; otherwise tune on these rows if enabled, or use defaults.
    `model_random_state` seeds models and searches. `tune_cv_folds` sets inner folds;
    `tune_n_iter` sets the number of randomized-search candidates.
    """
    fitted: Dict[str, object] = {}
    for spec in specs:
        hyperparams = None
        if best_hyperparams is not None and spec.key in best_hyperparams:
            hyperparams = best_hyperparams[spec.key]
        elif hyperparameter_tuning and spec.tune is not None:
            hyperparams = spec.tune(
                X_train,
                y_train,
                cv_folds=tune_cv_folds,
                random_state=model_random_state,
                n_iter=tune_n_iter,
            )
        fitted[spec.key] = spec.train(X_train, y_train, hyperparams=hyperparams, random_state=model_random_state)
    return fitted
