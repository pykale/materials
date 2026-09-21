from kalematerials.predict.fitting import fit_models
from kalematerials.predict.sklearn_models import MODEL_REGISTRY


def test_fit_models(toy_xy):
    X, y = toy_xy
    specs = [MODEL_REGISTRY["ridge"], MODEL_REGISTRY["linear"]]
    fitted = fit_models(X, y, specs, best_hyperparams={"ridge": {"alpha": 7.0}})
    assert list(fitted) == ["ridge", "linear"]
    assert fitted["ridge"].named_steps["model"].alpha == 7.0
    tuned = fit_models(X, y, specs, hyperparameter_tuning=True, tune_cv_folds=2)
    assert tuned["ridge"].named_steps["model"].alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
