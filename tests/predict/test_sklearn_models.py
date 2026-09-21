import pytest

from kalematerials.predict import sklearn_models as m

HYPERPARAMS = {
    "ridge": {"alpha": 0.5},
    "lasso": {"alpha": 0.5},
    "elasticnet": {"alpha": 0.5, "l1_ratio": 0.2},
    "rf": {"max_depth": 2},
    "xgb": {"max_depth": 2},
    "svr": {"C": 1.0},
    "mlp": {"hidden_layer_sizes": (4,), "max_iter": 50},
}


@pytest.mark.parametrize("key", list(m.MODEL_REGISTRY))
def test_every_registered_model_trains(key, toy_xy):
    X, y = toy_xy
    spec = m.MODEL_REGISTRY[key]
    assert spec.train(X, y).predict(X).shape == (len(X),)
    assert spec.train(X, y, hyperparams=HYPERPARAMS.get(key)).predict(X).shape == (len(X),)


@pytest.mark.parametrize("key", ["ridge", "lasso", "elasticnet", "svr", "rf", "mlp"])
def test_tuning_returns_hyperparams_train_accepts(key, toy_xy):
    X, y = toy_xy
    spec = m.MODEL_REGISTRY[key]
    hyperparams = spec.tune(X, y, cv_folds=2, random_state=0, n_iter=1)
    assert hyperparams and not any("__" in name for name in hyperparams)
    spec.train(X, y, hyperparams=hyperparams)


def test_hyperparam_prefixing():
    assert m._prefix_hyperparams({"alpha": [1]}) == {"model__alpha": [1]}
    assert m._prefix_hyperparams([{"a": 1}, {"b": 2}]) == [{"model__a": 1}, {"model__b": 2}]
    assert m._strip_hyperparams({"model__alpha": 1}) == {"alpha": 1}
    assert m.MODEL_REGISTRY["linear"].tune is None
