"""Regression models: a build, tune and train function per model, and the MODEL_REGISTRY that names them."""

from typing import Dict, List, Optional, Union

import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from kalematerials.utils.registry import ModelSpec


def _scaled(estimator) -> Pipeline:
    """Wrap `estimator` in a Pipeline with a preceding StandardScaler."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def _prefix_hyperparams(hyperparams: Union[Dict, List[Dict]]) -> Union[Dict, List[Dict]]:
    """Prefix names for a scaled Pipeline ("alpha" -> "model__alpha"); a list of grids is prefixed per grid."""
    if isinstance(hyperparams, list):
        return [_prefix_hyperparams(sub) for sub in hyperparams]
    return {f"model__{key}": value for key, value in hyperparams.items()}


def _strip_hyperparams(hyperparams: Dict) -> Dict:
    """Inverse of _prefix_hyperparams."""
    return {key.split("__", 1)[-1]: value for key, value in hyperparams.items()}


def _grid_search_best_hyperparams(
    model,
    param_grid: Union[Dict, List[Dict]],
    X_train,
    y_train,
    cv_folds: int,
    scale: bool = False,
) -> Dict:
    """Grid-search `model` and return the best hyperparameters as build_* kwargs."""
    grid_search = GridSearchCV(
        estimator=_scaled(model) if scale else model,
        param_grid=_prefix_hyperparams(param_grid) if scale else param_grid,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    return _strip_hyperparams(grid_search.best_params_) if scale else grid_search.best_params_


def _randomized_search_best_hyperparams(
    model,
    param_dist: Dict,
    X_train,
    y_train,
    cv_folds: int,
    random_state: int,
    n_iter: int = 20,
    scale: bool = False,
) -> Dict:
    """Randomized-search `model` and return the best hyperparameters as build_* kwargs."""
    random_search = RandomizedSearchCV(
        estimator=_scaled(model) if scale else model,
        param_distributions=_prefix_hyperparams(param_dist) if scale else param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=random_state,
    )
    random_search.fit(X_train, y_train)
    return _strip_hyperparams(random_search.best_params_) if scale else random_search.best_params_


# --- Linear Regression ---


def build_linear_regression() -> LinearRegression:
    """Construct a linear regression model."""
    return LinearRegression()


def train_linear_regression(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled linear regression."""
    model = _scaled(build_linear_regression())
    model.fit(X_train, y_train)
    return model


# --- Ridge ---


def build_ridge(alpha: float = 1.0) -> Ridge:
    """Construct a Ridge regression model."""
    return Ridge(alpha=alpha, max_iter=10000)


def tune_ridge_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search Ridge hyperparameters."""
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }
    return _grid_search_best_hyperparams(build_ridge(), param_grid, X_train, y_train, cv_folds, scale=True)


def train_ridge(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled Ridge regression."""
    if hyperparams is not None:
        model = build_ridge(**hyperparams)
    else:
        model = build_ridge(alpha=1.0)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Lasso ---


def build_lasso(alpha: float = 0.01) -> Lasso:
    """Construct a Lasso regression model."""
    return Lasso(alpha=alpha, max_iter=10000)


def tune_lasso_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search Lasso hyperparameters."""
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
    }
    return _grid_search_best_hyperparams(build_lasso(), param_grid, X_train, y_train, cv_folds, scale=True)


def train_lasso(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled Lasso regression."""
    if hyperparams is not None:
        model = build_lasso(**hyperparams)
    else:
        model = build_lasso(alpha=0.001)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- ElasticNet ---


def build_elasticnet(alpha: float = 0.01, l1_ratio: float = 0.5) -> ElasticNet:
    """Construct an ElasticNet regression model."""
    return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)


def tune_elasticnet_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search ElasticNet hyperparameters."""
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return _grid_search_best_hyperparams(build_elasticnet(), param_grid, X_train, y_train, cv_folds, scale=True)


def train_elasticnet(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled ElasticNet regression."""
    if hyperparams is not None:
        model = build_elasticnet(**hyperparams)
    else:
        model = build_elasticnet(alpha=0.001, l1_ratio=0.1)
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Random Forest ---


def build_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: Union[str, float] = "sqrt",
    random_state: int = 0,
    n_jobs: int = 1,
) -> RandomForestRegressor:
    """Construct a random forest regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def tune_random_forest_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search random forest hyperparameters."""
    param_grid = {
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": [0.5, 0.75, 1.0],
    }
    return _grid_search_best_hyperparams(
        build_random_forest(random_state=random_state), param_grid, X_train, y_train, cv_folds
    )


def train_random_forest(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a random forest (unscaled)."""
    if hyperparams is not None:
        model = build_random_forest(**hyperparams, random_state=random_state)
    else:
        model = build_random_forest(
            max_depth=15,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=random_state,
        )
    model.fit(X_train, y_train)
    return model


# --- XGBoost ---


def build_xgboost(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    min_child_weight: int = 1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1.0,
    random_state: int = 0,
    n_jobs: int = 1,
) -> xgboost.XGBRegressor:
    """Construct an XGBoost regressor."""
    return xgboost.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def tune_xgboost_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search XGBoost hyperparameters; learning_rate and n_estimators are paired to keep their product fixed."""
    param_grid = [
        {
            "learning_rate": [learning_rate],
            "n_estimators": [n_estimators],
            "max_depth": [3, 5, 7],
            "reg_lambda": [1.0, 10.0],
            "subsample": [0.6, 0.8, 1.0],
        }
        for learning_rate, n_estimators in [(0.01, 1500), (0.05, 600), (0.1, 300)]
    ]
    return _grid_search_best_hyperparams(
        build_xgboost(random_state=random_state), param_grid, X_train, y_train, cv_folds
    )


def train_xgboost(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit an XGBoost regressor (unscaled)."""
    if hyperparams is not None:
        model = build_xgboost(**hyperparams, random_state=random_state)
    else:
        model = build_xgboost(
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=7,
            subsample=0.6,
            colsample_bytree=0.6,
            random_state=random_state,
        )
    model.fit(X_train, y_train)
    return model


# --- Support Vector Regression ---


def build_svr(
    C: float = 10.0,
    epsilon: float = 0.1,
    kernel: str = "rbf",
    gamma: str = "scale",
    degree: int = 3,
) -> SVR:
    """Construct a Support Vector Regression model."""
    return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)


def tune_svr_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Grid-search SVR hyperparameters (rbf kernel)."""
    param_grid = {
        "C": [1.0, 10.0, 100.0, 1000.0],
        "gamma": ["scale", 0.01, 0.1, 1.0],
        "epsilon": [0.01, 0.05, 0.1],
    }
    return _grid_search_best_hyperparams(build_svr(), param_grid, X_train, y_train, cv_folds, scale=True)


def train_svr(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled SVR."""
    if hyperparams is not None:
        model = build_svr(**hyperparams)
    else:
        model = build_svr(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Multi-layer Perceptron ---


def build_mlp(
    hidden_layer_sizes: tuple = (128, 64),
    activation: str = "relu",
    alpha: float = 1e-4,
    learning_rate: str = "constant",
    learning_rate_init: float = 1e-3,
    max_iter: int = 1000,
    early_stopping: bool = True,
    random_state: int = 0,
) -> MLPRegressor:
    """Construct a multi-layer perceptron regressor."""
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        random_state=random_state,
    )


def tune_mlp_hyperparams(
    X_train,
    y_train,
    cv_folds: int = 3,
    random_state: int = 0,
    n_iter: int = 20,
) -> Dict:
    """Randomized-search MLP hyperparameters."""
    param_dist = {
        "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (256, 128), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "learning_rate_init": [1e-4, 1e-3, 5e-3, 1e-2],
    }
    return _randomized_search_best_hyperparams(
        build_mlp(random_state=random_state),
        param_dist,
        X_train,
        y_train,
        cv_folds,
        random_state,
        n_iter=n_iter,
        scale=True,
    )


def train_mlp(X_train, y_train, hyperparams: Optional[Dict] = None, random_state: int = 0):
    """Fit a scaled MLP."""
    if hyperparams is not None:
        model = build_mlp(**hyperparams, random_state=random_state)
    else:
        model = build_mlp(
            hidden_layer_sizes=(256, 128),
            activation="tanh",
            alpha=1e-5,
            learning_rate_init=1e-3,
            random_state=random_state,
        )
    model = _scaled(model)
    model.fit(X_train, y_train)
    return model


# --- Model Registry ---

MODEL_REGISTRY = {
    spec.key: spec
    for spec in [
        ModelSpec("linear", "Linear Regression", train_linear_regression),
        ModelSpec("ridge", "Ridge", train_ridge, tune_ridge_hyperparams),
        ModelSpec("lasso", "Lasso", train_lasso, tune_lasso_hyperparams),
        ModelSpec("elasticnet", "ElasticNet", train_elasticnet, tune_elasticnet_hyperparams),
        ModelSpec("rf", "Random Forest", train_random_forest, tune_random_forest_hyperparams),
        ModelSpec("xgb", "XGBoost", train_xgboost, tune_xgboost_hyperparams),
        ModelSpec("svr", "SVR", train_svr, tune_svr_hyperparams),
        ModelSpec("mlp", "MLP", train_mlp, tune_mlp_hyperparams),
    ]
}
