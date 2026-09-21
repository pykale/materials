import io

from kalematerials.loaddata.splits import build_kfold_splits, ID_KFOLD
from kalematerials.pipeline.cross_validation import cross_validate, score_models_on_splits
from kalematerials.utils.reporting import Logger


def test_score_models_on_splits(toy_xy, specs):
    X, y = toy_xy
    scores = score_models_on_splits(X, y, specs, build_kfold_splits(len(X), 3))
    assert set(scores) == {"Linear Regression", "Random Forest"} and len(scores["Linear Regression"]["mse"]) == 3


def test_cross_validate(toy_xy, specs):
    X, y = toy_xy
    log = io.StringIO()
    results = cross_validate(
        X, y, specs, folds=3, seeds=(0, 1), compare=("Linear Regression", "Random Forest"), logger=Logger(log)
    )
    assert len(results) == 2 * 3 * 2 * 4 and set(results["seed"]) == {0, 1} and set(results["split_type"]) == {ID_KFOLD}
    assert "CV Run 2/2" in log.getvalue() and "Linear Regression vs Random Forest" in log.getvalue()
