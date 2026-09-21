from kalematerials.loaddata.splits import ID_RANDOM
from kalematerials.pipeline.random_split import evaluate_random_splits


def test_evaluate_random_splits(toy_xy, specs):
    X, y = toy_xy
    results, models = evaluate_random_splits(X, y, specs, seeds=(0, 1), train_size=0.75)
    assert set(results["split_id"]) == {"seed0", "seed1"} and set(results["split_type"]) == {ID_RANDOM}
    assert set(models) == {"linear", "rf"} and models["rf"].predict(X).shape == (len(X),)
