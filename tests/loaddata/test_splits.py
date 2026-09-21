import numpy as np
import pandas as pd
import pytest

from kalematerials.loaddata import splits as s

ELEMENTS = [["Fe", "Co"], ["Fe", "Ni"], ["Co"], ["Ni"], ["Fe", "Co", "Ni"], ["Fe"]]
GROUP, PERIOD = {"Fe": 8, "Co": 9, "Ni": 10}, {"Fe": 4, "Co": 5, "Ni": 4}


def sides(split):
    _, train, test = split
    return sorted(train), sorted(test)


def test_kfold_and_random():
    folds = s.build_kfold_splits(10, 5, seed=0)
    assert [split_id for split_id, _, _ in folds] == [f"ID_fold{i}" for i in range(5)]
    assert all(len(t) == 8 and len(v) == 2 for _, t, v in folds)
    assert s.build_kfold_splits(3, 5) == []
    _, train, test = s.build_random_split(10, 0.7, seed=1)
    assert len(train) == 7 and not set(train) & set(test)


def test_size_matched():
    assert s.build_size_matched_split(5, 4, 2, seed=0) is None
    _, train, test = s.build_size_matched_split(10, 4, 3, seed=0)
    assert len(train) == 4 and len(test) == 3 and not set(train) & set(test)
    parent = [("E=Fe", np.arange(6), np.arange(6, 8)), ("E=Big", np.arange(9), np.arange(9, 12))]
    with pytest.warns(UserWarning, match="E=Big"):
        controls = s.build_size_matched_controls(parent, 10, seed=0, scenario="LOEO")
    assert list(controls) == ["E=Fe"]
    again = s.build_size_matched_controls(parent[:1], 10, seed=0, scenario="LOEO")["E=Fe"]
    assert np.array_equal(controls["E=Fe"][1], again[1])  # Split IDs give stable seeds.


def test_element_and_system():
    (split,) = s.build_loeo_splits(ELEMENTS, ["Ni"])
    assert split[0] == "E=Ni" and sides(split) == ([0, 2, 5], [1, 3, 4])
    (split,) = s.build_system_splits(ELEMENTS, [("Fe", "Co")])
    assert split[0] == "S=Fe-Co" and sides(split)[1] == [0, 4]
    assert s.build_loeo_splits(ELEMENTS, ["Ni"], min_test=4) == []


def test_period_and_group():
    (any_split,) = s.build_period_splits(ELEMENTS, PERIOD, [5])
    (strict,) = s.build_period_splits(ELEMENTS, PERIOD, [5], strict=True)
    assert sides(any_split)[1] == [0, 2, 4] and sides(strict)[1] == [2]
    (split,) = s.build_group_splits(ELEMENTS, GROUP, [10])
    assert split[0] == "G=10" and sides(split)[1] == [1, 3, 4]


def test_feature_and_target_space():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 2)))
    X.iloc[0] = 100
    ids = [split_id for split_id, _, _ in s.build_kmeans_cluster_splits(X, k=3)]
    assert ids == ["C=0", "C=1", "C=2"]
    (split,) = s.build_sparsex_splits(X, fractions=(0.1,), n_neighbors=2)
    assert split[0] == "SparseX_top10pct" and 0 in split[2] and len(split[2]) == 2
    y = pd.Series(np.arange(20.0))
    (split,) = s.build_sparsey_splits(y, fractions=(0.1,))
    assert sorted(split[2]) == [0, 19]
    for call in (
        lambda: s.build_kmeans_cluster_splits(X, k=1),
        lambda: s.build_sparsex_splits(X, n_neighbors=0),
        lambda: s.build_sparsex_splits(X, fractions=(1.5,)),
        lambda: s.build_sparsey_splits(y, center="mode"),
    ):
        with pytest.raises(ValueError):
            call()
