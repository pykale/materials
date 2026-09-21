import joblib
import pandas as pd
import pytest

from kalematerials.evaluate.metrics import build_result_rows
from kalematerials.utils import persistence as p


def bundle(X, y):
    return p.ModelBundle(
        None,
        "rf",
        "Random Forest",
        tuple(X.columns),
        "target",
        "toy",
        len(X),
        {p.TRAINING_FINGERPRINT: p.fingerprint_training_data(X, y)},
    )


def test_fingerprint(toy_xy):
    X, y = toy_xy
    digest = p.fingerprint_training_data(X, y)
    assert len(digest) == 16 and digest == p.fingerprint_training_data(X.copy(), y)
    assert digest not in {
        p.fingerprint_training_data(X + 1, y),
        p.fingerprint_training_data(X[["b", "a", "c"]], y),
        p.fingerprint_training_data(X.rename(columns={"a": "z"}), y),
    }


def test_bundle_roundtrip(tmp_path, toy_xy):
    X, y = toy_xy
    path = p.save_model_bundle(tmp_path / "m" / "b.joblib", bundle(X, y))
    loaded = p.load_model_bundle(path)
    assert (
        loaded.feature_columns == ("a", "b", "c")
        and "Random Forest (rf) trained on 40 row(s) of toy" in loaded.describe()
    )
    with pytest.raises(FileNotFoundError):
        p.load_model_bundle(tmp_path / "none.joblib")
    joblib.dump({"not": "a bundle"}, tmp_path / "x.joblib")
    with pytest.raises(ValueError, match="ModelBundle"):
        p.load_model_bundle(tmp_path / "x.joblib")
    newer = bundle(X, y)
    newer.format_version = p.BUNDLE_FORMAT_VERSION + 1
    joblib.dump(newer, tmp_path / "new.joblib")
    with pytest.raises(ValueError, match="format"):
        p.load_model_bundle(tmp_path / "new.joblib")


def test_align_features(toy_xy):
    X, _ = toy_xy
    assert list(p.align_features(X, ["c", "a"]).columns) == ["c", "a"]
    with pytest.raises(ValueError, match="in the input"):
        p.align_features(X, ["a", "zz"], source="the input")


def test_save_tables_and_results(tmp_path):
    rows = build_result_rows({"A": {"mse": [1.0, 3.0]}}, scenario="cv", split_type="ID", split_ids=["f0", "f1"], seed=0)
    summary, directory = p.save_results(rows, tmp_path / "out")
    assert (
        summary["mean"].tolist() == [2.0]
        and (directory / "results.csv").exists()
        and (directory / "summary.csv").exists()
    )
    assert p.save_results(rows, tmp_path / "skip", enabled=False)[1] is None and not (tmp_path / "skip").exists()
    written = p.save_tables({"a.csv": rows, "empty.csv": pd.DataFrame()}, tmp_path / "t")
    assert {f.name for f in written.iterdir()} == {"a.csv"}
    assert p.save_tables({"a.csv": rows}, None) is None
