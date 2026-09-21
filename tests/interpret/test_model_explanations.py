import pytest

from kalematerials.interpret.model_explanations import plot_model_explanations
from kalematerials.predict.sklearn_models import MODEL_REGISTRY


def test_explanations_for_tree_and_pipeline_models(tmp_path, toy_xy, specs):
    X, y = toy_xy
    models = {"Random Forest": specs[1].train(X, y), "Ridge": MODEL_REGISTRY["ridge"].train(X, y)}
    plot_model_explanations(models, X, X.iloc[:10], y.iloc[:10], save_dir=tmp_path / "plots", prefix="t_")
    written = {p.name for p in (tmp_path / "plots").iterdir()}
    assert written == {
        f"t_{kind}_{slug}.png" for kind in ("perm_importance", "shap_summary") for slug in ("random_forest", "ridge")
    }
    with pytest.raises(ValueError):
        plot_model_explanations({}, X, X, y, save_dir=tmp_path)
