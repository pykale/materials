"""Permutation-importance and SHAP summary plots."""

from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.inspection import permutation_importance


def plot_permutation_importance(
    model,
    X_valid,
    y_valid,
    title: str = "",
    save_path: Optional[str] = None,
    random_state: int = 0,
):
    """Bar chart of permutation importance on the validation rows."""
    perm_import = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=random_state)

    sorted_idx = perm_import.importances_mean.argsort()

    plt.figure(figsize=(14, 7))
    plt.barh(range(len(sorted_idx)), perm_import.importances_mean[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), X_valid.columns[sorted_idx], fontsize=16)
    plt.xlabel("Permutation Feature Importance", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(fontsize=16)
    if title:
        plt.title(title, fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def _build_explainer(model, X_train):
    """Return a SHAP explainer and call kwargs; use a model-agnostic explainer for pipelines."""
    try:
        return shap.Explainer(model, X_train), {"check_additivity": False}
    except TypeError:
        return shap.Explainer(model.predict, X_train), {}


def plot_shap_summary(model, X_train, X_valid, save_path: Optional[str] = None, random_state: int = 0):
    """Plot SHAP values for `X_valid`, using `X_train` as the background distribution.

    `random_state` seeds beeswarm jitter. Save to `save_path` when set; otherwise show the figure.
    """
    explainer, options = _build_explainer(model, X_train)
    shap_values = explainer(X_valid, **options)

    shap.summary_plot(
        shap_values,
        X_valid,
        feature_names=X_valid.columns,
        show=False,
        rng=np.random.default_rng(random_state),
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_model_explanations(
    models: Mapping[str, Any],
    X_train,
    X_valid,
    y_valid,
    *,
    save_dir: Path,
    prefix: str = "",
    random_state: int = 0,
) -> None:
    """Save permutation-importance and SHAP figures for each model.

    `models` maps display names to estimators; lowercase names and `prefix` form filenames in `save_dir`.
    Explain validation rows using `X_train` as the SHAP background. `random_state` seeds shuffling and jitter.
    """
    if not models:
        raise ValueError("plot_model_explanations needs at least one fitted model.")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        _plot_one_model(
            model, X_train, X_valid, y_valid, name=name, save_dir=save_dir, prefix=prefix, random_state=random_state
        )


def _plot_one_model(model, X_train, X_valid, y_valid, *, name, save_dir, prefix, random_state) -> None:
    """Write both figures for a single model."""
    slug = name.lower().replace(" ", "_")

    plot_permutation_importance(
        model,
        X_valid,
        y_valid,
        title=f"{name} Permutation Importance",
        save_path=save_dir / f"{prefix}perm_importance_{slug}.png",
        random_state=random_state,
    )
    plot_shap_summary(
        model,
        X_train,
        X_valid,
        save_path=save_dir / f"{prefix}shap_summary_{slug}.png",
        random_state=random_state,
    )
