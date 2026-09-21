"""Model specifications and their registry lookup."""

from dataclasses import dataclass
from typing import Callable, Collection, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelSpec:
    """Model registry entry with a config key, display name and training/search callables.

    `train(X, y, hyperparams=None, random_state=0)` returns a fitted estimator.
    `tune(X, y, cv_folds, random_state, n_iter)` returns hyperparameters; None disables search.
    """

    key: str
    name: str
    train: Callable
    tune: Optional[Callable] = None


def resolve_models(registry: Mapping[str, ModelSpec], keys: Sequence[str]) -> Tuple[ModelSpec, ...]:
    """Resolve model keys in order; raise ValueError for unknown keys."""
    unknown = [key for key in keys if key not in registry]
    if unknown:
        raise ValueError(f"Unknown model key(s): {sorted(unknown)}. Available: {sorted(registry)}")
    return tuple(registry[key] for key in keys)


def resolve_trained_models(
    registry: Mapping[str, ModelSpec],
    trained: Collection[str],
    keys: Sequence[str],
) -> Tuple[ModelSpec, ...]:
    """Resolve keys present in both `registry` and `trained`; raise ValueError for missing keys."""
    specs = resolve_models(registry, keys)
    missing = [spec.key for spec in specs if spec.key not in trained]
    if missing:
        raise ValueError(f"{missing} named for a figure, but this run fitted {sorted(trained)}")
    return specs
