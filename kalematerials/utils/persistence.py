"""Save and load model bundles and result tables."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from kalematerials.evaluate.metrics import summarize_scores, SUMMARY_GROUPS

# Increment for incompatible bundle-format changes.
BUNDLE_FORMAT_VERSION = 1

TRAINING_FINGERPRINT = "training_fingerprint"


def fingerprint_training_data(X: pd.DataFrame, y: pd.Series) -> str:
    """Return a 16-character digest of ordered feature names, feature values and targets."""
    digest = hashlib.sha256()
    digest.update("\x00".join(str(column) for column in X.columns).encode("utf-8"))
    digest.update(np.ascontiguousarray(X.to_numpy(dtype=float)).tobytes())
    digest.update(np.ascontiguousarray(y.to_numpy(dtype=float)).tobytes())
    return digest.hexdigest()[:16]


@dataclass
class ModelBundle:
    """Fitted estimator with feature order, target and training metadata.

    `model_key` identifies the registry entry; `model_name` is its display name.
    `feature_columns` follows training order. `format_version` identifies the saved bundle layout.
    """

    model: Any
    model_key: str
    model_name: str
    feature_columns: Tuple[str, ...]
    target_column: str
    dataset: str = ""
    n_train: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    format_version: int = BUNDLE_FORMAT_VERSION

    def describe(self) -> str:
        """One-line summary for console output."""
        return (
            f"{self.model_name} ({self.model_key}) trained on {self.n_train} row(s) "
            f"of {self.dataset or 'an unnamed dataset'}, "
            f"predicting {self.target_column!r} from {len(self.feature_columns)} feature(s)"
        )


def save_model_bundle(path: str, bundle: ModelBundle) -> Path:
    """Write `bundle` to `path` with joblib, creating parent directories; returns the resolved path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination.resolve()


def load_model_bundle(path: str) -> ModelBundle:
    """Load a ModelBundle, rejecting other objects and newer bundle formats."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No saved model at {source}")

    bundle = joblib.load(source)
    if not isinstance(bundle, ModelBundle):
        raise ValueError(f"{source} does not contain a ModelBundle (got {type(bundle).__name__}).")
    if bundle.format_version > BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"{source} was written in bundle format v{bundle.format_version}, but this "
            f"version of the code understands at most v{BUNDLE_FORMAT_VERSION}."
        )
    return bundle


def align_features(features, feature_columns: Sequence[str], source: Optional[str] = None):
    """Select features in training-column order; `source` labels errors for missing columns."""
    missing = [c for c in feature_columns if c not in features.columns]
    if missing:
        where = f" in {source}" if source else ""
        raise ValueError(f"Missing feature column(s){where}: {missing}. The model expects {list(feature_columns)}.")
    return features[list(feature_columns)]


RESULTS_FILENAME = "results.csv"
SUMMARY_FILENAME = "summary.csv"


def save_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Optional[str],
    *,
    enabled: bool = True,
) -> Optional[Path]:
    """Save nonempty tables from {filename: table} to `output_dir`.

    Return the output directory, or None if saving is disabled or no directory is set.
    """
    if not enabled or not output_dir:
        return None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        if not table.empty:
            table.to_csv(directory / filename, index=False)
    return directory


def save_results(
    results: pd.DataFrame,
    output_dir: Optional[str],
    *,
    by: Sequence[str] = SUMMARY_GROUPS,
    enabled: bool = True,
) -> Tuple[pd.DataFrame, Optional[Path]]:
    """Summarize long-form results by `by` and save results.csv and summary.csv.

    Return (summary, directory). If saving is disabled or no directory is set, return (summary, None).
    """
    summary = summarize_scores(results, by=by)

    if not enabled or not output_dir:
        return summary, None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / RESULTS_FILENAME, index=False)
    summary.to_csv(directory / SUMMARY_FILENAME, index=False)
    return summary, directory
