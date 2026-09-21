"""Build in-distribution and out-of-distribution train/test splits.

Each builder returns (split_id, train_idx, test_idx) tuples with positional row indices.
"""

import warnings
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# (split_id, train row positions, test row positions)
Split = Tuple[str, np.ndarray, np.ndarray]

# Values for the result table's `split_type` column.
OOD = "OOD"  # test set held out by chemistry or geometry
ID_REFERENCE = "ID-reference"  # held-out inner fold of an OOD split's own training rows
ID_RANDOM = "ID-random"  # a random split of the same sizes as an OOD split
ID_KFOLD = "ID-kfold"  # plain K-fold over the whole pool


def _standardized(X: pd.DataFrame) -> np.ndarray:
    """Return X as a z-scored array."""
    return StandardScaler().fit_transform(X.to_numpy())


def _finalize_split(
    split_id: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    min_train: int = 1,
    min_test: int = 1,
) -> Optional[Split]:
    """Return (split_id, train_idx, test_idx), or None if either side is below its minimum size."""
    train_idx = np.asarray(train_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if train_idx.size < min_train or test_idx.size < min_test:
        return None
    return split_id, train_idx, test_idx


# --- In-distribution references ---


def build_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    seed: int = 0,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """K-fold splits, ids "ID_fold<k>". Empty if there are fewer samples than folds."""
    if n_samples < n_splits:
        return []
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed if shuffle else None)

    splits: List[Split] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        split = _finalize_split(f"ID_fold{fold}", train_idx, test_idx, min_train, min_test)
        if split is not None:
            splits.append(split)
    return splits


def build_random_split(
    n_samples: int,
    train_size: float = 0.8,
    seed: int = 0,
    split_id: str = "random_split",
) -> Split:
    """One random train/test split."""
    train_idx, test_idx = train_test_split(np.arange(n_samples), train_size=train_size, random_state=seed)
    return split_id, np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def build_size_matched_split(
    n_samples: int,
    n_train: int,
    n_test: int,
    seed: int,
    split_id: str = "RandomControl",
) -> Optional[Split]:
    """A random disjoint train/test split with the given sizes, or None if they do not fit in `n_samples`."""
    if n_train + n_test > n_samples:
        return None
    order = np.random.default_rng(seed).permutation(n_samples)
    return _finalize_split(split_id, order[:n_train], order[n_train : n_train + n_test])


def build_size_matched_controls(
    splits: Sequence[Split],
    n_samples: int,
    seed: int,
    scenario: str,
) -> Dict[str, Split]:
    """Return {split_id: random control} with matching train/test sizes; omit splits that exceed `n_samples`."""
    controls: Dict[str, Split] = {}
    for split_id, train_idx, test_idx in splits:
        control = build_size_matched_split(
            n_samples,
            len(train_idx),
            len(test_idx),
            # crc32 gives stable seeds across processes.
            seed=zlib.crc32(f"{seed}|{scenario}|{split_id}".encode()),
            split_id=split_id,
        )
        if control is None:
            warnings.warn(f"No size-matched control fits for {scenario} {split_id}.")
        else:
            controls[split_id] = control
    return controls


# --- Held-out chemistry ---


def build_loeo_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_list: Sequence[str],
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-one-element-out: test = samples containing the element, ids "E=<symbol>"."""
    splits: List[Split] = []
    for element in element_list:
        test_mask = np.array([element in set(els) for els in elements_per_sample], dtype=bool)
        split = _finalize_split(
            f"E={element}",
            np.where(~test_mask)[0],
            np.where(test_mask)[0],
            min_train=min_train,
            min_test=min_test,
        )
        if split is not None:
            splits.append(split)
    return splits


def build_system_splits(
    elements_per_sample: Sequence[Sequence[str]],
    systems: Sequence[Sequence[str]],
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-one-system-out: test = samples containing every element of the system, ids "S=Fe-Co"."""
    splits: List[Split] = []
    for system in systems:
        wanted = set(system)
        test_mask = np.array([wanted <= set(els) for els in elements_per_sample], dtype=bool)
        split = _finalize_split(
            f"S={'-'.join(system)}",
            np.where(~test_mask)[0],
            np.where(test_mask)[0],
            min_train=min_train,
            min_test=min_test,
        )
        if split is not None:
            splits.append(split)
    return splits


def _build_membership_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_attr: Dict[str, int],
    attr_values: Sequence[int],
    label: str,
    strict: bool,
    min_train: int,
    min_test: int,
) -> List[Split]:
    """Hold out samples by an element attribute: any element has the value (default) or all elements do (strict)."""
    splits: List[Split] = []
    for value in attr_values:
        if strict:
            test_mask = np.array(
                [bool(els) and all(element_to_attr.get(e) == value for e in set(els)) for els in elements_per_sample],
                dtype=bool,
            )
        else:
            heldout_elements = {e for e, v in element_to_attr.items() if v == value}
            test_mask = np.array(
                [len(set(els).intersection(heldout_elements)) > 0 for els in elements_per_sample],
                dtype=bool,
            )
        split = _finalize_split(
            f"{label}={value}",
            np.where(~test_mask)[0],
            np.where(test_mask)[0],
            min_train=min_train,
            min_test=min_test,
        )
        if split is not None:
            splits.append(split)
    return splits


def build_period_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_period: Dict[str, int],
    periods: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Hold out a period (ids "P=<period>"); `strict` requires every element in a sample to belong to it."""
    return _build_membership_splits(elements_per_sample, element_to_period, periods, "P", strict, min_train, min_test)


def build_group_splits(
    elements_per_sample: Sequence[Sequence[str]],
    element_to_group: Dict[str, int],
    groups: Sequence[int],
    strict: bool = False,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Hold out a group (ids "G=<group>"); `strict` requires every element in a sample to belong to it."""
    return _build_membership_splits(elements_per_sample, element_to_group, groups, "G", strict, min_train, min_test)


# --- Held-out regions of feature or target space ---


def build_kmeans_cluster_splits(
    X: pd.DataFrame,
    k: int = 10,
    seed: int = 0,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Leave-one-cluster-out: KMeans on standardized X, test = one cluster, ids "C=<cluster>"."""
    if k < 2:
        raise ValueError("k must be >= 2")

    labels = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(_standardized(X))

    splits: List[Split] = []
    for c in range(k):
        split = _finalize_split(
            f"C={c}",
            np.where(labels != c)[0],
            np.where(labels == c)[0],
            min_train=min_train,
            min_test=min_test,
        )
        if split is not None:
            splits.append(split)
    return splits


def build_sparsex_splits(
    X: pd.DataFrame,
    fractions: Sequence[float] = (0.1, 0.2),
    n_neighbors: int = 5,
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Hold out the most isolated samples in standardized feature space, one split per fraction.

    Isolation is the mean distance to the `n_neighbors` nearest neighbours. Ids are "SparseX_top<pct>pct".
    """
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1")

    X_mat = _standardized(X)
    n = X_mat.shape[0]
    if n < 2:
        raise ValueError("SparseX requires at least 2 samples")

    # Request an extra neighbour, then exclude the query point itself.
    nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, n)).fit(X_mat)
    distances, _ = nn.kneighbors(X_mat)
    mean_dist = distances[:, 1:].mean(axis=1) if distances.shape[1] > 1 else distances[:, 0]

    order = np.argsort(mean_dist)[::-1]
    return _split_by_fraction(order, "SparseX", fractions, min_train, min_test)


def build_sparsey_splits(
    y: pd.Series,
    fractions: Sequence[float] = (0.1, 0.2),
    center: str = "median",
    min_train: int = 1,
    min_test: int = 1,
) -> List[Split]:
    """Hold out the samples whose target is farthest from the `center` ("median" or "mean"), one split per fraction.

    Ids are "SparseY_top<pct>pct".
    """
    y_arr = np.asarray(y, dtype=float)
    if center == "median":
        ref = np.median(y_arr)
    elif center == "mean":
        ref = np.mean(y_arr)
    else:
        raise ValueError("center must be 'median' or 'mean'")

    order = np.argsort(np.abs(y_arr - ref))[::-1]
    return _split_by_fraction(order, "SparseY", fractions, min_train, min_test)


def _split_by_fraction(
    order: np.ndarray,
    label: str,
    fractions: Sequence[float],
    min_train: int,
    min_test: int,
) -> List[Split]:
    """Hold out the first `fraction` of `order` (most extreme first) for each fraction."""
    n = order.size
    splits: List[Split] = []
    for frac in fractions:
        if not (0 < float(frac) < 1):
            raise ValueError(f"Each fraction must be in (0, 1), got {frac}")
        n_test = max(1, int(round(n * float(frac))))
        split = _finalize_split(
            f"{label}_top{int(round(frac * 100))}pct",
            np.sort(order[n_test:]),
            np.sort(order[:n_test]),
            min_train=min_train,
            min_test=min_test,
        )
        if split is not None:
            splits.append(split)
    return splits
