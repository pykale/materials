"""Run configuration registered with Hydra as "schema"."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from hydra.core.config_store import ConfigStore

EVALUATION_MODES = ("predict", "random_split", "cross_validation", "ood", "uncertainty")

OOD_SCENARIOS = ("element", "period", "group", "system", "cluster", "sparsex", "sparsey")


@dataclass
class KFoldConfig:
    """K-fold cross-validation."""

    folds: int = 5
    shuffle: bool = True
    seeds: Tuple[int, ...] = (0,)


@dataclass
class RandomSplitConfig:
    """Random train/validation splits, one per seed."""

    seeds: Tuple[int, ...] = (0,)
    train_size: float = 0.8


@dataclass
class TuningConfig:
    """Nested hyperparameter search."""

    enabled: bool = False
    cv_folds: int = 3
    n_iter: int = 20


@dataclass
class InterpretConfig:
    """Model-explanation and case-study figures (random_split mode)."""

    enabled: bool = False
    explain_models: Tuple[str, ...] = ()
    case_study_models: Tuple[str, ...] = ()


@dataclass
class OODConfig:
    """Out-of-distribution split settings; see pipeline.ood.build_scenarios.

    `systems` uses labels such as "Fe-Co". `size_matched_control` adds a random control per OOD split.
    """

    scenarios: Tuple[str, ...] = OOD_SCENARIOS
    n_clusters: int = 10
    cluster_seed: int = 0
    seeds: Tuple[int, ...] = (0,)
    max_splits: Optional[int] = None
    elements: Optional[Tuple[str, ...]] = None
    periods: Optional[Tuple[int, ...]] = None
    groups: Optional[Tuple[int, ...]] = None
    systems: Tuple[str, ...] = ()
    fractions: Tuple[float, ...] = (0.1, 0.2)
    sparsex_neighbors: int = 5
    sparsey_center: str = "median"
    min_test: int = 20
    min_train: int = 50
    period_strict: bool = False
    group_strict: bool = False
    size_matched_control: bool = True


@dataclass
class UncertaintyConfig:
    """Prediction-interval calibration settings."""

    model: str = "rf"
    alpha: float = 0.05
    calibration_fraction: float = 0.25
    seeds: Tuple[int, ...] = (0,)


@dataclass
class PredictConfig:
    """Predict formulas from `input_path` (CSV) or an inline `formulas` list.

    Load `model_path`, defaulting to an evaluation bundle under `model_dir`.
    `model` selects a model to fit when needed; `retrain` forces refitting.
    """

    model: Optional[str] = None
    model_path: Optional[str] = None
    retrain: bool = False
    input_path: Optional[str] = None
    formulas: Tuple[str, ...] = ()


@dataclass
class RunConfig:
    """Dataset, evaluation mode and settings for one run.

    `dataset_path` is a prepared feature table; `feature_columns` gives model input order.
    `models` and `compare_models` use registry keys. `model_random_state` seeds model fitting and tuning.
    Evaluation modes save the best model to `model_dir/<dataset>_<mode>.joblib`.
    Omitted reference-table paths use the packaged spreadsheets.
    """

    dataset: str
    dataset_path: str
    evaluation_mode: str
    feature_columns: Tuple[str, ...]
    periodic_table_path: Optional[str] = None
    miedema_path: Optional[str] = None
    model_dir: str = "./models"

    target_column: str = "saturation magnetization"
    formula_column: str = "chemical formula"

    models: Tuple[str, ...] = ()
    compare_models: Optional[Tuple[str, str]] = None

    model_random_state: int = 0
    plots_dir: str = "./plots"
    output_dir: str = "./results"
    save_results: bool = True
    print_results: bool = True

    kfold: KFoldConfig = field(default_factory=KFoldConfig)
    random_split: RandomSplitConfig = field(default_factory=RandomSplitConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    interpret: InterpretConfig = field(default_factory=InterpretConfig)
    ood: OODConfig = field(default_factory=OODConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    @property
    def file_prefix(self) -> str:
        """The dataset name as a filename prefix."""
        cleaned = "".join(c if c.isalnum() else "_" for c in self.dataset.lower())
        return cleaned.strip("_") or "dataset"


def validate(cfg: RunConfig) -> None:
    """Validate mode names, OOD scenarios and the model-comparison pair."""
    if cfg.evaluation_mode not in EVALUATION_MODES:
        raise ValueError(f"Invalid evaluation_mode {cfg.evaluation_mode!r}. Choose one of {sorted(EVALUATION_MODES)}.")
    unknown = [scenario for scenario in cfg.ood.scenarios if scenario not in OOD_SCENARIOS]
    if unknown:
        raise ValueError(f"Invalid ood.scenarios {unknown}. Choose from {list(OOD_SCENARIOS)}.")
    if cfg.compare_models is not None and len(cfg.compare_models) != 2:
        raise ValueError(f"compare_models must name exactly two models, got {list(cfg.compare_models)}.")


ConfigStore.instance().store(name="schema", node=RunConfig)
