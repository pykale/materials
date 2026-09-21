"""Check shipped configs and validation of invalid keys, types and values."""

import itertools
from typing import cast

import pytest
from config import EVALUATION_MODES, RunConfig, validate
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from ..conftest import EXAMPLE_ROOT

DATASETS = ("novamag", "materials_project")
MODES = ("cross_validation", "random_split", "ood", "uncertainty", "predict")


def composed(*overrides):
    with initialize_config_dir(version_base=None, config_dir=str(EXAMPLE_ROOT / "configs")):
        return cast(RunConfig, OmegaConf.to_object(compose(config_name="config", overrides=list(overrides))))


@pytest.mark.parametrize("dataset,mode", list(itertools.product(DATASETS, MODES)))
def test_every_dataset_and_mode_composes(dataset, mode):
    cfg = composed(f"dataset={dataset}", f"mode={mode}")
    validate(cfg)
    assert cfg.evaluation_mode == mode and (mode in ("predict", "uncertainty") or cfg.models)
    assert cfg.file_prefix == dataset


def test_modes_match_schema():
    assert set(MODES) == set(EVALUATION_MODES)
    assert composed("dataset=novamag").dataset_path != composed("dataset=materials_project").dataset_path


def test_bad_configs_are_rejected():
    with pytest.raises(ConfigCompositionException, match="ood.kk"):
        composed("ood.kk=5")
    with pytest.raises((ConfigCompositionException, ValidationError), match="kfold.folds"):
        composed("kfold.folds=ten")
    with pytest.raises(ValueError):
        validate(composed("ood.scenarios=[element,elemnt]"))
    with pytest.raises(ValueError):
        validate(composed("compare_models=[rf]"))
    with pytest.raises(ValueError):
        validate(composed("evaluation_mode=nope"))


def test_overrides_reach_nested_fields():
    cfg = composed("mode=ood", "ood.n_clusters=3", "kfold.seeds=[7]")
    assert cfg.ood.n_clusters == 3 and list(cfg.kfold.seeds) == [7]
