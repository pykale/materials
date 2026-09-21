"""Run an experiment with Hydra configuration overrides.

    python main.py dataset=novamag mode=ood
    python main.py dataset=materials_project mode=cross_validation tuning.enabled=false kfold.seeds=[0]

`dataset` and `mode` select config files; other arguments override individual settings.
"""

import os
from typing import cast

# Configure headless plotting before SHAP imports cv2 to avoid Qt plugin conflicts in subprocesses.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import hydra  # noqa: E402
import runner
from config import RunConfig, validate
from omegaconf import DictConfig, OmegaConf

from kalematerials.predict.sklearn_models import MODEL_REGISTRY


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(composed: DictConfig) -> None:
    """Run the configured evaluation or prediction job."""
    cfg = cast(RunConfig, OmegaConf.to_object(composed))
    validate(cfg)

    if cfg.evaluation_mode == "predict":
        runner.run_predict(cfg=cfg, registry=MODEL_REGISTRY)
        return

    if cfg.evaluation_mode == "cross_validation":
        runner.run_cross_validation(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "ood":
        runner.run_ood(cfg=cfg, registry=MODEL_REGISTRY)
    elif cfg.evaluation_mode == "uncertainty":
        runner.run_uncertainty(cfg=cfg, registry=MODEL_REGISTRY)
    else:
        runner.run_random_split(cfg=cfg, registry=MODEL_REGISTRY)

    if cfg.visualize.enabled:
        runner.run_data_visualization(cfg=cfg)


if __name__ == "__main__":
    main()
