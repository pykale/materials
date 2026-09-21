# KaleMaterials

**Standardizing Multimodal AI for Materials Prediction**

[![tests](https://github.com/pykale/materials/actions/workflows/test.yml/badge.svg)](https://github.com/pykale/materials/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/pykale/materials/branch/main/graph/badge.svg)](https://codecov.io/gh/pykale/materials)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

KaleMaterials is a machine learning library for predicting materials properties, following [PyKale](https://github.com/pykale/pykale)'s pipeline design.

It currently supports composition-based prediction, in-distribution and out-of-distribution evaluation, and uncertainty estimation. The included example predicts saturation magnetization for soft magnetic materials.

Support for structure, images, text, tabular characterization data and multimodal learning is planned.

## Installation

From the repository root, choose one command. Use the second to run the examples or develop the library.

```bash
pip install -e .          # library
pip install -e ".[dev]"   # plus examples and development tools
```

## Quick start

The example requires local raw data. Place the [Novamag](https://zenodo.org/records/3241267) JSON files under `examples/saturation_magnetism_prediction/data/novamag/Novamag_Data_Files/`, then build features and evaluate the models:

```bash
cd examples/saturation_magnetism_prediction
python build_feature_tables.py --dataset novamag
python main.py dataset=novamag mode=cross_validation
```

Choose a dataset and evaluation mode:

- `dataset`: `novamag` or `materials_project`
- `mode`: `cross_validation`, `random_split`, `ood`, `uncertainty` or `predict`

Override other settings in the same command, for example `tuning.enabled=false`.

## Package structure

Packages under `kalematerials/`, following PyKale's six-step pipeline:

- `loaddata`: datasets, element reference tables, train/test splits
- `prepdata`: formula parsing, stoichiometry, atomic fractions
- `embed`: composition descriptors
- `predict`: regression models, prediction intervals
- `evaluate`: metrics, significance tests, calibration
- `interpret`: model explanations, dataset distributions, case-study plots

The `pipeline` package combines these stages into feature-building and evaluation workflows.

## License

KaleMaterials is released under the MIT License. See [LICENSE](LICENSE) for details.
