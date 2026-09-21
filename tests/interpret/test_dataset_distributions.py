import numpy as np
import pandas as pd

from kalematerials.interpret import dataset_distributions as d

from ..conftest import FORMULA, TARGET

DATA = pd.DataFrame({FORMULA: ["Fe", "FeCo", "Co", "FeCoNi"], TARGET: [2.0, 2.3, 1.7, 1.9]})


def test_plots_are_written(tmp_path):
    d.plot_target_distribution_by_element(
        DATA, target_column=TARGET, elements=["Fe", "Co"], bins=np.arange(1.5, 2.6, 0.5), save_path=tmp_path / "h.png"
    )
    d.plot_target_violin_by_element(DATA, target_column=TARGET, elements=["Fe", "Co"], save_path=tmp_path / "v.png")
    assert {p.name for p in tmp_path.iterdir()} == {"h.png", "v.png"}


def test_count_compounds_by_radix():
    assert d.count_compounds_by_radix(DATA).to_dict() == {1: 2, 2: 1, 3: 1}
