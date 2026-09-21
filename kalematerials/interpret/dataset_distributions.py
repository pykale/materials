"""Target distributions for the full dataset and element-containing subsets."""

from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from kalematerials.prepdata.composition import formula_contains_elements, get_compound_radix


def _subsets_by_element(
    data: pd.DataFrame,
    elements: Sequence[str],
    formula_column: str,
) -> Dict[str, pd.DataFrame]:
    """{element: rows whose formula contains it}; the subsets may overlap."""
    return {
        element: data[formula_contains_elements(data, [element], formula_column=formula_column)] for element in elements
    }


def plot_target_distribution_by_element(
    data: pd.DataFrame,
    *,
    target_column: str,
    elements: Sequence[str],
    formula_column: str = "chemical formula",
    bins="auto",
    save_path=None,
) -> None:
    """Overlay target histograms for the full dataset and each element-containing subset.

    Subsets may overlap and share `bins`. Save to `save_path` when set; otherwise show the figure.
    """
    subsets = _subsets_by_element(data, elements, formula_column)

    plt.figure(figsize=(8, 6))
    for label, values in [("all", data[target_column])] + [(e, s[target_column]) for e, s in subsets.items()]:
        sns.histplot(x=values, kde=True, bins=bins, label=label)

    plt.xlabel(target_column)
    plt.ylabel("Count")
    plt.legend(title="Element", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_target_violin_by_element(
    data: pd.DataFrame,
    *,
    target_column: str,
    elements: Sequence[str],
    formula_column: str = "chemical formula",
    target_label: Optional[str] = None,
    title: str = "Violin Plot",
    save_path=None,
) -> None:
    """Plot target violins for the full dataset and each element-containing subset.

    Use `target_column` as the label when `target_label` is None.
    Save to `save_path` when set; otherwise show the figure.
    """
    subsets = _subsets_by_element(data, elements, formula_column)

    plot_data = pd.concat(
        [data[target_column].rename("all")]
        + [subset[target_column].rename(element) for element, subset in subsets.items()],
        axis=1,
    )

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_data, inner="quartile")
    plt.xlabel("Element", fontsize=16)
    plt.ylabel(target_label or target_column, fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def count_compounds_by_radix(data, formula_column: str = "chemical formula") -> pd.Series:
    """Count compounds by number of distinct elements, in ascending order."""
    return get_compound_radix(data, formula_column=formula_column).value_counts().sort_index()
