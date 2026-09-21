"""Case-study alloy series: FeAl, FeCo and FeCr, with measured saturation magnetization in tesla."""

from typing import Tuple

from kalematerials.interpret.alloy_series import AlloySeries

FEAL_FORMULAS = [
    "Fe98Al2",
    "Fe96Al4",
    "Fe94Al6",
    "Fe93Al7",
    "Fe90Al10",
    "Fe88Al12",
    "Fe85Al15",
    "Fe82Al18",
    "Fe78Al22",
    "Fe75Al25",
    "Fe72Al28",
    "Fe68Al32",
    "Fe66Al34",
]
# Al atomic fraction -> saturation magnetization [T]
FEAL_LITERATURE_MS = {
    0.02: 2.14,
    0.038: 2.12,
    0.058: 2.09,
    0.074: 2.05,
    0.099: 2.01,
    0.124: 1.98,
    0.152: 1.92,
    0.183: 1.86,
    0.219: 1.80,
    0.251: 1.75,
    0.281: 1.69,
    0.315: 1.65,
    0.341: 1.60,
}

FECO_FORMULAS = [
    "Fe100Co0",
    "Fe96Co4",
    "Fe92Co8",
    "Fe90Co10",
    "Fe88Co12",
    "Fe85Co15",
    "Fe82Co18",
    "Fe79Co21",
    "Fe71Co29",
    "Fe59Co41",
    "Fe45Co55",
    "Fe26Co74",
    "Fe7Co93",
]
# Co atomic fraction -> saturation magnetization [T]
FECO_LITERATURE_MS = {
    0.00: 2.18,
    0.04: 2.21,
    0.08: 2.24,
    0.10: 2.26,
    0.12: 2.30,
    0.15: 2.33,
    0.18: 2.36,
    0.21: 2.39,
    0.29: 2.43,
    0.41: 2.44,
    0.55: 2.31,
    0.74: 2.09,
    0.93: 1.8,
}

FECR_FORMULAS = [
    "Fe99Cr1",
    "Fe98Cr2",
    "Fe96Cr4",
    "Fe95Cr5",
    "Fe93Cr7",
    "Fe92Cr8",
    "Fe91Cr9",
    "Fe90Cr10",
    "Fe89Cr11",
    "Fe87Cr13",
    "Fe85Cr15",
    "Fe83Cr17",
    "Fe80Cr20",
]
# Cr atomic fraction -> saturation magnetization [T]
FECR_LITERATURE_MS = {
    0.01: 2.14,
    0.02: 2.09,
    0.04: 2.05,
    0.05: 2.00,
    0.07: 1.96,
    0.08: 1.92,
    0.09: 1.89,
    0.10: 1.86,
    0.11: 1.83,
    0.13: 1.78,
    0.15: 1.73,
    0.17: 1.66,
    0.20: 1.60,
}


CASE_STUDIES: Tuple[AlloySeries, ...] = (
    AlloySeries("FeAl", "Al", FEAL_FORMULAS, FEAL_LITERATURE_MS),
    AlloySeries("FeCo", "Co", FECO_FORMULAS, FECO_LITERATURE_MS),
    AlloySeries("FeCr", "Cr", FECR_FORMULAS, FECR_LITERATURE_MS),
)
