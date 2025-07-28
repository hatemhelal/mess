import numpy as np
from functools import cache
from mess.types import FloatN
from periodictable import elements

"""
Bragg-Slater atomic radii in Angstroms.

References
----------
The values are parsed from Table I of {cite}`slater1964atomic` and are referred to as
the Bragg-Slater radii in {cite}`becke1988multicenter`
"""
# fmt:off
BRAGG_SLATER = np.array([
    # Placeholder at index 0
    np.nan,

    # Period 1 (H - He)
    0.25, np.nan,

    # Period 2 (Li - Ne)
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, np.nan,

    # Period 3 (Na - Ar)
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, np.nan,

    # Period 4 (K - Kr)
    2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35,
    1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, np.nan,

    # Period 5 (Rb - Xe)
    2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40,
    1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, np.nan,

    # Period 6 (Cs - Rn)
    2.60, 2.15, 1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85, 1.80,  # Lanthanides
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.55, 1.45, 1.35,
    1.35, 1.30, 1.35, 1.35, 1.35, 1.50, 1.90, 1.80, 1.60, 1.90,
    np.nan, np.nan,

    # Period 7 (Fr - Og)
    np.nan, 2.15, 1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75, np.nan,  # Actinides
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
])
# fmt:on


@cache
def covalent_radii() -> FloatN:
    """Covalent radii in Angstroms.

    References
    ----------
    The values are parsed from `periodictable.elements` which in turn uses data from
    {cite}`cordero2008covalent`.
    """

    R = np.float64([e.covalent_radius for e in elements])
    R = np.insert(R, 0, np.nan)
    return R
