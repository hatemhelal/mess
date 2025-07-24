"""Interoperation tools for working across MESS, PySCF."""

from typing import Tuple

import numpy as np
from periodictable import elements
from pyscf import gto

from mess.basis import Basis, basisset
from mess.structure import Structure
from mess.units import to_bohr
from mess.package_utils import requires_package


def to_pyscf(
    structure: Structure, basis_name: str = "sto-3g", spherical: bool = True
) -> "gto.Mole":
    """Convert a MESS Structure to a PySCF Mole object.

    Args:
        structure (Structure): The MESS Structure object to convert.
        basis_name (str, optional): The name of the basis set to use. Defaults to
            "sto-3g".
        spherical (bool, optional): Whether to use spherical Gaussian basis functions.
            Defaults to ``True``.

    Returns:
        gto.Mole: The converted PySCF Mole object.
    """

    mol = gto.Mole(unit="Bohr", spin=structure.num_electrons % 2, cart=not spherical)
    mol.atom = [
        (symbol, pos)
        for symbol, pos in zip(structure.atomic_symbol, structure.position)
    ]
    mol.basis = basis_name
    mol.build(unit="Bohr")
    return mol


def from_pyscf(mol: "gto.Mole") -> Tuple[Structure, Basis]:
    """Convert a PySCF Mole object to a MESS Structure and Basis object.

    Args:
        mol (gto.Mole): The PySCF Mole object to convert.

    Returns:
        Tuple[Structure, Basis]: A tuple containing the converted MESS Structure
            and Basis objects.
    """

    atoms = [(elements.symbol(sym).number, pos) for sym, pos in mol.atom]
    atomic_number, position = [np.array(x) for x in zip(*atoms)]

    if mol.unit == "Angstrom":
        position = to_bohr(position)

    structure = Structure(atomic_number, position)

    basis = basisset(structure, basis_name=mol.basis, spherical=not mol.cart)

    return structure, basis


@requires_package("pyquante2")
def from_pyquante(name: str) -> Structure:
    """Load molecular structure from pyquante2.geo.samples module

    Args:
        name (str): Possible names include ch4, c6h6, aspirin, caffeine, hmx, petn,
                    prozan, rdx, taxol, tylenol, viagara, zoloft

    Returns:
        Structure
    """
    from pyquante2.geo import samples

    pqmol = getattr(samples, name)
    atomic_number, position = zip(*[(a.Z, a.r) for a in pqmol])
    atomic_number, position = [np.asarray(x) for x in (atomic_number, position)]
    return Structure(atomic_number, position)
