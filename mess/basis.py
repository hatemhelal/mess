"""basis sets of Gaussian type orbitals"""

from functools import cache
from typing import Literal, Tuple, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.ops import segment_sum

from mess.orbital import Orbital, batch_orbitals
from mess.primitive import Primitive
from mess.structure import Structure
from mess.types import (
    FloatN,
    FloatNx3,
    FloatNxM,
    FloatNxN,
    IntN,
    default_fptype,
)


class Basis(eqx.Module):
    orbitals: Tuple[Orbital]
    structure: Structure
    primitives: Primitive
    coefficients: FloatN
    orbital_index: IntN
    basis_name: str = eqx.field(static=True)
    max_L: int = eqx.field(static=True)

    @property
    def num_orbitals(self) -> int:
        return len(self.orbitals)

    @property
    def num_primitives(self) -> int:
        return sum(ao.num_primitives for ao in self.orbitals)

    @property
    def occupancy(self) -> FloatN:
        # Assumes uncharged systems in restricted Kohn-Sham
        occ = jnp.full(self.num_orbitals, 2.0)
        mask = occ.cumsum() > self.structure.num_electrons
        occ = jnp.where(mask, 0.0, occ)
        return occ

    def to_dataframe(self) -> pd.DataFrame:
        def fixer(x):
            # simple workaround for storing 2d array as a pandas column
            return [x[i, :] for i in range(x.shape[0])]

        df = pd.DataFrame()
        df["orbital"] = self.orbital_index
        df["atom"] = self.primitives.atom_index
        df["coefficient"] = self.coefficients
        df["norm"] = self.primitives.norm
        df["center"] = fixer(self.primitives.center)
        df["lmn"] = fixer(self.primitives.lmn)
        df["alpha"] = self.primitives.alpha
        df.index.name = "primitive"
        return df

    def density_matrix(self, C: FloatNxN) -> FloatNxN:
        """Evaluate the density matrix from the molecular orbital coefficients

        Args:
            C (FloatNxN): the molecular orbital coefficients

        Returns:
            FloatNxN: the density matrix.
        """
        return jnp.einsum("k,ik,jk->ij", self.occupancy, C, C)

    @jit
    def __call__(self, pos: FloatNx3) -> FloatNxM:
        prim = self.coefficients[jnp.newaxis, :] * self.primitives(pos)
        orb = segment_sum(prim.T, self.orbital_index, num_segments=self.num_orbitals)
        return orb.T

    def __repr__(self) -> str:
        return repr(self.to_dataframe())

    def _repr_html_(self) -> str | None:
        df = self.to_dataframe()
        return df._repr_html_()

    def __hash__(self) -> int:
        return hash(self.primitives)


def basisset(structure: Structure, basis_name: str = "sto-3g") -> Basis:
    """Factory function for building a basis set for a structure.

    Args:
        structure (Structure): Used to define the basis function parameters.
        basis_name (str, optional): Basis set name to look up on the
            `basis set exchange <https://www.basissetexchange.org/>`_.
            Defaults to ``sto-3g``.

    Returns:
        Basis constructed from inputs
    """
    orbitals = []
    atom_index = []

    for atom_id in range(structure.num_atoms):
        element = int(structure.atomic_number[atom_id])
        out = _bse_to_orbitals(basis_name, element)
        atom_index.extend([atom_id] * sum(len(ao.primitives) for ao in out))
        orbitals += out

    primitives, coefficients, orbital_index = batch_orbitals(orbitals)
    primitives = eqx.tree_at(lambda p: p.atom_index, primitives, jnp.array(atom_index))
    center = structure.position[primitives.atom_index, :]
    primitives = eqx.tree_at(lambda p: p.center, primitives, center)

    basis = Basis(
        orbitals=orbitals,
        structure=structure,
        primitives=primitives,
        coefficients=coefficients,
        orbital_index=orbital_index,
        basis_name=basis_name,
        max_L=int(np.max(primitives.lmn)),
    )

    # TODO(hh): this introduces some performance overhead into basis construction that
    # could be pushed down into the cached orbitals.
    basis = renorm(basis)
    return basis


# Mapping from L to Cartesian lmn angular momentum quantum numbers
# fmt: off
LMN_MAP = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)],
    3: [(3, 0, 0), (2, 1, 0), (2, 0, 1), (1, 2, 0), (1, 1, 1),
        (1, 0, 2), (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
}
# fmt: on


@cache
def _bse_to_orbitals(basis_name: str, atomic_number: int) -> Tuple[Orbital]:
    """
    Look up basis set parameters on the basis set exchange and build a tuple of Orbital.

    The output is cached to reuse the same objects for a given basis set and atomic
    number.  This can help save time when batching over different coordinates.

    Args:
        basis_name (str): The name of the basis set to lookup on the basis set exchange.
        atomic_number (int): The atomic number for the element to retrieve.

    Returns:
        Tuple[Orbital]: Tuple of Orbital objects corresponding to the specified basis
            set and atomic number.
    """
    from basis_set_exchange import get_basis
    from basis_set_exchange.sort import sort_basis

    bse_basis = get_basis(
        basis_name,
        elements=atomic_number,
        uncontract_spdf=True,
        uncontract_general=True,
    )
    bse_basis = sort_basis(bse_basis)["elements"]
    orbitals = []

    for s in bse_basis[str(atomic_number)]["electron_shells"]:
        for lmn in LMN_MAP[s["angular_momentum"][0]]:
            ao = Orbital.from_bse(
                center=np.zeros(3, dtype=default_fptype()),
                alphas=np.array(s["exponents"], dtype=default_fptype()),
                lmn=np.array(lmn, dtype=np.int32),
                coefficients=np.array(s["coefficients"], dtype=default_fptype()),
            )
            orbitals.append(ao)

    return tuple(orbitals)


def basis_iter(basis: Basis):
    from jax import tree

    from mess.special import triu_indices

    def take_primitives(indices):
        p = tree.map(lambda x: jnp.take(x, indices, axis=0), basis.primitives)
        c = jnp.take(basis.coefficients, indices)
        return p, c

    ii, jj = triu_indices(basis.num_primitives)
    lhs, cl = take_primitives(ii.reshape(-1))
    rhs, cr = take_primitives(jj.reshape(-1))
    return (ii, cl, lhs), (jj, cr, rhs)


RenormMode = Literal["orthonormal", "pyscf_cart", "pyscf_sph"]


def renorm(basis: Basis, mode: RenormMode = "orthonormal") -> Basis:
    """Renormalise the basis set.

    Args:
        basis (Basis): The basis set to renormalise.
        mode (str, optional): The normalisation mode. Can be "orthonormal" to
            normalise such that the self-overlap integral is 1, or either "pyscf_cart"
            or "pyscf_sph" to normalise according to PySCF's convention.
            Defaults to "orthonormal".

    Raises:
        ValueError: If an unknown normalisation mode is provided.

    Returns:
        Basis: The renormalised basis set.
    """
    match mode:
        case "orthonormal":
            from mess.integrals import overlap_basis

            S = overlap_basis(basis)
            n = 1 / jnp.sqrt(jnp.diag(S))
        case "pyscf_cart" | "pyscf_sph":
            from mess.interop import to_pyscf

            mol = to_pyscf(basis.structure, basis.basis_name)
            kind = mode.split("_")[1]
            n = np.sqrt(np.diag(mol.intor(f"int1e_ovlp_{kind}")))
        case _:
            modes = get_args(RenormMode)
            modes = ", ".join(modes)
            msg = f"Unknown renorm mode: {mode}"
            msg += f"\nMust be one of the following: {modes}"
            raise ValueError(msg)

    C = n[basis.orbital_index] * basis.coefficients
    return eqx.tree_at(lambda b: b.coefficients, basis, C)


def c_cart2sph(lmn: tuple[int, int, int], l: int, m: int) -> complex:
    """Transformation coefficients for Cartesian to spherical Gaussian coefficients.

    This function calculates the transformation coefficient from a Cartesian Gaussian
    function (defined by `lmn`) to a spherical Gaussian function (defined by `l` and
    `m`). The formula used is based on the relationship between Cartesian and spherical
    harmonics. See equation 15 of <https://doi.org/10.1002/qua.560540202>.


    Args:
        lmn (tuple): A tuple (lx, ly, lz) representing the angular momentum
                     components of the Cartesian Gaussian function.
        l (int): The total angular momentum quantum number of the spherical harmonic.
        m (int): The magnetic quantum number of the spherical harmonic.

    Returns:
        complex: The transformation coefficient as a complex scalar
    """
    from scipy.special import binom, factorial

    lmn, l, m = np.array(lmn), np.array(l), np.array(m)

    # Check if j = (lx + ly - |m|) / 2 is an integer
    abs_m = np.abs(m)
    j = lmn[0] + lmn[1] - abs_m

    if j % 2 != 0:
        # j must be half-integral so coefficient must be zero
        return np.array(0.0, dtype=complex)
    else:
        j = j // 2

    # constant pre-factor
    num = np.prod(factorial(2 * lmn)) * factorial(l) * factorial(l - abs_m)
    dem = factorial(2 * l) * np.prod(factorial(lmn)) * factorial(l + abs_m)
    out = np.sqrt(num / dem) / (2**l * factorial(l))

    # sum_i taking into account that binom(p, q) is zero for q < 0 and q > p
    # as well as avoiding negative factorial in the denominator
    i = np.arange(np.maximum(j, 0), (l - abs_m) // 2 + 1)
    iterm_num = binom(l, i) * binom(i, j) * (-1) ** i * factorial(2 * l - 2 * i)
    out *= np.sum(iterm_num / factorial(l - abs_m - 2 * i))

    # sum_k taking into account that binom(p, q) is zero for q < 0 and q > p
    ki = np.maximum((lmn[0] - abs_m // 2), 0)
    kf = np.minimum(j, lmn[0] // 2)
    k = np.arange(ki, kf + 1)

    if len(k) > 0:
        kterm = binom(j, k) * binom(abs_m, lmn[0] - 2 * k)
        power = np.sign(m) * 0.5 * (abs_m - lmn[0] + 2 * k)
        out *= np.sum(kterm * np.power(-1.0, power, dtype=complex))

    return out.astype(complex)


@cache
def transform_cart2sph(l: int) -> np.ndarray:
    """Transformation matrix for Cartesian to spherical Gaussian coefficients.

    This function generates a transformation matrix that converts Cartesian Gaussian
    coefficients to spherical Gaussian coefficients for a given total angular momentum
    `l`. Each row of the matrix corresponds to a Cartesian basis function (defined by
    `lmn` from `LMN_MAP[l]`), and each column corresponds to a spherical basis function
    (defined by `m` from `-l` to `l`).

    Args:
        l (int): The total angular momentum quantum number.

    Returns:
        np.ndarray: A 2D NumPy array representing the transformation matrix.
                    The shape of the matrix is `(num_cartesian_functions, 2*l + 1)`.
    """

    out = []
    for lmn in LMN_MAP[l]:
        row = []
        for m in range(-l, l + 1):
            row.append(c_cart2sph(lmn, l, m))

        out.append(row)

    return np.asarray(out)
