"""Discretised sampling of orbitals and charge density."""

from functools import partial
from typing import Optional, Tuple, Union

import equinox as eqx
import jax.numpy as jnp
from jax import jit, vjp, vmap
from pyscf import dft
from scipy.integrate import lebedev_rule

from mess.atomic_constants import sg1_atomic_radii
from mess.basis import Basis
from mess.interop import to_pyscf
from mess.structure import Structure
from mess.types import FloatN, FloatNx3, FloatNxN, MeshAxes


class Mesh(eqx.Module):
    points: FloatNx3
    weights: Optional[FloatN] = None
    axes: Optional[MeshAxes] = None


def uniform_mesh(
    n: Union[int, Tuple] = 50, b: Union[float, Tuple] = 10.0, ndim: int = 3
) -> Mesh:
    if isinstance(n, int):
        n = (n,) * ndim

    if isinstance(b, float):
        b = (b,) * ndim

    if not isinstance(n, (tuple, list)):
        raise ValueError("Expected an integer ")

    if len(n) != ndim:
        raise ValueError("n must be a tuple with {ndim} elements")

    if len(b) != ndim:
        raise ValueError("b must be a tuple with {ndim} elements")

    axes = [jnp.linspace(-bi, bi, ni) for bi, ni in zip(b, n)]
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    points = points.reshape(-1, ndim)
    return Mesh(points, axes=axes)


def density(basis: Basis, mesh: Mesh, P: Optional[FloatNxN] = None) -> FloatN:
    P = jnp.diag(basis.occupancy) if P is None else P
    orbitals = basis(mesh.points)
    return jnp.einsum("ij,pi,pj->p", P, orbitals, orbitals)


def density_and_grad(
    basis: Basis, mesh: Mesh, P: Optional[FloatNxN] = None
) -> Tuple[FloatN, FloatNx3]:
    def f(points):
        return density(basis, eqx.combine(points, rest), P)

    points, rest = eqx.partition(mesh, lambda x: id(x) == id(mesh.points))
    rho, df = vjp(f, points)
    grad_rho = df(jnp.ones_like(rho))[0].points
    return rho, grad_rho


def molecular_orbitals(
    basis: Basis, mesh: Mesh, C: Optional[FloatNxN] = None
) -> FloatN:
    C = jnp.eye(basis.num_orbitals) if C is None else C
    orbitals = basis(mesh.points) @ C
    return orbitals


def xcmesh_from_pyscf(structure: Structure, level: int = 3) -> Mesh:
    grids = dft.gen_grid.Grids(to_pyscf(structure))
    grids.level = level
    grids.build()
    return Mesh(points=grids.coords, weights=grids.weights)


def cell_function(mu, k=3):
    """A polynomial smoothing function for Becke partitioning."""

    def f(x):
        for _ in range(k):
            x = 1.5 * x - 0.5 * x**3
        return x

    return 0.5 * (1 - f(mu))


def becke_partition(structure: Structure, points: FloatNx3, weights: FloatN) -> Mesh:
    """Partitions a molecular grid using Becke's scheme.

    This function implements Becke's partitioning scheme [1]_ to assign weights to grid
    points based on their proximity to different atomic centers in a molecule. This
    ensures that each point contributes to the integral in a way that smoothly
    transitions between atomic regions.

    Args:
        structure (Structure): The molecular structure defining the atomic centers.
        points (FloatNx3): An array of grid points, typically generated from
            atom-centered grids. The shape is (num_atoms, num_grid_points_per_atom, 3).
        weights (FloatN): An array of initial weights for each grid point,
            corresponding to the `points` array. The shape is
            (num_atoms, num_grid_points_per_atom).

    Returns:
        Mesh: A Mesh object containing the partitioned points and their new weights.

    .. [1] A. D. Becke, "A multicenter numerical integration scheme for polyatomic
           molecules", The Journal of Chemical Physics, vol. 88, no. 4, pp. 2547-2553,
           Feb. 1988, https://doi.org/10.1063/1.454033.
    """

    # confocal elliptical coordinate ([1] eq 11) vmap is used to convert this scalar
    # function to a map over mesh points and pairs of atom centers
    @partial(vmap, in_axes=(None, 0, 0))
    @partial(vmap, in_axes=(0, None, None))
    def calculate_mu(rp, Ri, Rj):
        ri = jnp.linalg.norm(rp - Ri)
        rj = jnp.linalg.norm(rp - Rj)
        Rij = jnp.linalg.norm(Ri - Rj)
        return (ri - rj) / Rij

    # For each atom-centered grid, calc the distance from each mesh point to atom pairs
    num_pairs = structure.num_atoms * (structure.num_atoms - 1)
    ii, jj = jnp.nonzero(~jnp.eye(structure.num_atoms, dtype=bool), size=num_pairs)
    Ri = structure.position[ii]
    Rj = structure.position[jj]
    mu = calculate_mu(points.reshape(-1, 3), Ri, Rj)
    s = cell_function(mu)
    s = s.reshape(structure.num_atoms, structure.num_atoms - 1, *weights.shape)
    P = jnp.prod(s, axis=1)
    weights = weights * (P[jnp.diag_indices(structure.num_atoms)] / jnp.sum(P, axis=0))

    return Mesh(points.reshape(-1, 3), weights.reshape(-1))


@partial(jit, static_argnums=(1, 2))
def sg1_mesh(
    structure: Structure, num_radial: int = 50, angular_order: int = 23
) -> Mesh:
    """Builds a molecular quadrature grid using the SG1 scheme.

    The SG1 grid introduced by Gill et al. [1]_ for each atomic center is generated by
    taking the product of a radial mesh following an Euler-Maclaurin scheme with an
    angular Lebedev-Laikov [2]_ mesh. The resulting mesh is partitioned following the
    scheme introduced by Becke [3]_ to reweight each point based on their proximity to
    different atomic centers.

    Args:
        structure (Structure): The molecular structure for which to build the grid.
        num_radial (int, optional): The number of radial points used in the
            Euler-Maclaurin scheme. Defaults to 50.
        angular_order (int, optional): The order of the Lebedev-Laikov angular grid.
            Defaults to 23.

    Returns:
        Mesh: A Mesh object containing the SG1 points and weights.

    .. [1] P. M. W. Gill, B. G. Johnson, and J. A. Pople, "A standard grid for density
           functional calculations", Chemical Physics Letters, vol. 209, no. 5-6, pp.
           506-512, Jul. 1993, doi: https://doi.org/10.1016/0009-2614(93)80125-9.
    .. [2] V.I. Lebedev, and D.N. Laikov. "A quadrature formula for the sphere of
           the 131st algebraic order of accuracy". Doklady Mathematics, Vol. 59,
           No. 3, 1999, pp. 477-481.
    .. [3] A. D. Becke, "A multicenter numerical integration scheme for polyatomic
           molecules", The Journal of Chemical Physics, vol. 88, no. 4, pp. 2547-2553,
           Feb. 1988, https://doi.org/10.1063/1.454033.
    """

    atom_radius = jnp.asarray(sg1_atomic_radii())[structure.atomic_number]
    atom_radius = atom_radius.reshape(-1, 1)
    ii = jnp.arange(1, num_radial + 1)
    rad_weights = (
        2 * atom_radius**3 * (num_radial + 1) * ii**5 / (num_radial + 1 - ii) ** 7
    )
    rad_points = atom_radius * ii**2 / (num_radial + 1 - ii) ** 2
    ang_points, ang_weights = lebedev_rule(angular_order)

    # Outer product of radial and angular points to form atom centered meshes
    # [num_atoms, num_rad] x [num_ang, 3] -> [num_atoms, num_rad, num_ang, 3]
    points = jnp.einsum("ij,kl->ijlk", rad_points, ang_points)

    # [num_atoms, num_rad] x [num_ang -> [num_atoms, num_rad, num_ang]
    weights = jnp.einsum("ij,l->ijl", rad_weights, ang_weights)

    # Points and weights grouped by atom and centered on atoms
    points = points.reshape(structure.num_atoms, -1, 3)  # [num_atoms, num_grid, 3]
    points = points + structure.position[:, None, :]
    weights = weights.reshape(structure.num_atoms, -1)  # [num_atoms, num_grid]

    return becke_partition(structure, points, weights)
