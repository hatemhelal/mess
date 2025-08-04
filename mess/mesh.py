"""Discretised sampling of orbitals and charge density."""

from typing import Optional, Tuple, Union

import numpy as np
import equinox as eqx
import jax.numpy as jnp
from pyscf import dft
from jax import vjp
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
    def f(x):
        for _ in range(k):
            x = 1.5 * x - 0.5 * x**3
        return x

    return 0.5 * (1 - f(mu))


def sg1_mesh(
    structure: Structure,
    num_radial: int = 50,
    angular_order: int = 23,
    epsilon: float = 1e-12,
) -> Mesh:
    atom_radius = sg1_atomic_radii()[structure.atomic_number]
    atom_radius = atom_radius.reshape(-1, 1)
    ii = np.arange(1, num_radial + 1)
    rad_weights = (
        2 * atom_radius**3 * (num_radial + 1) * ii**5 / (num_radial + 1 - ii) ** 7
    )
    rad_points = atom_radius * ii**2 / (num_radial + 1 - ii) ** 2
    ang_points, ang_weights = lebedev_rule(angular_order)

    # Outer product of radial and angular points to form atom centered meshes
    # [num_atoms, num_rad] x [num_ang, 3] -> [num_atoms, num_rad, num_ang, 3]
    points = np.einsum("ij,kl->ijlk", rad_points, ang_points)
    points = points + structure.position[:, None, None, :]

    # [num_atoms, num_rad] x [num_ang -> [num_atoms, num_rad, num_ang]
    weights = np.einsum("ij,l->ijl", rad_weights, ang_weights)

    # Points and weights grouped by atom
    points = points.reshape(structure.num_atoms, -1, 3)
    weights = weights.reshape(structure.num_atoms, -1)

    ii, jj = np.triu_indices(structure.num_atoms, 1)
    ri_vec = points[:, None, :] - structure.position[None, ii, None, :]
    ri = np.linalg.norm(ri_vec, axis=-1)

    rj_vec = points[:, None, :] - structure.position[None, jj, None, :]
    rj = np.linalg.norm(rj_vec, axis=-1)

    R_ij_vec = structure.position[ii, :] - structure.position[jj, :]
    R_ij = np.linalg.norm(R_ij_vec, axis=1)

    mu = (ri - rj) / (R_ij[:, None] + epsilon)
    s = cell_function(mu)
    P = np.prod(s, axis=1)
    w = P / (np.sum(P, axis=0) + epsilon)

    weights = weights * w
    return Mesh(points.reshape(-1, 3), weights.reshape(-1))
