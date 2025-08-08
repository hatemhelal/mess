import equinox as eqx
import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mess.basis import basisset
from mess.mesh import density, sg1_mesh
from mess.primitive import Primitive, product
from mess.structure import Structure, molecule


@pytest.mark.parametrize("lmn", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_sg1_overlap(lmn):
    # Use He to just to center the mesh on the origin
    structure = Structure(2, (0, 0, 0))
    mesh = sg1_mesh(structure)

    # Primitives are normalized to have a self-overlap of 1.0
    p = Primitive(lmn=lmn)
    p = product(p, p)

    actual = mesh.weights @ p(mesh.points)
    assert_allclose(actual, 1.0)


def test_charge_quadrature():
    structure = molecule("benzene")
    mesh = sg1_mesh(structure, num_radial=128, angular_order=41)
    basis = basisset(structure, "sto-3g")
    actual_Q = mesh.weights @ density(basis, mesh)
    assert_allclose(actual_Q, structure.num_electrons)


def test_charge_grad():
    @jax.grad
    def charge(pos, rest):
        structure = eqx.combine(pos, rest)
        mesh = sg1_mesh(structure)
        return mesh.weights @ density(basis, mesh)

    structure = molecule("benzene")
    basis = basisset(structure, "sto-3g")
    pos, rest = eqx.partition(structure, lambda x: id(x) == id(structure.position))
    grad_q = charge(pos, rest)
    assert_allclose(grad_q.position, np.zeros_like(structure.position), atol=1e-2)
