import pytest
from numpy.testing import assert_allclose

from mess.mesh import sg1_mesh
from mess.primitive import Primitive, product
from mess.structure import Structure


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
