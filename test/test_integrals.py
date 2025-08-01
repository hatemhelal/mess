import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mess.basis import basisset
from mess.integrals import (
    eri_basis,
    eri_basis_sparse,
    eri_primitives,
    kinetic_basis,
    kinetic_primitives,
    nuclear_basis,
    nuclear_primitives,
    overlap_basis,
    overlap_primitives,
)
from mess.interop import to_pyscf
from mess.primitive import Primitive
from mess.structure import molecule
from conftest import is_mem_limited


def test_overlap():
    # Exercise 3.21 of "Modern quantum chemistry: introduction to advanced
    # electronic structure theory."" by Szabo and Ostlund
    alpha = 0.270950 * 1.24 * 1.24
    a = Primitive(alpha=alpha)
    b = Primitive(alpha=alpha, center=jnp.array([1.4, 0.0, 0.0]))
    assert_allclose(overlap_primitives(a, a), 1.0, atol=1e-5)
    assert_allclose(overlap_primitives(b, b), 1.0, atol=1e-5)
    assert_allclose(overlap_primitives(b, a), 0.6648, atol=1e-5)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g", "6-31+g*"])
def test_water_overlap(basis_name):
    basis = basisset(molecule("water"), basis_name)
    actual_overlap = overlap_basis(basis)

    scfmol = to_pyscf(molecule("water"), basis_name=basis_name)
    expect_overlap = scfmol.intor("int1e_ovlp_sph")
    assert_allclose(actual_overlap, expect_overlap, atol=1e-5)


def test_kinetic():
    # PyQuante test case for kinetic primitive integral
    p = Primitive()
    assert_allclose(kinetic_primitives(p, p), 1.5, atol=1e-5)

    # Reproduce the kinetic energy matrix for H2 using STO-3G basis set
    # See equation 3.230 of "Modern quantum chemistry: introduction to advanced
    # electronic structure theory."" by Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")
    actual = kinetic_basis(basis)
    expect = np.array([[0.7600, 0.2365], [0.2365, 0.7600]])
    assert_allclose(actual, expect, atol=1e-4)


@pytest.mark.parametrize("basis_name", ["sto-3g", "6-31+g", "6-31+g*"])
def test_water_kinetic(basis_name):
    basis = basisset(molecule("water"), basis_name)
    actual = kinetic_basis(basis)

    expect = to_pyscf(molecule("water"), basis_name=basis_name).intor("int1e_kin_sph")
    assert_allclose(actual, expect, atol=1e-5)


def test_nuclear():
    # PyQuante test case for nuclear attraction integral
    p = Primitive()
    c = jnp.zeros(3)
    assert_allclose(nuclear_primitives(p, p, c), -1.595769, atol=1e-5)

    # Reproduce the nuclear attraction matrix for H2 using STO-3G basis set
    # See equation 3.231 and 3.232 of Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")
    actual = nuclear_basis(basis)
    expect = np.array([
        [[-1.2266, -0.5974], [-0.5974, -0.6538]],
        [[-0.6538, -0.5974], [-0.5974, -1.2266]],
    ])

    assert_allclose(actual, expect, atol=1e-4)


def test_water_nuclear():
    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    actual = nuclear_basis(basis).sum(axis=0)
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int1e_nuc_sph")
    assert_allclose(actual, expect, atol=1e-3)


def test_eri():
    # PyQuante test cases for ERI
    a, b, c, d = [Primitive()] * 4
    assert_allclose(eri_primitives(a, b, c, d), 1.128379, atol=1e-5)

    c, d = [Primitive(lmn=jnp.array([1, 0, 0]))] * 2
    assert_allclose(eri_primitives(a, b, c, d), 0.940316, atol=1e-5)

    # H2 molecule in sto-3g: See equation 3.235 of Szabo and Ostlund
    h2 = molecule("h2")
    basis = basisset(h2, "sto-3g")

    actual = eri_basis(basis)
    expect = np.empty((2, 2, 2, 2), dtype=np.float32)
    expect[0, 0, 0, 0] = expect[1, 1, 1, 1] = 0.7746
    expect[0, 0, 1, 1] = expect[1, 1, 0, 0] = 0.5697
    expect[1, 0, 0, 0] = expect[0, 0, 0, 1] = 0.4441
    expect[0, 1, 0, 0] = expect[0, 0, 1, 0] = 0.4441
    expect[0, 1, 1, 1] = expect[1, 1, 1, 0] = 0.4441
    expect[1, 0, 1, 1] = expect[1, 1, 0, 1] = 0.4441
    expect[1, 0, 1, 0] = expect[0, 1, 1, 0] = 0.2970
    expect[0, 1, 0, 1] = expect[1, 0, 0, 1] = 0.2970
    assert_allclose(actual, expect, atol=1e-4)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.skipif(is_mem_limited(), reason="Not enough host memory!")
def test_water_eri(sparse):
    basis_name = "sto-3g"
    h2o = molecule("water")
    basis = basisset(h2o, basis_name)
    actual = eri_basis_sparse(basis) if sparse else eri_basis(basis)
    aosym = "s8" if sparse else "s1"
    expect = to_pyscf(h2o, basis_name=basis_name).intor("int2e_cart", aosym=aosym)
    assert_allclose(actual, expect, atol=1e-4)
