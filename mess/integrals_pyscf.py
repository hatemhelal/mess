import jax
import jax.numpy as jnp
from mess.interop import to_pyscf
from mess.types import FloatNxN, FloatNxNxNxN, default_fptype
from mess.basis import Basis


def _1e_pyscf(basis: Basis, name: str) -> FloatNxN:
    def func(basis: Basis) -> FloatNxN:
        mol = to_pyscf(basis.structure, basis.basis_name)
        kind = "sph" if basis.spherical else "cart"
        return jnp.array(mol.intor(f"int1e_{name}_{kind}"), dtype=default_fptype())

    out_spec = jax.ShapeDtypeStruct(
        shape=2 * (basis.num_orbitals,), dtype=default_fptype()
    )
    return jax.pure_callback(func, out_spec, basis, vmap_method="sequential")


def overlap_pyscf(basis: Basis) -> FloatNxN:
    return _1e_pyscf(basis, "ovlp")


def kinetic_pyscf(basis: Basis) -> FloatNxN:
    return _1e_pyscf(basis, "kin")


def nuclear_pyscf(basis: Basis) -> FloatNxN:
    return _1e_pyscf(basis, "nuc")


def eri_pyscf(basis: Basis) -> FloatNxNxNxN:
    def func(basis: Basis) -> FloatNxNxNxN:
        mol = to_pyscf(basis.structure, basis.basis_name)
        kind = "sph" if basis.spherical else "cart"
        return jnp.array(mol.intor(f"int2e_{kind}", aosim="s1"), dtype=default_fptype())

    out_spec = jax.ShapeDtypeStruct(
        shape=4 * (basis.num_orbitals,), dtype=default_fptype()
    )

    return jax.pure_callback(func, out_spec, basis, vmap_method="sequential")
