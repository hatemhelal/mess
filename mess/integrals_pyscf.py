import jax
import jax.numpy as jnp
from mess.interop import to_pyscf
from mess.types import FloatNxN, default_fptype
from mess.basis import Basis


def overlap_pyscf(basis: Basis) -> FloatNxN:
    def func(basis: Basis) -> FloatNxN:
        mol = to_pyscf(basis.structure, basis.basis_name)
        kind = "sph" if basis.spherical else "cart"
        return jnp.array(mol.intor(f"int1e_ovlp_{kind}"), dtype=default_fptype())

    out_spec = jax.ShapeDtypeStruct(
        shape=2 * (basis.num_orbitals,), dtype=default_fptype()
    )
    return jax.pure_callback(func, out_spec, basis, vmap_method="sequential")
