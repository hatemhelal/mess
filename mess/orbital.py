"""Container for a linear combination of Gaussian Primitives (aka contraction)."""

from functools import partial
from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import tree, vmap

from mess.primitive import Primitive, eval_primitive
from mess.types import Float3, FloatN, FloatNx3, IntN, IntNx3


class Orbital(eqx.Module):
    primitives: Tuple[Primitive]
    coefficients: FloatN

    @property
    def num_primitives(self) -> int:
        return len(self.primitives)

    def __call__(self, pos: FloatNx3) -> FloatN:
        pos = jnp.atleast_2d(pos)
        assert pos.ndim == 2 and pos.shape[1] == 3, "pos must have shape [N,3]"

        @partial(vmap, in_axes=(0, 0, None))
        def eval_orbital(p: Primitive, coef: float, pos: FloatNx3):
            return coef * eval_primitive(p, pos)

        batch = tree.map(lambda *xs: jnp.stack(xs), *self.primitives)
        out = jnp.sum(eval_orbital(batch, self.coefficients, pos), axis=0)
        return out


def make_contraction(
    center: Float3, alphas: FloatN, lmns: IntNx3, coefficients: FloatN
) -> Orbital:
    """Make an Orbital from a contraction of primitives with a shared center.

    A contracted Gaussian orbital is a linear combination of primitive Gaussian
    functions that share a common center.

    Args:
        center (Float3): The common center of all primitives in the orbital.
        alphas (FloatN): Array of exponent values for each primitive.
        lmns (IntNx3): Array of angular momentum quantum numbers (lx, ly, lz)
                        for each primitive.
        coefficients (FloatN): Array of contraction coefficients for each primitive.

    Returns:
        Orbital: A new Orbital instance.
    """
    p = [Primitive(center=center, alpha=a, lmn=lmn) for a, lmn in zip(alphas, lmns)]
    return Orbital(primitives=p, coefficients=coefficients)


def batch_orbitals(orbitals: Tuple[Orbital]) -> Tuple[Primitive, FloatN, IntN]:
    """Flattens a sequence of `Orbital` objects into a batched representation.

    This utility function takes a tuple of `Orbital` objects, each potentially
    containing multiple `Primitive` functions, and consolidates them into a
    single, batched `Primitive` object. This "Structure of Arrays" format is
    highly efficient for vectorized computations in JAX, for example with `jax.vmap`.

    The function also returns the corresponding flattened contraction coefficients
    and an index map to trace each primitive back to its original orbital.

    Args:
        orbitals (Tuple[Orbital]): A tuple of Orbital objects to be batched.

    Returns:
        A tuple containing the batched data:
            - primitives (Primitive): A single `Primitive` object where each attribute
                is a batched array (e.g., `primitives.center` is a `FloatNx3` array).
            - coefficients (FloatN): A 1D array of the concatenated contraction
                coefficients.
            - orbital_index (IntN): An array mapping each primitive to its original
                orbital's index in the input tuple.
    """
    primitives = [p for o in orbitals for p in o.primitives]
    primitives = tree.map(lambda *xs: jnp.stack(xs), *primitives)
    coefficients = jnp.concatenate([o.coefficients for o in orbitals])
    orbital_index = jnp.concatenate([
        i * jnp.ones(o.num_primitives, dtype=jnp.int32) for i, o in enumerate(orbitals)
    ])
    return primitives, coefficients, orbital_index
