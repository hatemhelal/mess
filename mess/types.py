# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import config
from jaxtyping import Array, Float, Int

Float3 = Float[Array, "3"]
FloatNx3 = Float[Array, "N 3"]
FloatN = Float[Array, "N"]
Float3xNxN = Float[Array, "3 N N"]
FloatNxN = Float[Array, "N N"]
FloatNxM = Float[Array, "N M"]
Int3 = Int[Array, "3"]
IntN = Int[Array, "N"]

MeshAxes = Tuple[FloatN, FloatN, FloatN]

asintarray = partial(jnp.asarray, dtype=jnp.int32)


def default_fptype():
    return jnp.float64 if config.x64_enabled else jnp.float32


asfparray = partial(jnp.asarray, dtype=default_fptype())
