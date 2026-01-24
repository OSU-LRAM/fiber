import functools

import jax.numpy as jnp
from jaxtyping import Array

from ..._linalg import softnorm


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _normalize(matrix: Array) -> Array:
    x_raw, y_raw, _ = jnp.split(matrix, 3)
    x_raw, y_raw = x_raw.squeeze(), y_raw.squeeze()
    x_norm = softnorm(x_raw)
    x = x_raw / jnp.maximum(x_norm, 1e-8)
    z = jnp.cross(x, y_raw)
    z_norm = softnorm(z)
    z = z / jnp.maximum(z_norm, 1e-8)
    y = jnp.cross(z, x)
    return jnp.stack((x, y, z)).squeeze()
