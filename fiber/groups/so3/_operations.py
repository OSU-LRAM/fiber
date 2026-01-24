import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._linalg import skew3, softnorm


def _logm(w: Array) -> Array:
    cos = (jnp.trace(w) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)
    w_hat = jnp.zeros((3, 3))
    return jax.lax.cond(
        jnp.sin(theta) < 1e-6,
        lambda: w_hat,
        lambda: 1 / 2 * theta / jnp.sin(theta) * (w - w.T),
    )


def _left_jac_inv(w: Array) -> Array:
    jac = jnp.eye(3)
    return jax.lax.cond(
        jnp.isclose(softnorm(w), 0.0),
        lambda: jac,
        lambda: jac
        - 0.5 * skew3(w)
        + (
            (1 / (softnorm(w) ** 2))
            - (1 + jnp.cos(softnorm(w))) / (2 * softnorm(w) * jnp.sin(softnorm(w)))
        )
        * (skew3(w) @ skew3(w)),
    )
