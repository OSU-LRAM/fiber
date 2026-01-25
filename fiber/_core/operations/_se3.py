import jax.numpy as jnp
from jaxtyping import Array

from .._linalg import vex3
from ._so3 import left_jac_inv as so3_left_jac_inv
from ._so3 import logm as so3_logm


def expm(g: Array) -> Array: ...


def logm(g: Array) -> Array:
    t, rot = g[:3, 3], g[:3, :3]
    w_hat = so3_logm(rot)
    return jnp.block([[w_hat, so3_left_jac_inv(vex3(w_hat)) @ t], [jnp.zeros(4)]])
