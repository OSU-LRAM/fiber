# Copyright 2026, Evan Palmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array

from .._custom_types import RealScalarLike
from .._elements._isometry import Isometry
from .._operations import expm, rminus


def lerp(y0: Array, y1: Array, coeff: RealScalarLike) -> Array:
    return y0 + coeff * (y1 - y0)


def slerp(y0: R, y1: R, coeff: RealScalarLike) -> R:
    q0, q1 = y0.as_quat(), y1.as_quat()
    dot = jnp.dot(q0, q1)

    def lerp_quat():
        return lerp(q0, q1, coeff)

    def slerp_quat():
        theta = jnp.acos(dot)
        coeff0 = jnp.sin((1 - coeff) * theta) / jnp.sin(theta)
        coeff1 = jnp.sin(coeff * theta) / jnp.sin(theta)
        return coeff0 * q0 + coeff1 * q1

    result = jax.lax.cond(jnp.isclose(dot, 1.0), lerp_quat, slerp_quat)
    return R.from_quat(result)


def glerp(y0: Isometry, y1: Isometry, coeff: RealScalarLike) -> Isometry:
    return y0 @ expm(coeff * rminus(y1, y0))
