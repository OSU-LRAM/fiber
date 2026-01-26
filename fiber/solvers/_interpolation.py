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

from typing import Optional, cast

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractLocalInterpolation
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array, PyTree

from .._custom_types import RealScalarLike
from .._elements._isometry import _flatten, _normalize_rotation
from .._utils import split_state


def _linear_rescale(t0, t, t1) -> Array:
    cond = t0 == t1
    numerator = cast(Array, jnp.where(cond, 0, t - t0))
    denominator = cast(Array, jnp.where(cond, 1, t1 - t0))
    return numerator / denominator


def _lerp(y0: Array, y1: Array, coeff: RealScalarLike) -> Array:
    return y0 + coeff * (y1 - y0)


def _slerp(y0: R, y1: R, coeff: RealScalarLike) -> R:
    q0, q1 = y0.as_quat(), y1.as_quat()
    dot = jnp.dot(q0, q1)

    def lerp_quat():
        return _lerp(q0, q1, coeff)

    def slerp_quat():
        theta = jnp.acos(dot)
        coeff0 = jnp.sin((1 - coeff) * theta) / jnp.sin(theta)
        coeff1 = jnp.sin(coeff * theta) / jnp.sin(theta)
        return coeff0 * q0 + coeff1 * q1

    result = jax.lax.cond(jnp.isclose(dot, 1.0), lerp_quat, slerp_quat)
    return R.from_quat(result)


class GeodesicInterpolation(AbstractLocalInterpolation):
    t0: RealScalarLike  # type: ignore
    t1: RealScalarLike  # type: ignore
    y0: Array
    y1: Array

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> PyTree[Array]:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, (self.y0, self.y1))
                coeff = _linear_rescale(self.t0, t0, self.t1)
                p1, rot1 = g0.position, g0.rotation
                p2, rot2 = g1.position, g1.rotation
                p = _lerp(p1, p2, coeff)
                g_circ = _lerp(v0.as_vector(), v1.as_vector(), coeff)
                rot = _slerp(rot1, rot2, coeff)
                rot_norm = _normalize_rotation(rot.as_matrix())
                return jnp.concatenate([p, rot_norm.flatten(), g_circ])
            else:
                s0, s1 = self.evaluate(t0), self.evaluate(t1)
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, (s0, s1))
                g_interval = _flatten(jnp.linalg.inv(g0.as_matrix()) @ g1.as_matrix())
                v_interval = v1.as_vector() - v0.as_vector()
                return jnp.concatenate([g_interval, v_interval])
