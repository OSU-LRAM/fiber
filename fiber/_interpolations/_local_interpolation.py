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
from jaxtyping import Array, PyTree

from .._custom_types import RealScalarLike
from .._elements._isometry import Isometry, _normalize_rotation
from .._operations import inv
from .._utils import join_state, split_state
from ._interpolations import glerp, lerp, slerp


def _linear_rescale(t0, t, t1) -> Array:
    cond = t0 == t1
    numerator = cast(Array, jnp.where(cond, 0, t - t0))
    denominator = cast(Array, jnp.where(cond, 1, t1 - t0))
    return numerator / denominator


class LocalDirectInterpolation(AbstractLocalInterpolation):
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
                g0, g1 = jtu.tree_map(Isometry.unflatten, (self.y0, self.y1))
                coeff = _linear_rescale(self.t0, t0, self.t1)
                pi = lerp(g0.position, g1.position, coeff)
                ri = slerp(g0.rotation, g1.rotation, coeff)
                ri_norm = _normalize_rotation(ri.as_matrix())
                return jnp.concatenate([pi, ri_norm.flatten()])
            else:
                ys = jtu.tree_map(self.evaluate, [t0, t1])
                g0, g1 = jtu.tree_map(Isometry.unflatten, ys)
                return (inv(g0) @ g1).flatten()


class LocalGeodesicInterpolation(AbstractLocalInterpolation):
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
                g0, g1 = jtu.tree_map(Isometry.unflatten, (self.y0, self.y1))
                coeff = _linear_rescale(self.t0, t0, self.t1)
                return glerp(g0, g1, coeff).flatten()
            else:
                ys = jtu.tree_map(self.evaluate, [t0, t1])
                (g0, g1) = jtu.tree_map(Isometry.unflatten, ys)
                return (inv(g0) @ g1).flatten()


class LocalPartitionedGeodesicInterpolation(AbstractLocalInterpolation):
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
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, [self.y0, self.y1])
                coeff = _linear_rescale(self.t0, t0, self.t1)
                gi = glerp(g0, g1, coeff)
                vi = lerp(v0, v1, coeff)
                return join_state(gi, vi)
            else:
                ys = jtu.tree_map(self.evaluate, [t0, t1])
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, ys)
                g_interval = inv(g0) @ g1
                v_interval = v1 - v0
                return join_state(g_interval, v_interval)


class LocalPartitionedDirectInterpolation(AbstractLocalInterpolation):
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
                pi = lerp(g0.position, g1.position, coeff)
                vi = lerp(v0, v1, coeff).as_vector()  # type: ignore
                ri = slerp(g0.rotation, g1.rotation, coeff)
                ri_norm = _normalize_rotation(ri.as_matrix())
                return jnp.concatenate([pi, ri_norm.flatten(), vi])
            else:
                ys = jtu.tree_map(self.evaluate, [t0, t1])
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, ys)
                g_interval = inv(g0) @ g1
                v_interval = v1 - v0
                return join_state(g_interval, v_interval)
