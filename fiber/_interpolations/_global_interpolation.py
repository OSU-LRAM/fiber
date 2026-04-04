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

from typing import Optional

import equinox as eqx
import jax
import jax.tree_util as jtu
from diffrax import AbstractGlobalInterpolation
from jaxtyping import Array, PyTree, Real, Shaped

from .._custom_types import IntScalarLike, RealScalarLike
from .._elements._isometry import Isometry
from .._operations import inv, rminus
from .._utils import join_state, split_state
from ._interpolations import glerp, lerp


class GeodesicInterpolation(AbstractGlobalInterpolation):
    ts: Real[Array, " times"]  # type: ignore
    ys: PyTree[Shaped[Array, "times ..."]]

    def __check_init__(self):
        def _check(_ys):
            if _ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                    "number of entries along the timelike dimension."
                )

        jtu.tree_map(_check, self.ys)

    @property
    def ts_size(self) -> IntScalarLike:  # type: ignore
        return self.ts.shape[0]

    @eqx.filter_jit
    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> Array:
        with jax.numpy_dtype_promotion("standard"):
            if t1 is not None:
                ys = jtu.tree_map(self.evaluate, [t0, t1], [left, left])
                g0, g1 = jtu.tree_map(Isometry.unflatten, ys)
                return (inv(g0) @ g1).flatten()

            index, fractional_part = self._interpret_t(t0, left)

            prev_y = self.ys[index]
            next_y = self.ys[index + 1]
            prev_t = self.ts[index]
            next_t = self.ts[index + 1]
            diff_t = next_t - prev_t

            prev_y, next_y = jtu.tree_map(Isometry.unflatten, (prev_y, next_y))
            coeff = fractional_part / diff_t
            return glerp(prev_y, next_y, coeff).flatten()

    @eqx.filter_jit
    def derivative(self, t: RealScalarLike, left: bool = True) -> Array:
        index, _ = self._interpret_t(t, left)

        prev_y = self.ys[index]
        next_y = self.ys[index + 1]
        prev_t = self.ts[index]
        next_t = self.ts[index + 1]
        diff_t = next_t - prev_t

        prev_y, next_y = jtu.tree_map(Isometry.unflatten, (prev_y, next_y))

        with jax.numpy_dtype_promotion("standard"):
            return (prev_y @ (rminus(next_y, prev_y) / diff_t)).flatten()


class PartitionedGeodesicInterpolation(AbstractGlobalInterpolation):
    ts: Real[Array, " times"]  # type: ignore
    ys: PyTree[Shaped[Array, "times ..."]]

    def __check_init__(self):
        def _check(_ys):
            if _ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                    "number of entries along the timelike dimension."
                )

        jtu.tree_map(_check, self.ys)

    @property
    def ts_size(self) -> IntScalarLike:  # type: ignore
        return self.ts.shape[0]

    @eqx.filter_jit
    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> Array:
        with jax.numpy_dtype_promotion("standard"):
            if t1 is not None:
                ys = jtu.tree_map(self.evaluate, [t0, t1], [left, left])
                (g0, v0), (g1, v1) = jtu.tree_map(split_state, ys)
                g_interval = inv(g0) @ g1
                v_interval = v1 - v0
                return join_state(g_interval, v_interval)

            index, fractional_part = self._interpret_t(t0, left)

            prev_y = self.ys[index]
            next_y = self.ys[index + 1]
            prev_t = self.ts[index]
            next_t = self.ts[index + 1]
            diff_t = next_t - prev_t

            (g0, v0), (g1, v1) = jtu.tree_map(split_state, [prev_y, next_y])
            coeff = fractional_part / diff_t
            gi = glerp(g0, g1, coeff)
            vi = lerp(v0, v1, coeff)
            return join_state(gi, vi)
