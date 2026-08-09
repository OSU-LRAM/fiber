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

from typing import cast

import equinox as eqx
import jax
import jax.tree_util as jtu
from diffrax import AbstractGlobalInterpolation
from jaxtyping import Array, Real

from ._custom_types import IntScalarLike, RealScalarLike
from ._groups._element import AbstractGroupElement, AbstractTangentVector
from ._interpolation import left_glerp, lerp
from ._operations import inv


class GeodesicInterpolation[GroupT: AbstractGroupElement](AbstractGlobalInterpolation):
    ts: Real[Array, " times"]  # type: ignore[reportIncompatibleVariableOverride]
    ys: GroupT

    def __check_init__(self):
        if self.ys.shape[0] != self.ts.shape[0]:
            raise ValueError(
                "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                "number of entries along the timelike dimension."
            )

    @property
    def ts_size(self) -> IntScalarLike:  # type: ignore[reportIncompatibleVariableOverride]
        return cast(IntScalarLike, self.ts.shape[0])

    @eqx.filter_jit
    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> GroupT:
        with jax.numpy_dtype_promotion("standard"):
            if t1 is not None:
                y0, y1 = jtu.tree_map(self.evaluate, [t0, t1], [left, left])
                return inv(y0) @ y1

            index, fractional_part = self._interpret_t(t0, left)

            prev_y = self.ys[index]
            next_y = self.ys[index + 1]
            prev_t = self.ts[index]
            next_t = self.ts[index + 1]
            diff_t = next_t - prev_t

            coeff = fractional_part / diff_t

            return left_glerp(prev_y, next_y, coeff)


class LeftBundleInterpolation[VectorT: AbstractTangentVector](
    AbstractGlobalInterpolation
):
    ts: Real[Array, " times"]  # type: ignore[reportIncompatibleVariableOverride]
    ys: VectorT

    def __check_init__(self):
        if self.ys.shape[0] != self.ts.shape[0]:
            raise ValueError(
                "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                "number of entries along the timelike dimension."
            )

    @property
    def ts_size(self) -> IntScalarLike:  # type: ignore[reportIncompatibleVariableOverride]
        return cast(IntScalarLike, self.ts.shape[0])

    @eqx.filter_jit
    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> VectorT:
        with jax.numpy_dtype_promotion("standard"):
            if t1 is not None:
                y0, y1 = jtu.tree_map(self.evaluate, [t0, t1], [left, left])
                g_prime = inv(y0.point) @ y1.point
                w_prime = y1 - y0
                return eqx.tree_at(lambda w: w.point.value, w_prime, g_prime.value)

            index, fractional_part = self._interpret_t(t0, left)

            prev_y = self.ys[index]
            next_y = self.ys[index + 1]
            prev_t = self.ts[index]
            next_t = self.ts[index + 1]
            diff_t = next_t - prev_t

            coeff = fractional_part / diff_t

            g_prime = left_glerp(prev_y.point, next_y.point, coeff)
            w_prime = lerp(prev_y, next_y, coeff)
            return eqx.tree_at(lambda w: w.point.value, w_prime, g_prime.value)
