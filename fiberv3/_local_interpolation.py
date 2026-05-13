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

from typing import Generic, Optional, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import AbstractLocalInterpolation
from jaxtyping import Array

from ._custom_types import RealScalarLike
from ._groups._element import AbstractGroupElement, AbstractTangentVector
from ._interpolation import glerp, lerp
from ._ops import expm, rminus

_GroupElement = TypeVar("_GroupElement", bound=AbstractGroupElement)
_TangentVector = TypeVar("_TangentVector", bound=AbstractTangentVector)


def linear_rescale(t0, t, t1) -> Array:
    cond = t0 == t1
    numerator = cast(Array, jnp.where(cond, 0, t - t0))
    denominator = cast(Array, jnp.where(cond, 1, t1 - t0))
    return numerator / denominator


class LocalGeodesicInterpolation(AbstractLocalInterpolation, Generic[_GroupElement]):
    t0: RealScalarLike  # type: ignore[reportIncompatibleVariableOverride]
    t1: RealScalarLike  # type: ignore[reportIncompatibleVariableOverride]
    y0: _GroupElement
    y1: _GroupElement

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _GroupElement:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                coeff = linear_rescale(self.t0, t0, self.t1)
                return glerp(self.y0, self.y1, coeff)
            else:
                coeff = (t1 - t0) / (self.t1 - self.t0)
                incr = expm(coeff * rminus(self.y1, self.y0))  # type: ignore
                return cast(_GroupElement, incr)


class LocalBundleInterpolation(AbstractLocalInterpolation, Generic[_TangentVector]):
    t0: RealScalarLike  # type: ignore[reportIncompatibleVariableOverride]
    t1: RealScalarLike  # type: ignore[reportIncompatibleVariableOverride]
    y0: _TangentVector
    y1: _TangentVector

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _TangentVector:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                coeff = linear_rescale(self.t0, t0, self.t1)
                g_prime = glerp(self.y0.point, self.y1.point, coeff)
                w_prime = lerp(self.y0, self.y1, coeff)
            else:
                coeff = (t1 - t0) / (self.t1 - self.t0)
                g_prime = expm(coeff * rminus(self.y1.point, self.y0.point))
                w_prime = coeff * (self.y1 - self.y0)  # type: ignore

            return eqx.tree_at(lambda w: w.point.value, w_prime, g_prime.value)
