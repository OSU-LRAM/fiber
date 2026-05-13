from typing import Generic, Optional, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from diffrax import AbstractLocalInterpolation
from jaxtyping import Array

from ._custom_types import RealScalarLike
from ._groups._element import AbstractGroupElement, AbstractTangentVector
from ._interpolation import glerp, lerp
from ._ops import expm, rminus

_G = TypeVar("_G", bound=AbstractGroupElement)
_T = TypeVar("_T", bound=AbstractTangentVector)


def linear_rescale(t0, t, t1) -> Array:
    cond = t0 == t1
    numerator = cast(Array, jnp.where(cond, 0, t - t0))
    denominator = cast(Array, jnp.where(cond, 1, t1 - t0))
    return numerator / denominator


class LocalGeodesicInterpolation(AbstractLocalInterpolation, Generic[_G]):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: _G
    y1: _G

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _G:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                coeff = linear_rescale(self.t0, t0, self.t1)
                return glerp(self.y0, self.y1, coeff)
            else:
                coeff = (t1 - t0) / (self.t1 - self.t0)
                return expm(coeff * rminus(self.y1, self.y0))


class LocalPartitionedInterpolation(AbstractLocalInterpolation, Generic[_T]):
    t0: RealScalarLike
    t1: RealScalarLike
    y0: _T
    y1: _T

    def evaluate(
        self, t0: RealScalarLike, t1: Optional[RealScalarLike] = None, left: bool = True
    ) -> _T:
        del left
        with jax.numpy_dtype_promotion("standard"):
            if t1 is None:
                coeff = linear_rescale(self.t0, t0, self.t1)
                g_prime = glerp(self.y0.point, self.y1.point, coeff)
                w_prime = lerp(self.y0, self.y1, coeff)
            else:
                coeff = (t1 - t0) / (self.t1 - self.t0)
                g_prime = expm(coeff * rminus(self.y1.point, self.y0.point))
                w_prime = coeff * (self.y1 - self.y0)

            return eqx.tree_at(lambda w: w.point.value, w_prime, g_prime.value)
