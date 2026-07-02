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

import functools
from collections.abc import Sequence
from typing import Optional, Union, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ..._vecfuncs import skew3, softnorm, vex3
from .._element import AbstractGroupElement, AbstractTangentVector


class Rotation3d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nparams: int = eqx.field(static=True, default=9)

    @property
    def angle(self) -> float:
        return _rotation_angle(self.value)

    @classmethod
    def from_matrix(cls, mat: ArrayLike):
        return cls(_normalize_rotation(jnp.asarray(mat)))

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_quat(cls, quat: ArrayLike):
        return cls(R.from_quat(jnp.asarray(quat)).as_matrix())

    def as_quat(self) -> Array:
        return R.from_matrix(self.value).as_quat()

    @classmethod
    def from_euler(
        cls,
        seq: str,
        angles: Union[ArrayLike, Sequence[ArrayLike]],
        degrees: bool = False,
    ):
        return cls(R.from_euler(seq, jnp.asarray(angles), degrees).as_matrix())

    def as_euler(self, seq: str, degrees: bool = False) -> Array:
        euler = R.from_matrix(self.value).as_euler(seq, degrees)
        return cast(Array, euler)

    @classmethod
    def unflatten(cls, params: Array):
        return cls(_unflatten(params))

    def flatten(self) -> Array:
        return _flatten(self.value)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(3))

    @classmethod
    def concatenate(cls, elements: Sequence[Rotation3d]):
        return cls(jnp.concatenate([element.value for element in elements]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single element is not subscriptable.")
        return Rotation3d(self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single element is not iterable.")
        for value in self.value:
            yield Rotation3d(value)

    @dispatch
    def __matmul__(self, other: Rotation3d) -> Rotation3d:  # type: ignore[reportRedeclaration]
        return Rotation3d(self.value @ other.value)

    @dispatch
    def __matmul__(self, other: Spin3d) -> Spin3d:
        return Spin3d(self, self.value @ other.value)

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Rotation3d(")
        return f"Rotation3d({repr})"


class Spin3d(AbstractTangentVector[Rotation3d]):
    point: Rotation3d
    value: Array = eqx.field(converter=jnp.asarray)
    nparams: int = eqx.field(static=True, default=3)
    nbundle: int = eqx.field(static=True, default=12)

    def __check_init__(self):
        if not isinstance(self.point, Rotation3d):
            raise ValueError("The tangent vector point must be a Rotation3d instance")

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: Array, point: Optional[Rotation3d] = None):
        vec = jnp.asarray(mat)
        if point is None:
            shape = vec.shape[:-2]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Optional[Rotation3d] = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, skew3(jnp.asarray(vec)))

    def as_vector(self) -> Array:
        return vex3(self.value)

    @classmethod
    def from_bundle(cls, bundle: Array):
        point, vector = _from_bundle(bundle)
        return cls(Rotation3d(point), vector)

    def as_bundle(self) -> Array:
        return _as_bundle(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Spin3d]):
        points = Rotation3d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single vector is not subscriptable.")
        return Spin3d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single vector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Spin3d(point, value)

    def __matmul__(self, other: Rotation3d) -> Spin3d:
        return Spin3d(other, self.value @ other.value)

    @dispatch
    def __add__(self, other: Spin3d) -> Spin3d:  # type: ignore[reportRedeclaration]
        return Spin3d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Spin3d:
        return Spin3d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Spin3d) -> Spin3d:  # type: ignore[reportRedeclaration]
        return Spin3d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Spin3d:
        return Spin3d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Spin3d) -> Spin3d:  # type: ignore[reportRedeclaration]
        return Spin3d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Spin3d:
        return Spin3d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __div__(self, other: Spin3d) -> Spin3d:  # type: ignore[reportRedeclaration]
        return Spin3d(self.point, self.value / other.value)

    @dispatch
    def __div__(self, other: ArrayLike) -> Spin3d:
        return Spin3d(self.point, self.value / other)

    __rdiv__ = __div__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Spin3d(")
        return f"Spin3d({repr})"


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _normalize_rotation(mat: Array) -> Array:
    x_raw, y_raw, _ = jnp.split(mat, 3)
    x_raw, y_raw = x_raw.squeeze(), y_raw.squeeze()
    x_norm = softnorm(x_raw)
    x = x_raw / jnp.maximum(x_norm, 1e-8)
    z = jnp.cross(x, y_raw)
    z_norm = softnorm(z)
    z = z / jnp.maximum(z_norm, 1e-8)
    y = jnp.cross(z, x)
    return jnp.stack((x, y, z)).squeeze()


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def _rotation_angle(mat: Array) -> RealScalarLike:
    cos = (jnp.trace(mat) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)
    return theta


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten(params: Array) -> Array:
    return params.reshape((2, 2))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _flatten(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_bundle(bundle: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(bundle, (Rotation3d.nparams,))
    return _unflatten(point), skew3(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_bundle(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_flatten(point), vex3(vector)], axis=0)
