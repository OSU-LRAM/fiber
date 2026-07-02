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

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Optional, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ..._vecfuncs import skew2, softnorm, vex2
from .._element import AbstractGroupElement, AbstractTangentVector


class Rotation2d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nparams: int = eqx.field(static=True, default=4)

    @property
    def angle(self) -> Array:
        return _rotation_angle(self.value)

    @classmethod
    def from_matrix(cls, mat: ArrayLike):
        return cls(_normalize_rotation(jnp.asarray(mat)))

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_angle(cls, angle: ArrayLike):
        return cls(_from_angle(jnp.asarray(angle)))

    def as_angle(self) -> Array:
        return self.angle

    @classmethod
    def unflatten(cls, params: Array):
        return cls(_unflatten(params))

    def flatten(self) -> Array:
        return _flatten(self.value)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(2))

    @classmethod
    def concatenate(cls, elements: Sequence[Rotation2d]):
        return cls(jnp.concatenate([element.value for element in elements]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single element is not subscriptable.")
        return Rotation2d(self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single element is not iterable.")
        for value in self.value:
            yield Rotation2d(value)

    @dispatch
    def __matmul__(self, other: Rotation2d) -> Rotation2d:  # type: ignore[reportRedeclaration]
        return Rotation2d(self.value @ other.value)

    @dispatch
    def __matmul__(self, other: Spin2d) -> Spin2d:
        return Spin2d(self, self.value @ other.value)

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Rotation2d(")
        return f"Rotation2d({repr})"


class Spin2d(AbstractTangentVector[Rotation2d]):
    point: Rotation2d
    value: Array = eqx.field(converter=jnp.asarray)
    nparams: int = eqx.field(static=True, default=1)
    nbundle: int = eqx.field(static=True, default=5)

    def __check_init__(self):
        if not isinstance(self.point, Rotation2d):
            raise ValueError("The tangent vector point must be a Rotation2d instance")

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: Array, point: Optional[Rotation2d] = None):
        vec = jnp.asarray(mat)
        if point is None:
            shape = vec.shape[:-2]
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Optional[Rotation2d] = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, skew2(vec))

    def as_vector(self) -> Array:
        return vex2(self.value)

    @classmethod
    def from_bundle(cls, bundle: Array):
        point, vector = _from_bundle(bundle)
        return cls(Rotation2d(point), vector)

    def as_bundle(self) -> Array:
        return _as_bundle(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Spin2d]):
        points = Rotation2d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single vector is not subscriptable.")
        return Spin2d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single vector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Spin2d(point, value)

    def __matmul__(self, other: Rotation2d) -> Spin2d:
        return Spin2d(other, self.value @ other.value)

    @dispatch
    def __add__(self, other: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
        return Spin2d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Spin2d:
        return Spin2d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
        return Spin2d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Spin2d:
        return Spin2d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
        return Spin2d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Spin2d:
        return Spin2d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __div__(self, other: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
        return Spin2d(self.point, self.value / other.value)

    @dispatch
    def __div__(self, other: ArrayLike) -> Spin2d:
        return Spin2d(self.point, self.value / other)

    __rdiv__ = __div__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Spin2d(")
        return f"Spin2d({repr})"


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _normalize_rotation(mat: Array) -> Array:
    x_raw, y_raw = jnp.split(mat, 2)
    x_raw, y_raw = x_raw.squeeze(), y_raw.squeeze()
    x_norm = softnorm(x_raw)
    x = x_raw / jnp.maximum(x_norm, 1e-8)
    perp = jnp.array([-x[1], x[0]])
    sign = jnp.sign(jnp.dot(perp, y_raw))
    sign = jnp.where(sign == 0, 1.0, sign)
    y = sign * perp
    return jnp.stack((x, y))


@functools.partial(jnp.vectorize, signature="()->(n,n)")
def _from_angle(angle: RealScalarLike) -> Array:
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    return jnp.array([[cos, -sin], [sin, cos]])


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def _rotation_angle(mat: Array) -> RealScalarLike:
    return jnp.arctan2(mat[1, 0], mat[0, 0])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten(params: Array) -> Array:
    return params.reshape((2, 2))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _flatten(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_bundle(bundle: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(bundle, (Rotation2d.nparams,))
    return _unflatten(point), skew2(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_bundle(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_flatten(point), vex2(vector)], axis=0)
