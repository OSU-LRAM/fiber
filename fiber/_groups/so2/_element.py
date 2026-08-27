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
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ..._vecfuncs import skew2, softnorm, vex2
from .._element import (
    AbstractCotangentVector,
    AbstractGroupElement,
    AbstractTangentVector,
)


class Rotation2d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nq: int = eqx.field(static=True, default=4)

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
    def unravel(cls, params: Array):
        return cls(_unravel_element(params))

    def ravel(self) -> Array:
        return _ravel_element(self.value)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(2))

    @classmethod
    def concatenate(cls, elements: Sequence[Rotation2d]):
        return cls(jnp.concatenate([element.value for element in elements]))

    @classmethod
    def stack(cls, elements: Sequence[Rotation2d]):
        return cls(jnp.stack([element.value for element in elements]))

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
    nv: int = eqx.field(static=True, default=1)
    nx: int = eqx.field(static=True, default=5)

    def __check_init__(self):
        if not isinstance(self.point, Rotation2d):
            raise TypeError("The tangent vector point must be a Rotation2d instance")

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: ArrayLike, point: Rotation2d | None = None):
        mat = jnp.asarray(mat)
        if point is None:
            shape = mat.shape[:-2]
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Rotation2d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, skew2(vec))

    def as_vector(self) -> Array:
        return vex2(self.value)

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_vector(coords)
        return cls(Rotation2d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_vector(self.value, self.point.value)

    @classmethod
    def unravel(cls, params: ArrayLike, point: Rotation2d | None = None):
        params = jnp.asarray(params)
        if point is None:
            shape = params.shape[:-1]
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, _unravel_vector(params))

    def ravel(self) -> Array:
        return _ravel_vector(self.value)

    @classmethod
    def zeros(cls, point: Rotation2d | None = None):
        if point is None:
            point = Rotation2d.eye()
        return cls(point, jnp.zeros_like(point.value))

    @classmethod
    def concatenate(cls, vectors: Sequence[Spin2d]):
        points = Rotation2d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    @classmethod
    def stack(cls, vectors: Sequence[Spin2d]):
        points = Rotation2d.stack([vector.point for vector in vectors])
        return cls(points, jnp.stack([vector.value for vector in vectors]))

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
    def __truediv__(self, other: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
        return Spin2d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Spin2d:
        return Spin2d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Spin2d(")
        return f"Spin2d({repr})"


class Moment2d(AbstractCotangentVector[Rotation2d]):
    point: Rotation2d
    value: Array = eqx.field(converter=jnp.asarray)
    nf: int = eqx.field(static=True, default=1)
    nx: int = eqx.field(static=True, default=5)

    @property
    def single(self) -> bool:
        # so(2)* is 1-dimensional, so `value` is a bare scalar per batch element
        # (no trailing component axis, matching `Spin2d.as_vector()`'s convention).
        return self.value.ndim == 0

    def __check_init__(self):
        if not isinstance(self.point, Rotation2d):
            raise TypeError("The cotangent vector point must be a Rotation2d instance")

        if self.point.shape[:-2] != self.value.shape:
            raise ValueError(
                "Must have point.shape[:-2] == value.shape, that is to say "
                "each cotangent vector should be assigned a point with matching "
                "batch dimensions."
            )

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Rotation2d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape
            point = Rotation2d(jnp.broadcast_to(jnp.eye(2), (*shape, 2, 2)))
        return cls(point, vec)

    def as_vector(self) -> Array:
        return self.value

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_covector(coords)
        return cls(Rotation2d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_covector(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Moment2d]):
        points = Rotation2d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    @classmethod
    def stack(cls, vectors: Sequence[Moment2d]):
        points = Rotation2d.stack([vector.point for vector in vectors])
        return cls(points, jnp.stack([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single covector is not subscriptable.")
        return Moment2d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single covector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Moment2d(point, value)

    def pair(self, tangent: Spin2d) -> RealScalarLike:
        return self.value * tangent.as_vector()

    @dispatch
    def __add__(self, other: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
        return Moment2d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Moment2d:
        return Moment2d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
        return Moment2d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Moment2d:
        return Moment2d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
        return Moment2d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Moment2d:
        return Moment2d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
        return Moment2d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Moment2d:
        return Moment2d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Moment2d(")
        return f"Moment2d({repr})"


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
def _unravel_element(params: Array) -> Array:
    return params.reshape((2, 2))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_element(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_vector(params: Array) -> Array:
    return params.reshape((2, 2))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_vector(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_coords_vector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Rotation2d.nq,))
    return _unravel_element(point), skew2(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_coords_vector(vector: Array, point: Array) -> Array:
    return jnp.concatenate(
        [_ravel_element(point), jnp.reshape(vex2(vector), (1,))], axis=0
    )


@functools.partial(jnp.vectorize, signature="(n)->(m,m),()")
def _from_coords_covector(coords: Array) -> tuple[Array, RealScalarLike]:
    point, vector = jnp.split(coords, (Rotation2d.nq,))
    return _unravel_element(point), vector[0]


@functools.partial(jnp.vectorize, signature="(),(n,n)->(m)")
def _as_coords_covector(vector: RealScalarLike, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), jnp.reshape(vector, (1,))], axis=0)
