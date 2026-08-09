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
from ..._vecfuncs import skew2, vex2
from .._element import (
    AbstractCotangentVector,
    AbstractGroupElement,
    AbstractTangentVector,
)
from ..so2._element import Rotation2d, _normalize_rotation, _rotation_angle


class Isometry2d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nq: int = eqx.field(static=True, default=6)

    @property
    def position(self) -> Array:
        return self.value[..., :2, 2]

    @property
    def rotation(self) -> Rotation2d:
        return Rotation2d(self.value[..., :2, :2])

    @classmethod
    def from_matrix(cls, mat: ArrayLike):
        return cls(_from_matrix(jnp.asarray(mat)))

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def unravel(cls, params: Array):
        return cls(_unravel_element(params))

    def ravel(self) -> Array:
        return _ravel_element(self.value)

    @classmethod
    def from_euclidean(
        cls,
        eucl: ArrayLike,
        degrees: bool = False,
    ):
        fn = functools.partial(_from_euclidean, degrees=degrees)
        vec_fn = jnp.vectorize(fn, signature="(n)->(m,m)")
        return cls(vec_fn(eucl))

    def as_euclidean(self, degrees: bool = False) -> Array:
        fn = functools.partial(_as_euclidean, degrees=degrees)
        vec_fn = jnp.vectorize(fn, signature="(n,n)->(m)")
        return vec_fn(self.value)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(3))

    @classmethod
    def concatenate(cls, elements: Sequence[Isometry2d]):
        return cls(jnp.concatenate([element.value for element in elements]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single element is not subscriptable.")
        return Isometry2d(self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single element is not iterable.")
        for value in self.value:
            yield Isometry2d(value)

    @dispatch
    def __matmul__(self, other: Isometry2d) -> Isometry2d:  # type: ignore[reportRedeclaration]
        return Isometry2d(self.value @ other.value)

    @dispatch
    def __matmul__(self, other: Twist2d) -> Twist2d:
        return Twist2d(self, self.value @ other.value)

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Isometry2d(")
        return f"Isometry2d({repr})"


class Twist2d(AbstractTangentVector[Isometry2d]):
    point: Isometry2d
    value: Array = eqx.field(converter=jnp.asarray)
    nv: int = eqx.field(static=True, default=3)
    nx: int = eqx.field(static=True, default=9)

    def __check_init__(self):
        if not isinstance(self.point, Isometry2d):
            raise TypeError("The tangent vector point must be a `Isometry2d` instance!")

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: ArrayLike, point: Isometry2d | None = None):
        mat = jnp.asarray(mat)
        if point is None:
            shape = mat.shape[:-2]
            point = Isometry2d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Isometry2d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Isometry2d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, _from_vector(vec))

    def as_vector(self) -> Array:
        return _as_vector(self.value)

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_vector(coords)
        return cls(Isometry2d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_vector(self.value, self.point.value)

    @classmethod
    def unravel(cls, params: ArrayLike, point: Isometry2d | None = None):
        params = jnp.asarray(params)
        if point is None:
            shape = params.shape[:-1]
            point = Isometry2d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, _unravel_vector(params))

    def ravel(self) -> Array:
        return _ravel_vector(self.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Twist2d]):
        points = Isometry2d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single vector is not subscriptable.")
        return Twist2d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single vector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Twist2d(point, value)

    def __matmul__(self, other: Isometry2d) -> Twist2d:
        return Twist2d(other, self.value @ other.value)

    @dispatch
    def __add__(self, other: Twist2d) -> Twist2d:  # type: ignore[reportRedeclaration]
        return Twist2d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Twist2d:
        return Twist2d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Twist2d) -> Twist2d:  # type: ignore[reportRedeclaration]
        return Twist2d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Twist2d:
        return Twist2d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Twist2d) -> Twist2d:  # type: ignore[reportRedeclaration]
        return Twist2d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Twist2d:
        return Twist2d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Twist2d) -> Twist2d:  # type: ignore[reportRedeclaration]
        return Twist2d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Twist2d:
        return Twist2d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Twist2d(")
        return f"Twist2d({repr})"


class Wrench2d(AbstractCotangentVector[Isometry2d]):
    point: Isometry2d
    value: Array = eqx.field(converter=jnp.asarray)
    nf: int = eqx.field(static=True, default=3)
    nx: int = eqx.field(static=True, default=9)

    def __check_init__(self):
        if not isinstance(self.point, Isometry2d):
            raise TypeError(
                "The cotangent vector point must be a `Isometry2d` instance!"
            )

        if self.point.shape[:-2] != self.value.shape[:-1]:
            raise ValueError(
                "Must have point.shape[:-2] == value.shape[:-1], that is to say "
                "each cotangent vector should be assigned a point with matching "
                "batch dimensions."
            )

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Isometry2d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Isometry2d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, vec)

    def as_vector(self) -> Array:
        return self.value

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_covector(coords)
        return cls(Isometry2d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_covector(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Wrench2d]):
        points = Isometry2d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single covector is not subscriptable.")
        return Wrench2d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single covector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Wrench2d(point, value)

    def pair(self, tangent: Twist2d) -> RealScalarLike:
        return jnp.sum(self.value * tangent.as_vector(), axis=-1)

    @dispatch
    def __add__(self, other: Wrench2d) -> Wrench2d:  # type: ignore[reportRedeclaration]
        return Wrench2d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Wrench2d:
        return Wrench2d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Wrench2d) -> Wrench2d:  # type: ignore[reportRedeclaration]
        return Wrench2d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Wrench2d:
        return Wrench2d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Wrench2d) -> Wrench2d:  # type: ignore[reportRedeclaration]
        return Wrench2d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Wrench2d:
        return Wrench2d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Wrench2d) -> Wrench2d:  # type: ignore[reportRedeclaration]
        return Wrench2d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Wrench2d:
        return Wrench2d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Wrench2d(")
        return f"Wrench2d({repr})"


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix(mat: Array) -> Array:
    mat = mat.at[:2, :2].set(_normalize_rotation(mat[:2, :2]))
    mat = mat.at[2, :3].set(jnp.array([0, 0, 1]))
    return mat


def _from_euclidean(vec: Array, degrees: bool) -> Array:
    pos, angle = vec[:2], vec[2]
    if degrees:
        angle = jnp.deg2rad(angle)
    cos, sin = jnp.cos(angle), jnp.sin(angle)
    rot = jnp.array([[cos, -sin], [sin, cos]])
    return jnp.block([[rot, pos.reshape(2, 1)], [jnp.zeros(2), 1]])


def _as_euclidean(mat: Array, degrees: bool) -> Array:
    pos = mat[:2, 2]
    angle = _rotation_angle(mat[:2, :2])
    if degrees:
        angle = jnp.rad2deg(angle)
    return jnp.concatenate([pos, jnp.reshape(angle, (1,))])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vec: Array) -> Array:
    lin, ang = vec[:2], vec[2]
    return jnp.block([[skew2(ang), lin.reshape(2, 1)], [jnp.zeros(3)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(mat: Array) -> Array:
    lin, ang = mat[:2, 2], vex2(mat[:2, :2])
    return jnp.concatenate([lin, jnp.reshape(ang, (1,))])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_vector(params: Array) -> Array:
    lin, ang = jnp.split(params, (2,))
    return jnp.block([[ang.reshape((2, 2)), lin.reshape(2, 1)], [jnp.zeros(3)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_vector(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:2, 2], matrix[:2, :2].flatten()])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_element(params: Array) -> Array:
    pos, rot = jnp.split(params, (2,))
    mat = jnp.block([[rot.reshape((2, 2)), pos.reshape(2, 1)], [jnp.zeros(2), 1]])
    return mat


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_element(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:2, 2], matrix[:2, :2].flatten()])


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_coords_vector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Isometry2d.nq,))
    return _unravel_element(point), _from_vector(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_coords_vector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), _as_vector(vector)], axis=0)


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(q)")
def _from_coords_covector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Isometry2d.nq,))
    return _unravel_element(point), vector


@functools.partial(jnp.vectorize, signature="(n),(m,m)->(q)")
def _as_coords_covector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), vector], axis=0)
