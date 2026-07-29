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
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ..._vecfuncs import skew3, softnorm, vex3
from .._element import (
    AbstractCotangentVector,
    AbstractGroupElement,
    AbstractTangentVector,
)


class Rotation3d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nq: int = eqx.field(static=True, default=9)

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

    def as_quat(self, canonical: bool = False, scalar_first: bool = False) -> Array:
        return R.from_matrix(self.value).as_quat(canonical, scalar_first)

    @classmethod
    def from_euler(
        cls,
        seq: str,
        angles: ArrayLike | Sequence[ArrayLike],
        degrees: bool = False,
    ):
        return cls(R.from_euler(seq, jnp.asarray(angles), degrees).as_matrix())

    def as_euler(self, seq: str, degrees: bool = False) -> Array:
        euler = R.from_matrix(self.value).as_euler(seq, degrees)
        return cast(Array, euler)

    @classmethod
    def unravel(cls, params: Array):
        return cls(_unravel_element(params))

    def ravel(self) -> Array:
        return _ravel_element(self.value)

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
    nv: int = eqx.field(static=True, default=3)
    nx: int = eqx.field(static=True, default=12)

    def __check_init__(self):
        if not isinstance(self.point, Rotation3d):
            raise TypeError("The tangent vector point must be a Rotation3d instance")

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: ArrayLike, point: Rotation3d | None = None):
        mat = jnp.asarray(mat)
        if point is None:
            shape = mat.shape[:-2]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Rotation3d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, skew3(vec))

    def as_vector(self) -> Array:
        return vex3(self.value)

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_vector(coords)
        return cls(Rotation3d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_vector(self.value, self.point.value)

    @classmethod
    def unravel(cls, params: ArrayLike, point: Rotation3d | None = None):
        params = jnp.asarray(params)
        if point is None:
            shape = params.shape[:-1]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, _unravel_vector(params))

    def ravel(self) -> Array:
        return _ravel_vector(self.value)

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
    def __truediv__(self, other: Spin3d) -> Spin3d:  # type: ignore[reportRedeclaration]
        return Spin3d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Spin3d:
        return Spin3d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Spin3d(")
        return f"Spin3d({repr})"


class Moment3d(AbstractCotangentVector[Rotation3d]):
    point: Rotation3d
    value: Array = eqx.field(converter=jnp.asarray)
    nf: int = eqx.field(static=True, default=3)
    nx: int = eqx.field(static=True, default=12)

    def __check_init__(self):
        if not isinstance(self.point, Rotation3d):
            raise TypeError("The cotangent vector point must be a Rotation3d instance")

        if self.point.shape[:-2] != self.value.shape[:-1]:
            raise ValueError(
                "Must have point.shape[:-2] == value.shape[:-1], that is to say "
                "each cotangent vector should be assigned a point with matching "
                "batch dimensions."
            )

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Rotation3d | None = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Rotation3d(jnp.broadcast_to(jnp.eye(3), (*shape, 3, 3)))
        return cls(point, vec)

    def as_vector(self) -> Array:
        return self.value

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_covector(coords)
        return cls(Rotation3d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_covector(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Moment3d]):
        points = Rotation3d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single covector is not subscriptable.")
        return Moment3d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single covector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Moment3d(point, value)

    def pair(self, tangent: Spin3d) -> RealScalarLike:
        return jnp.sum(self.value * tangent.as_vector(), axis=-1)

    @dispatch
    def __add__(self, other: Moment3d) -> Moment3d:  # type: ignore[reportRedeclaration]
        return Moment3d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Moment3d:
        return Moment3d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Moment3d) -> Moment3d:  # type: ignore[reportRedeclaration]
        return Moment3d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Moment3d:
        return Moment3d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Moment3d) -> Moment3d:  # type: ignore[reportRedeclaration]
        return Moment3d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Moment3d:
        return Moment3d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Moment3d) -> Moment3d:  # type: ignore[reportRedeclaration]
        return Moment3d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Moment3d:
        return Moment3d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Moment3d(")
        return f"Moment3d({repr})"


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
def _unravel_element(params: Array) -> Array:
    return params.reshape((3, 3))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_element(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_vector(params: Array) -> Array:
    return params.reshape((3, 3))


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_vector(matrix: Array) -> Array:
    return matrix.flatten()


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_coords_vector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Rotation3d.nq,))
    return _unravel_element(point), skew3(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_coords_vector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), vex3(vector)], axis=0)


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(q)")
def _from_coords_covector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Rotation3d.nq,))
    return _unravel_element(point), vector


@functools.partial(jnp.vectorize, signature="(n),(m,m)->(q)")
def _as_coords_covector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), vector], axis=0)
