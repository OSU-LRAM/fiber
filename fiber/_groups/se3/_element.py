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
from ..._vecfuncs import skew3, vex3
from .._element import (
    AbstractCotangentVector,
    AbstractGroupElement,
    AbstractTangentVector,
)
from ..so3._element import Rotation3d, _normalize_rotation


class Isometry3d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)
    nq: int = eqx.field(static=True, default=12)

    @property
    def position(self) -> Array:
        return self.value[..., :3, 3]

    @property
    def rotation(self) -> Rotation3d:
        return Rotation3d(self.value[..., :3, :3])

    @classmethod
    def from_matrix(cls, mat: ArrayLike):
        return cls(_from_matrix(mat))

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
        seq: str,
        eucl: Union[ArrayLike, Sequence[ArrayLike]],
        degrees: bool = False,
    ):
        fn = functools.partial(_from_euclidean, seq=seq, degrees=degrees)
        vec_fn = jnp.vectorize(fn, signature="(n)->(m,m)")
        return cls(vec_fn(eucl))

    def as_euclidean(self, seq: str, degrees: bool = False) -> Array:
        fn = functools.partial(_as_euclidean, seq=seq, degrees=degrees)
        vec_fn = jnp.vectorize(fn, signature="(n,n)->(m)")
        return vec_fn(self.value)

    @classmethod
    def from_pq(cls, vec: ArrayLike):
        vec_fn = jnp.vectorize(_from_pq, signature="(n)->(m,m)")
        return cls(vec_fn(jnp.asarray(vec)))

    def as_pq(self, canonical: bool = False, scalar_first: bool = False) -> Array:
        fn = functools.partial(_as_pq, canonical=canonical, scalar_first=scalar_first)
        vec_fn = jnp.vectorize(fn, signature="(n,n)->(m)")
        return vec_fn(self.value)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(4))

    @classmethod
    def concatenate(cls, elements: Sequence[Isometry3d]):
        return cls(jnp.concatenate([element.value for element in elements]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single element is not subscriptable.")
        return Isometry3d(self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single element is not iterable.")
        for value in self.value:
            yield Isometry3d(value)

    @dispatch
    def __matmul__(self, other: Isometry3d) -> Isometry3d:  # type: ignore[reportRedeclaration]
        return Isometry3d(self.value @ other.value)

    @dispatch
    def __matmul__(self, other: Twist3d) -> Twist3d:
        return Twist3d(self, self.value @ other.value)

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Isometry3d(")
        return f"Isometry3d({repr})"


class Twist3d(AbstractTangentVector[Isometry3d]):
    point: Isometry3d
    value: Array = eqx.field(converter=jnp.asarray)
    nv: int = eqx.field(static=True, default=6)
    nx: int = eqx.field(static=True, default=18)

    def __check_init__(self):
        if not isinstance(self.point, Isometry3d):
            raise ValueError(
                "The tangent vector point must be a `Isometry3d` instance!"
            )

        if self.point.shape != self.value.shape:
            raise ValueError(
                "Must have point.shape == value.shape, that is to say each tangent "
                "vector should be assigned a point."
            )

    @classmethod
    def from_matrix(cls, mat: ArrayLike, point: Optional[Isometry3d] = None):
        mat = jnp.asarray(mat)
        if point is None:
            shape = mat.shape[:-2]
            point = Isometry3d(jnp.broadcast_to(jnp.eye(4), (*shape, 4, 4)))
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Optional[Isometry3d] = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Isometry3d(jnp.broadcast_to(jnp.eye(4), (*shape, 4, 4)))
        return cls(point, _from_vector(vec))

    def as_vector(self) -> Array:
        return _as_vector(self.value)

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_vector(coords)
        return cls(Isometry3d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_vector(self.value, self.point.value)

    @classmethod
    def unravel(cls, params: ArrayLike, point: Optional[Isometry3d] = None):
        params = jnp.asarray(params)
        if point is None:
            shape = params.shape[:-1]
            point = Isometry3d(jnp.broadcast_to(jnp.eye(4), (*shape, 4, 4)))
        return cls(point, _unravel_vector(params))

    def ravel(self) -> Array:
        return _ravel_vector(self.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Twist3d]):
        points = Isometry3d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single vector is not subscriptable.")
        return Twist3d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single vector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Twist3d(point, value)

    def __matmul__(self, other: Isometry3d) -> Twist3d:
        return Twist3d(other, self.value @ other.value)

    @dispatch
    def __add__(self, other: Twist3d) -> Twist3d:  # type: ignore[reportRedeclaration]
        return Twist3d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Twist3d:
        return Twist3d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Twist3d) -> Twist3d:  # type: ignore[reportRedeclaration]
        return Twist3d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Twist3d:
        return Twist3d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Twist3d) -> Twist3d:  # type: ignore[reportRedeclaration]
        return Twist3d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Twist3d:
        return Twist3d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Twist3d) -> Twist3d:  # type: ignore[reportRedeclaration]
        return Twist3d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Twist3d:
        return Twist3d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Twist3d(")
        return f"Twist3d({repr})"


class Wrench3d(AbstractCotangentVector[Isometry3d]):
    point: Isometry3d
    value: Array = eqx.field(converter=jnp.asarray)
    nf: int = eqx.field(static=True, default=6)
    nx: int = eqx.field(static=True, default=18)

    def __check_init__(self):
        if not isinstance(self.point, Isometry3d):
            raise ValueError(
                "The cotangent vector point must be a `Isometry3d` instance!"
            )

        if self.point.shape[:-2] != self.value.shape[:-1]:
            raise ValueError(
                "Must have point.shape[:-2] == value.shape[:-1], that is to say "
                "each cotangent vector should be assigned a point with matching "
                "batch dimensions."
            )

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Optional[Isometry3d] = None):
        vec = jnp.asarray(vec)
        if point is None:
            shape = vec.shape[:-1]
            point = Isometry3d(jnp.broadcast_to(jnp.eye(4), (*shape, 4, 4)))
        return cls(point, vec)

    def as_vector(self) -> Array:
        return self.value

    @classmethod
    def from_coords(cls, coords: ArrayLike):
        point, vector = _from_coords_covector(coords)
        return cls(Isometry3d(point), vector)

    def as_coords(self) -> Array:
        return _as_coords_covector(self.value, self.point.value)

    @classmethod
    def concatenate(cls, vectors: Sequence[Wrench3d]):
        points = Isometry3d.concatenate([vector.point for vector in vectors])
        return cls(points, jnp.concatenate([vector.value for vector in vectors]))

    def __getitem__(self, indexer):
        if self.single:
            raise TypeError("Single covector is not subscriptable.")
        return Wrench3d(self.point[indexer], self.value[indexer])

    def __iter__(self):
        if self.single:
            raise TypeError("Single covector is not iterable.")
        for point, value in zip(self.point, self.value):
            yield Wrench3d(point, value)

    def pair(self, tangent: Twist3d) -> RealScalarLike:
        return jnp.sum(self.value * tangent.as_vector(), axis=-1)

    @dispatch
    def __add__(self, other: Wrench3d) -> Wrench3d:  # type: ignore[reportRedeclaration]
        return Wrench3d(self.point, self.value + other.value)

    @dispatch
    def __add__(self, other: ArrayLike) -> Wrench3d:
        return Wrench3d(self.point, self.value + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Wrench3d) -> Wrench3d:  # type: ignore[reportRedeclaration]
        return Wrench3d(self.point, self.value - other.value)

    @dispatch
    def __sub__(self, other: ArrayLike) -> Wrench3d:
        return Wrench3d(self.point, self.value - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Wrench3d) -> Wrench3d:  # type: ignore[reportRedeclaration]
        return Wrench3d(self.point, self.value * other.value)

    @dispatch
    def __mul__(self, other: ArrayLike) -> Wrench3d:
        return Wrench3d(self.point, self.value * other)

    __rmul__ = __mul__

    @dispatch
    def __truediv__(self, other: Wrench3d) -> Wrench3d:  # type: ignore[reportRedeclaration]
        return Wrench3d(self.point, self.value / other.value)

    @dispatch
    def __truediv__(self, other: ArrayLike) -> Wrench3d:
        return Wrench3d(self.point, self.value / other)

    __rtruediv__ = __truediv__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Wrench3d(")
        return f"Wrench3d({repr})"


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix(mat: Array) -> Array:
    mat = mat.at[:3, :3].set(_normalize_rotation(mat[:3, :3]))
    mat = mat.at[3, :4].set(jnp.array([0, 0, 0, 1]))
    return mat


def _from_euclidean(vec: Array, seq: str, degrees: bool) -> Array:
    pos, angles = jnp.split(vec, 2, axis=-1)
    rot = R.from_euler(seq, angles, degrees).as_matrix()
    return jnp.block([[rot, pos.reshape(3, 1)], [jnp.zeros(3), 1]])


def _as_euclidean(mat: Array, seq: str, degrees: bool) -> Array:
    pos = mat[:3, 3]
    angles = R.from_matrix(mat[:3, :3]).as_euler(seq, degrees)
    return jnp.concatenate([pos, cast(Array, angles)], axis=-1)


def _from_pq(vec: Array) -> Array:
    pos, quat = jnp.split(vec, (3,))
    rot = R.from_quat(quat).as_matrix()
    return jnp.block([[rot, pos.reshape(3, 1)], [jnp.zeros(3), 1]])


def _as_pq(mat: Array, canonical: bool, scalar_first: bool) -> Array:
    pos = mat[:3, 3]
    quat = R.from_matrix(mat[:3, :3]).as_quat(canonical, scalar_first)
    return jnp.concatenate([pos, cast(Array, quat)], axis=-1)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vec: Array) -> Array:
    lin, ang = jnp.split(vec, 2)
    return jnp.block([[skew3(ang), lin.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(mat: Array) -> Array:
    lin, ang = mat[:3, 3], vex3(mat[:3, :3])
    return jnp.concatenate([lin, ang])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_vector(params: Array) -> Array:
    lin, ang = jnp.split(params, (3,))
    return jnp.block([[ang.reshape((3, 3)), lin.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_vector(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:3, 3], matrix[:3, :3].flatten()])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unravel_element(params: Array) -> Array:
    pos, rot = jnp.split(params, (3,))
    mat = jnp.block([[rot.reshape((3, 3)), pos.reshape(3, 1)], [jnp.zeros(3), 1]])
    return mat


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _ravel_element(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:3, 3], matrix[:3, :3].flatten()])


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(m,m)")
def _from_coords_vector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Isometry3d.nq,))
    return _unravel_element(point), _from_vector(vector)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(m)")
def _as_coords_vector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), _as_vector(vector)], axis=0)


@functools.partial(jnp.vectorize, signature="(n)->(m,m),(q)")
def _from_coords_covector(coords: Array) -> tuple[Array, Array]:
    point, vector = jnp.split(coords, (Isometry3d.nq,))
    return _unravel_element(point), vector


@functools.partial(jnp.vectorize, signature="(n),(m,m)->(q)")
def _as_coords_covector(vector: Array, point: Array) -> Array:
    return jnp.concatenate([_ravel_element(point), vector], axis=0)
