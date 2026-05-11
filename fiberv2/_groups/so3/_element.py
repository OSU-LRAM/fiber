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

from collections.abc import Sequence
from functools import partial
from typing import final, override

import jax.numpy as jnp
import numpy as np
from equinox import field
from jax.scipy.spatial.transform import Rotation
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ...linalg._vecfuncs import skew3, softnorm, vex3
from .._element import Element


@final
class Rotation3d(Element):
    """SO(3) group element."""

    coordinates: Array
    size: int = field(static=True, default=9)

    @override
    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(cls.normalize(jnp.asarray(matrix)))

    @classmethod
    def from_euler(cls, seq: str, angles: Sequence[int | float], degrees: bool = False):
        return cls(Rotation.from_euler(seq, jnp.asarray(angles), degrees).as_matrix())

    @classmethod
    def from_quat(cls, quat: Sequence[int | float]):
        return cls(Rotation.from_quat(jnp.asarray(quat)).as_matrix())

    @override
    def as_matrix(self) -> Array:
        return self.coordinates

    def as_euler(self, seq: str, degrees: bool = False) -> ArrayLike:
        return Rotation.from_matrix(self.coordinates).as_euler(seq, degrees)

    def as_quat(self) -> ArrayLike:
        return Rotation.from_matrix(self.coordinates).as_quat()

    @override
    def flatten(self) -> Array:
        return _flatten(self.coordinates)

    @override
    @classmethod
    def unflatten(cls, flat: ArrayLike) -> Rotation3d:
        return cls(cls.normalize(_unflatten(jnp.asarray(flat))))

    @property
    def angle(self):
        _rotation_angle(self.coordinates)

    @override
    @classmethod
    def eye(cls) -> Rotation3d:
        return cls(jnp.eye(3))

    @override
    @classmethod
    def pack(cls, elements: Sequence[Rotation3d]) -> Rotation3d:
        coordinates = jnp.vstack([e.coordinates for e in elements])
        return cls(coordinates)

    @override
    def unpack(self) -> Sequence[Rotation3d]:
        if self.single:
            raise ValueError("Cannot unpack single element.")
        coordinates = self.coordinates.reshape(-1, 3, 3)
        return [Rotation3d(c) for c in coordinates]

    @staticmethod
    def normalize(matrix: Array) -> Array:
        return _normalize_rotation(matrix)

    @dispatch
    def __matmul__(self, other: Spin3d) -> Spin3d:
        return Spin3d(self.coordinates @ other.coordinates)

    @dispatch
    def __matmul__(self, other: Rotation3d) -> Rotation3d:
        return Rotation3d(self.coordinates @ other.coordinates)

    @override
    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="Rotation3d(")  # type: ignore
        return f"Rotation3d({repr})"


@final
class Spin3d(Element):
    """so(3) tangent vector."""

    coordinates: Array
    size: int = field(static=True, default=3)

    @override
    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(jnp.asarray(matrix))

    @classmethod
    def from_vector(cls, vector: ArrayLike):
        return cls(skew3(vector))

    @override
    def as_matrix(self):
        return self.coordinates

    def as_vector(self) -> Array:
        return _as_vector(self.coordinates)

    @override
    def flatten(self) -> Array:
        return _flatten(self.coordinates)

    @override
    @classmethod
    def unflatten(cls, flat: ArrayLike) -> Spin3d:
        return cls(_unflatten(jnp.asarray(flat)))

    @override
    @classmethod
    def pack(cls, elements: Sequence[Spin3d]) -> Spin3d:
        coordinates = jnp.vstack([e.coordinates for e in elements])
        return cls(coordinates)

    @override
    def unpack(self) -> Sequence[Spin3d]:
        if self.single:
            raise ValueError("Cannot unpack single element.")
        coordinates = self.coordinates.reshape(-1, 3, 3)
        return [Spin3d(c) for c in coordinates]

    @override
    @classmethod
    def eye(cls):
        return cls(jnp.zeros((3, 3)))

    def __matmul__(self, other: Rotation3d) -> Spin3d:
        return Spin3d(self.coordinates @ other.coordinates)

    @dispatch
    def __add__(self, other: Spin3d) -> Spin3d:
        return Spin3d(self.coordinates + other.coordinates)

    @dispatch
    def __add__(self, other: Array) -> Spin3d:
        if other.shape[-1] == 3:
            return Spin3d.from_vector(self.as_vector() + other)
        return Spin3d(self.coordinates + other)

    __radd__ = __add__

    @dispatch
    def __sub__(self, other: Spin3d) -> Spin3d:
        return Spin3d(self.coordinates - other.coordinates)

    @dispatch
    def __sub__(self, other: Array) -> Spin3d:
        if other.shape[-1] == 3:
            return Spin3d.from_vector(self.as_vector() - other)
        return Spin3d(self.coordinates - other)

    __rsub__ = __sub__

    @dispatch
    def __mul__(self, other: Spin3d) -> Spin3d:
        return Spin3d(self.coordinates * other.coordinates)

    @dispatch
    def __mul__(self, other: Array) -> Spin3d:
        return Spin3d(self.coordinates * other)

    __rmul__ = __mul__

    @override
    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="Spin3d(")  # type: ignore
        return f"Spin3d({repr})"


@partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _normalize_rotation(matrix: Array) -> Array:
    x_raw, y_raw, _ = jnp.split(matrix, 3)
    x_raw, y_raw = x_raw.squeeze(), y_raw.squeeze()
    x_norm = softnorm(x_raw)
    x = x_raw / jnp.maximum(x_norm, 1e-8)
    z = jnp.cross(x, y_raw)
    z_norm = softnorm(z)
    z = z / jnp.maximum(z_norm, 1e-8)
    y = jnp.cross(z, x)
    return jnp.stack((x, y, z)).squeeze()


@partial(jnp.vectorize, signature="(n,n)->(m)")
def _flatten(matrix: Array) -> Array:
    return matrix.flatten()


@partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten(flattened: Array) -> Array:
    return flattened.reshape((3, 3))


@partial(jnp.vectorize, signature="(n,n)->()")
def _rotation_angle(matrix: Array) -> RealScalarLike:
    cos = (jnp.trace(matrix) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)
    return theta


@partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(matrix: Array) -> Array:
    return vex3(matrix)
