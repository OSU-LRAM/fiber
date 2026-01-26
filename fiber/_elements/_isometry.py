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
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from equinox import field
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array

from .._custom_types import ArrayLike
from ..linalg import softnorm
from ._element import GroupElement
from ._twist import Twist, is_twist


class Isometry(GroupElement):
    coordinates: Array
    shape: Sequence[int] = field(static=True, default=(4, 4))
    size: int = field(static=True, default=12)

    @classmethod
    def from_vector(cls, vector: ArrayLike):
        return cls(_from_vector(jnp.asarray(vector)))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike, normalize: bool = True):
        if normalize:
            return cls(_from_matrix(jnp.asarray(matrix)))
        return cls(jnp.asarray(matrix))

    @classmethod
    def unflatten(cls, flat: ArrayLike, normalize: bool = True):
        if normalize:
            return cls(_unflatten_normalized(jnp.asarray(flat)))
        return cls(_unflatten(jnp.asarray(flat)))

    def as_matrix(self) -> Array:
        return self.coordinates

    def as_vector(self) -> Array:
        return _as_vector(self.coordinates)

    def flatten(self) -> Array:
        return _flatten(self.coordinates)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(4))

    @property
    def position(self) -> Array:
        return self.coordinates[..., :3, 3]

    @property
    def rotation(self) -> R:
        return R.from_matrix(self.coordinates[..., :3, :3])

    def __matmul__(self, other):
        if is_twist(other):
            return Twist(self.coordinates @ other.coordinates)
        return Isometry(self.coordinates @ other.coordinates)

    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="Isometry(")  # type: ignore
        return f"Isometry({repr})"

    def __getitem__(self, key: int):
        if self.single:
            raise ValueError("Single transformation is not subscriptable.")
        return Isometry.from_matrix(self.coordinates[key])

    def __iter__(self):
        if self.single:
            raise ValueError("Single transformation is not iterable.")
        return (Isometry.from_matrix(c) for c in self.coordinates)


def is_isometry(value: Any) -> bool:
    return isinstance(value, Isometry)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vector: Array) -> Array:
    trans, quat = jnp.split(vector, (3,), axis=-1)
    rot = R.from_quat(quat).as_matrix()
    return jnp.block([[rot, trans.reshape(3, 1)], [jnp.zeros(3), 1]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix(matrix: Array) -> Array:
    return matrix.at[:3, :3].set(_normalize_rotation(matrix[:3, :3]))


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten_normalized(flat: Array) -> Array:
    trans, rot = jnp.split(flat, (3,))
    mat = jnp.block([[rot.reshape((3, 3)), trans.reshape(3, 1)], [jnp.zeros(3), 1]])
    return _from_matrix(mat)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten(flat: Array) -> Array:
    trans, rot = jnp.split(flat, (3,))
    mat = jnp.block([[rot.reshape((3, 3)), trans.reshape(3, 1)], [jnp.zeros(3), 1]])
    return mat


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
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


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(matrix: Array) -> Array:
    trans, rot = matrix[:3, 3], matrix[:3, :3]
    quat = R.from_matrix(rot).as_quat()
    return jnp.concatenate([trans, quat])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _flatten(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:3, 3], matrix[:3, :3].flatten()])
