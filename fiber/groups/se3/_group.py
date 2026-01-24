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

from ..._custom_types import ArrayLike
from ..._linalg import skew3, vex3
from .._group import GroupElement
from ..so3._group import _normalize


class SE3(GroupElement):
    coordinates: Array
    shape: Sequence[int] = field(static=True, default=(4, 4))
    size: int = field(static=True, default=12)

    @classmethod
    def from_vector(cls, vector: ArrayLike):
        return cls(_from_vector_group(jnp.asarray(vector)))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(_from_matrix_group(jnp.asarray(matrix)))

    @classmethod
    def from_flat(cls, flat: ArrayLike):
        return cls(_from_flat_group(jnp.asarray(flat)))

    def as_matrix(self) -> Array:
        return self.coordinates

    def as_vector(self) -> Array:
        return _as_vector_group(self.coordinates)

    def as_flat(self) -> Array:
        return _as_flat(self.coordinates)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(4))

    @property
    def position(self) -> Array:
        return self.coordinates[..., :3, 3]

    @property
    def rotation(self) -> Array:
        return self.coordinates[..., :3, :3]

    def __matmul__(self, other):
        if is_se3_algebra_element(other):
            return se3(self.coordinates @ other.coordinates)
        return SE3(self.coordinates @ other.coordinates)

    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="SE3(")  # type: ignore
        return f"SE3({repr})"


class se3(GroupElement):
    coordinates: Array
    shape: Sequence[int] = field(static=True, default=(4, 4))
    size: int = field(static=True, default=6)

    @classmethod
    def from_vector(cls, vector: ArrayLike):
        return cls(_from_vector_algebra(jnp.asarray(vector)))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(jnp.asarray(matrix))

    @classmethod
    def from_flat(cls, flat: ArrayLike):
        return cls(_from_flat_algebra(jnp.asarray(flat)))

    def as_matrix(self) -> Array:
        return self.coordinates

    def as_vector(self) -> Array:
        return _as_vector_algebra(self.coordinates)

    def as_flat(self) -> Array:
        return _as_flat(self.coordinates)

    @classmethod
    def eye(cls):
        return cls(jnp.zeros((4, 4)))

    @property
    def linear(self) -> Array:
        return self.coordinates[..., :3, 3]

    @property
    def angular(self) -> Array:
        return vex3(self.coordinates[..., :3, :3])

    def __matmul__(self, other):
        return se3(self.coordinates @ other.coordinates)

    def __add__(self, other):
        if is_se3_algebra_element(other):
            return se3(self.coordinates + other.coordinates)
        return se3.from_vector(self.as_vector() + other)

    __radd__ = __add__

    def __sub__(self, other):
        if is_se3_algebra_element(other):
            return se3(self.coordinates - other.coordinates)
        return se3.from_vector(self.as_vector() - other)

    __rsub__ = __sub__

    def __mul__(self, other):
        if is_se3_algebra_element(other):
            return se3(self.coordinates * other.coordinates)
        return se3.from_vector(self.as_vector() * other)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="se3(")  # type: ignore
        return f"se3({repr})"


def is_se3_group_element(value: Any) -> bool:
    return isinstance(value, SE3)


def is_se3_algebra_element(value: Any) -> bool:
    return isinstance(value, se3)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector_group(vector: Array) -> Array:
    trans, quat = jnp.split(vector, (3,), axis=-1)
    rot = R.from_quat(quat).as_matrix()
    return jnp.block([[rot, trans.reshape(3, 1)], [jnp.zeros(3), 1]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix_group(matrix: Array) -> Array:
    return matrix.at[:3, :3].set(_normalize(matrix[:3, :3]))


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_flat_group(flat: Array) -> Array:
    trans, rot = jnp.split(flat, (3,))
    mat = jnp.block([[rot.reshape((3, 3)), trans.reshape(3, 1)], [jnp.zeros(3), 1]])
    return _from_matrix_group(mat)


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector_group(matrix: Array) -> Array:
    trans, rot = matrix[:3, 3], matrix[:3, :3]
    quat = R.from_matrix(rot).as_quat()
    return jnp.concatenate([trans, quat])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector_algebra(vector: Array) -> Array:
    v, w = jnp.split(vector, 2)
    return jnp.block([[skew3(w), v.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_flat_algebra(flat: Array) -> Array:
    v, w = jnp.split(flat, (3,))
    return jnp.block([[w.reshape((3, 3)), v.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector_algebra(matrix: Array) -> Array:
    v, w = matrix[:3, 3], vex3(matrix[:3, :3])
    return jnp.concatenate([v, w])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_flat(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:3, 3], matrix[:3, :3].flatten()])
