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
from typing import Any

import jax.numpy as jnp
import numpy as np
from equinox import field
from jaxtyping import Array

from .._custom_types import ArrayLike
from ..linalg import skew3, vex3
from ._element import GroupElement


class Twist(GroupElement):
    coordinates: Array
    size: int = field(static=True, default=6)

    @classmethod
    def from_vector(cls, vector: ArrayLike):
        return cls(_from_vector(jnp.asarray(vector)))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(jnp.asarray(matrix))

    @classmethod
    def unflatten(cls, flat: ArrayLike):
        return cls(_unflatten(jnp.asarray(flat)))

    def as_matrix(self) -> Array:
        return self.coordinates

    def as_vector(self) -> Array:
        return _as_vector(self.coordinates)

    def flatten(self) -> Array:
        return _flatten(self.coordinates)

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
        return Twist(self.coordinates @ other.coordinates)

    def __add__(self, other):
        if is_algebra_element(other):
            return Twist(self.coordinates + other.coordinates)
        return Twist.from_vector(self.as_vector() + other)

    __radd__ = __add__

    def __sub__(self, other):
        if is_algebra_element(other):
            return Twist(self.coordinates - other.coordinates)
        return Twist.from_vector(self.as_vector() - other)

    __rsub__ = __sub__

    def __mul__(self, other):
        if is_algebra_element(other):
            return Twist(self.coordinates * other.coordinates)
        return Twist.from_vector(self.as_vector() * other)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="Twist(")  # type: ignore
        return f"Twist({repr})"

    def __getitem__(self, key: int):
        if self.single:
            raise ValueError("Single transformation is not subscriptable.")
        return Twist.from_matrix(self.coordinates[key])

    def __iter__(self):
        if self.single:
            raise ValueError("Single transformation is not iterable.")
        return (Twist.from_matrix(c) for c in self.coordinates)


def is_algebra_element(value: Any) -> bool:
    return isinstance(value, Twist)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vector: Array) -> Array:
    v, w = jnp.split(vector, 2)
    return jnp.block([[skew3(w), v.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _unflatten(flat: Array) -> Array:
    v, w = jnp.split(flat, (3,))
    return jnp.block([[w.reshape((3, 3)), v.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(matrix: Array) -> Array:
    v, w = matrix[:3, 3], vex3(matrix[:3, :3])
    return jnp.concatenate([v, w])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _flatten(matrix: Array) -> Array:
    return jnp.concatenate([matrix[:3, 3], matrix[:3, :3].flatten()])
