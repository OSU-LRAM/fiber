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
from typing import override

import jax.numpy as jnp
from equinox import field
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array

from ..._custom_types import ArrayLike
from ...linalg._vecfuncs import softnorm
from .._element import Element


class Rotation3d(Element):
    """SO(3) group element."""

    coordinates: Array
    dim: int = field(static=True, default=3)
    size: int = field(static=True, default=9)

    @override
    @classmethod
    def from_matrix(cls, matrix: ArrayLike):
        return cls(R.from_matrix(jnp.asarray(matrix)).as_matrix())

    @classmethod
    def from_euler(cls, seq: str, angles: Sequence[int | float], degrees: bool = False):
        return cls(R.from_euler(seq, jnp.asarray(angles), degrees).as_matrix())

    @classmethod
    def from_quat(cls, quat: Sequence[int | float]):
        return cls(R.from_quat(jnp.asarray(quat)).as_matrix())

    @override
    def as_matrix(self) -> Array:
        return self.coordinates

    def as_euler(self, seq: str, degrees: bool = False) -> ArrayLike:
        return R.from_matrix(self.coordinates).as_euler(seq, degrees)

    def as_quat(self) -> ArrayLike:
        return R.from_matrix(self.coordinates).as_quat()

    @override
    def flatten(self) -> Array: ...

    @override
    @classmethod
    def unflatten(cls, flat: ArrayLike): ...

    @override
    @classmethod
    def eye(cls) -> Rotation3d: ...

    @override
    @classmethod
    def pack(cls, elements: Sequence[Rotation3d]) -> Rotation3d: ...

    @override
    def unpack(self) -> Sequence[Rotation3d]: ...

    @staticmethod
    def normalize(matrix: Array) -> Array:
        return _normalize(matrix)


@partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _normalize(matrix: Array) -> Array:
    x_raw, y_raw, _ = jnp.split(matrix, 3)
    x_raw, y_raw = x_raw.squeeze(), y_raw.squeeze()
    x_norm = softnorm(x_raw)
    x = x_raw / jnp.maximum(x_norm, 1e-8)
    z = jnp.cross(x, y_raw)
    z_norm = softnorm(z)
    z = z / jnp.maximum(z_norm, 1e-8)
    y = jnp.cross(z, x)
    return jnp.stack((x, y, z)).squeeze()
