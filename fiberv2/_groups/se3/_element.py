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
import numpy as np
from equinox import field
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike
from .._element import Element
from ._tangent import Twist3d


class Isometry3d(Element):
    """SE(3) group element."""

    coordinates: Array
    dim: int = field(static=True, default=3)
    size: int = field(static=True, default=12)

    @classmethod
    @override
    def from_vector(cls, vector: ArrayLike):
        return cls(_from_vector(jnp.asarray(vector)))

    @classmethod
    def from_matrix(cls, matrix: ArrayLike, normalize: bool = True): ...

    def as_matrix(self) -> Array: ...

    def as_vector(self) -> Array: ...

    def flatten(self) -> Array: ...

    @classmethod
    def unflatten(cls, flat: ArrayLike): ...

    @classmethod
    def eye(cls): ...

    @classmethod
    def pack(cls, elements: Sequence): ...

    def unpack(self): ...

    @dispatch
    def __matmul__(self, other: Isometry3d) -> Isometry3d:
        return Isometry3d(self.coordinates @ other.coordinates)

    @dispatch
    def __matmul__(self, other: Twist3d) -> Twist3d:
        return Twist3d(self.coordinates @ other.coordinates)

    @override
    def __repr__(self) -> str:
        repr = np.array2string(self.coordinates, prefix="Isometry3d(")  # type: ignore
        return f"Isometry3d({repr})"

    def __getitem__(self, key: int):
        if self.single:
            raise ValueError("Single transformation is not subscriptable.")
        return Isometry3d.from_matrix(self.coordinates[key])

    def __iter__(self):
        if self.single:
            raise ValueError("Single transformation is not iterable.")
        return (Isometry3d.from_matrix(c) for c in self.coordinates)


@partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vector: Array) -> Array:
    trans, quat = jnp.split(vector, (3,), axis=-1)
    rot = R.from_quat(quat).as_matrix()
    return jnp.block([[rot, trans.reshape(3, 1)], [jnp.zeros(3), 1]])


@partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix(matrix: Array) -> Array:
    return matrix.at[:3, :3].set(_normalize_rotation(matrix[:3, :3]))
