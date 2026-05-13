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

from ..._custom_types import ArrayLike
from ..._vecfuncs import skew3, vex3
from .._element import AbstractGroupElement, AbstractTangentVector
from ..so3._element import Rotation3d, _normalize_rotation


class Isometry3d(AbstractGroupElement):
    value: Array = eqx.field(converter=jnp.asarray)

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
    def eye(cls):
        return cls(jnp.eye(4))

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

    def __check_init__(self):
        if not isinstance(self.point, Isometry3d):
            raise ValueError(
                "The tangent vector point must be a `Isometry3d` instance!"
            )

    @classmethod
    def from_matrix(cls, mat: Array, point: Optional[Isometry3d] = None):
        point = Isometry3d.eye() if point is None else point
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: ArrayLike, point: Optional[Isometry3d] = None):
        point = Isometry3d.eye() if point is None else point
        return cls(point, _from_vector(jnp.asarray(vec)))

    def as_vector(self) -> Array:
        return _as_vector(self.value)

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
    def __div__(self, other: Twist3d) -> Twist3d:  # type: ignore[reportRedeclaration]
        return Twist3d(self.point, self.value / other.value)

    @dispatch
    def __div__(self, other: ArrayLike) -> Twist3d:
        return Twist3d(self.point, self.value / other)

    __rdiv__ = __div__

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Twist3d(")
        return f"Twist3d({repr})"


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _from_matrix(mat: Array) -> Array:
    return mat.at[:3, :3].set(_normalize_rotation(mat[:3, :3]))


def _from_euclidean(vec: Array, seq: str, degrees: bool) -> Array:
    pos, angles = jnp.split(vec, 2, axis=-1)
    rot = R.from_euler(seq, angles, degrees).as_matrix()
    return jnp.block([[rot, pos.reshape(3, 1)], [jnp.zeros(3), 1]])


def _as_euclidean(mat: Array, seq: str, degrees: bool) -> Array:
    pos = mat[:3, 3]
    angles = R.from_matrix(mat[:3, :3]).as_euler(seq, degrees)
    return jnp.concatenate([pos, cast(Array, angles)], axis=-1)


@functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def _from_vector(vec: Array) -> Array:
    lin, ang = jnp.split(vec, 2)
    return jnp.block([[skew3(ang), lin.reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def _as_vector(mat: Array) -> Array:
    lin, ang = mat[:3, 3], vex3(mat[:3, :3])
    return jnp.concatenate([lin, ang])
