import functools
from collections.abc import Sequence
from typing import Optional, cast

import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation as R
from jaxtyping import Array
from plum import dispatch

from ..._custom_types import ArrayLike, RealScalarLike
from ..._vecfuncs import softnorm
from .._element import AbstractGroupElement, AbstractTangentVector


class Rotation3d(AbstractGroupElement):
    value: Array

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

    def as_quat(self) -> Array:
        return R.from_matrix(self.value).as_quat()

    @classmethod
    def from_euler(cls, seq: str, angles: Sequence[float], degrees: bool = False):
        return cls(R.from_euler(seq, jnp.asarray(angles), degrees).as_matrix())

    def as_euler(self, seq: str, degrees: bool = False) -> Array:
        euler = R.from_matrix(self.value).as_euler(seq, degrees)
        return cast(Array, euler)

    @classmethod
    def eye(cls):
        return cls(jnp.eye(3))

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
    value: Array

    def __check_init__(self):
        if not isinstance(self.point, Rotation3d):
            raise ValueError(
                "The tangent vector point must be a `Rotation3d` instance!"
            )

    @classmethod
    def from_matrix(cls, mat: Array, point: Optional[Rotation3d] = None):
        point = Rotation3d.eye() if point is None else point
        return cls(point, mat)

    def as_matrix(self) -> Array:
        return self.value

    @classmethod
    def from_vector(cls, vec: Array, point: Optional[Rotation3d] = None): ...

    def as_vector(self) -> Array: ...

    def __repr__(self) -> str:
        repr = np.array2string(cast(np.ndarray, self.value), prefix="Spin3d(")
        return f"Spin3d({repr})"


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
