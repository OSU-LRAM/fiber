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

import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from plum import dispatch

from .._groups import se2, se3, so2, so3
from .._groups.se2 import Isometry2d, Twist2d
from .._groups.se3 import Isometry3d, Twist3d
from .._groups.so2 import Rotation2d, Spin2d
from .._groups.so3 import Rotation3d, Spin3d


@dispatch
def gaussian(  # type: ignore[reportRedeclaration]
    key: PRNGKeyArray,
    mean: Rotation2d,
    cov: Array,
    shape: Sequence[int] | None = None,
    method: str = "cholesky",
    left: bool = True,
) -> tuple[Rotation2d, Spin2d]:
    elements, vectors = so2.random.gaussian(key, mean.value, cov, shape, method, left)
    point = Rotation2d(jnp.broadcast_to(mean.value, vectors.shape))
    return Rotation2d.from_matrix(elements), Spin2d.from_matrix(vectors, point)


@dispatch
def gaussian(  # type: ignore[reportRedeclaration]
    key: PRNGKeyArray,
    mean: Rotation3d,
    cov: Array,
    shape: Sequence[int] | None = None,
    method: str = "cholesky",
    left: bool = True,
) -> tuple[Rotation3d, Spin3d]:
    elements, vectors = so3.random.gaussian(key, mean.value, cov, shape, method, left)
    point = Rotation3d(jnp.broadcast_to(mean.value, vectors.shape))
    return Rotation3d.from_matrix(elements), Spin3d.from_matrix(vectors, point)


@dispatch
def gaussian(  # type: ignore[reportRedeclaration]
    key: PRNGKeyArray,
    mean: Isometry2d,
    cov: Array,
    shape: Sequence[int] | None = None,
    method: str = "cholesky",
    left: bool = True,
) -> tuple[Isometry2d, Twist2d]:
    elements, vectors = se2.random.gaussian(key, mean.value, cov, shape, method, left)
    point = Isometry2d(jnp.broadcast_to(mean.value, vectors.shape))
    return Isometry2d.from_matrix(elements), Twist2d.from_matrix(vectors, point)


@dispatch
def gaussian(  # type: ignore[reportRedeclaration]
    key: PRNGKeyArray,
    mean: Isometry3d,
    cov: Array,
    shape: Sequence[int] | None = None,
    method: str = "cholesky",
    left: bool = True,
) -> tuple[Isometry3d, Twist3d]:
    elements, vectors = se3.random.gaussian(key, mean.value, cov, shape, method, left)
    point = Isometry3d(jnp.broadcast_to(mean.value, vectors.shape))
    return Isometry3d.from_matrix(elements), Twist3d.from_matrix(vectors, point)


@dispatch
def mean(  # type: ignore[reportRedeclaration]
    samples: Rotation2d,
    rtol: float = 1e-6,
    atol=1e-6,
    max_steps: int = 100,
    throw: bool = True,
) -> Rotation2d:
    est_mean = so2.random.mean(samples.value, rtol, atol, max_steps, throw)
    return Rotation2d.from_matrix(est_mean)


@dispatch
def mean(  # type: ignore[reportRedeclaration]
    samples: Rotation3d,
    rtol: float = 1e-6,
    atol=1e-6,
    max_steps: int = 100,
    throw: bool = True,
) -> Rotation3d:
    est_mean = so3.random.mean(samples.value, rtol, atol, max_steps, throw)
    return Rotation3d.from_matrix(est_mean)


@dispatch
def mean(  # type: ignore[reportRedeclaration]
    samples: Isometry2d,
    rtol: float = 1e-6,
    atol=1e-6,
    max_steps: int = 100,
    throw: bool = True,
) -> Isometry2d:
    est_mean = se2.random.mean(samples.value, rtol, atol, max_steps, throw)
    return Isometry2d.from_matrix(est_mean)


@dispatch
def mean(  # type: ignore[reportRedeclaration]
    samples: Isometry3d,
    rtol: float = 1e-6,
    atol=1e-6,
    max_steps: int = 100,
    throw: bool = True,
) -> Isometry3d:
    est_mean = se3.random.mean(samples.value, rtol, atol, max_steps, throw)
    return Isometry3d.from_matrix(est_mean)


@dispatch
def cov(mean: Rotation2d, samples: Rotation2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.random.cov(mean.value, samples.value)


@dispatch
def cov(mean: Rotation3d, samples: Rotation3d) -> Array:  # type: ignore[reportRedeclaration]
    return so3.random.cov(mean.value, samples.value)


@dispatch
def cov(mean: Isometry2d, samples: Isometry2d) -> Array:  # type: ignore[reportRedeclaration]
    return se2.random.cov(mean.value, samples.value)


@dispatch
def cov(mean: Isometry3d, samples: Isometry3d) -> Array:  # type: ignore[reportRedeclaration]
    return se3.random.cov(mean.value, samples.value)
