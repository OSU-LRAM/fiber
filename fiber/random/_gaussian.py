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

from functools import singledispatch

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from .._elements import Isometry, Twist
from .._elements._twist import _from_vector
from .._operations import expm


def _sample_algebra(key: PRNGKeyArray, num_samples: int | None = None) -> Array:
    shape = (num_samples,) if num_samples is not None else None
    return jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), shape)


@singledispatch
def left_gaussian(key, mean, num_samples):
    return _left_gaussian(key, mean, num_samples)


def _left_gaussian(
    key: PRNGKeyArray, mean: Array, num_samples: int | None = None
) -> tuple[Array, Array]:
    s = _sample_algebra(key, num_samples)
    gs = jax.vmap(lambda x: mean @ expm(x))(s)
    vs = _from_vector(s)
    return gs, vs


@left_gaussian.register
def _left_gaussian_type(
    key: PRNGKeyArray, mean: Isometry, num_samples: int | None = None
) -> tuple[Isometry, Twist]:
    gs, vs = _left_gaussian(key, mean.coordinates, num_samples)
    return Isometry.from_matrix(gs), Twist.from_vector(vs)


@jaxtyped(typechecker=beartype)
def right_gaussian(
    key: PRNGKeyArray, mean: Num[Array, "n n"], num_samples: int
) -> tuple[Num[Array, "m n n"], Num[Array, "m n n"]]:
    vels = _sample_lie_algebra(key, num_samples)
    gs = jax.vmap(lambda ξ: jax.scipy.linalg.expm(ξ) @ mean)(vels)
    g_circs = jax.vmap(to_matrix)(vels)
    return gs, g_circs
