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


def _sample(num_samples: int | None = None, *, key: PRNGKeyArray) -> Array:
    shape = (num_samples,) if num_samples is not None else None
    return jax.random.multivariate_normal(key, jnp.zeros(6), jnp.eye(6), shape)


@singledispatch
def left_gaussian(mean: Array, num_samples: int = 1, *, key):
    return _left_gaussian(mean, num_samples, key=key)


def _left_gaussian(
    mean: Array,
    num_samples: int,
    *,
    key: PRNGKeyArray,
) -> tuple[Array, Array]:
    s = _sample(num_samples, key=key)
    vs = _from_vector(s)
    gs = jax.vmap(lambda x: mean @ expm(x))(vs)
    return gs, vs


@left_gaussian.register  # type: ignore
def _left_gaussian_type(
    mean: Isometry, num_samples: int = 1, *, key: PRNGKeyArray
) -> tuple[Isometry, Twist]:
    gs, vs = _left_gaussian(mean.coordinates, num_samples, key=key)
    return Isometry.from_matrix(gs), Twist.from_matrix(vs)


@singledispatch
def right_gaussian(mean: Array, num_samples: int = 1, *, key):
    return _right_gaussian(mean, num_samples, key=key)


def _right_gaussian(
    mean: Array, num_samples: int, *, key: PRNGKeyArray
) -> tuple[Array, Array]:
    s = _sample(num_samples, key=key)
    vs = _from_vector(s)
    gs = jax.vmap(lambda x: expm(x) @ mean)(vs)
    return gs, vs


@right_gaussian.register  # type: ignore
def _right_gaussian_type(
    mean: Isometry, num_samples: int = 1, *, key: PRNGKeyArray
) -> tuple[Isometry, Twist]:
    gs, vs = _right_gaussian(mean.coordinates, num_samples, key=key)
    return Isometry.from_matrix(gs), Twist.from_matrix(vs)
