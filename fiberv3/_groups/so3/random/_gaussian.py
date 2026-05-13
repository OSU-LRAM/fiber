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
from typing import Optional

import jax.numpy as jnp
import jax.random as jr
import optimistix as optx
from jaxtyping import Array, PRNGKeyArray

from ...._vecfuncs import skew3, softnorm, vex3
from .._ops import expm, logm, lplus, rminus, rplus


def sample_lie_algebra(
    key: PRNGKeyArray,
    cov: Array,
    shape: Optional[Sequence[int]] = None,
    method: str = "cholesky",
) -> Array:
    return jr.multivariate_normal(key, jnp.zeros(3), cov, shape, method=method)


def gaussian(
    key: PRNGKeyArray,
    mean: Array,
    cov: Array,
    shape: Optional[Sequence[int]] = None,
    method: str = "cholesky",
    left: bool = True,
) -> tuple[Array, Array]:
    samples = sample_lie_algebra(key, cov, shape, method)
    vectors = skew3(samples)

    if left:
        elements = rplus(mean, vectors)
    else:
        elements = lplus(mean, vectors)

    return elements, vectors


def normal(
    key: PRNGKeyArray,
    shape: Optional[Sequence[int]] = None,
    left: bool = True,
) -> tuple[Array, Array]:
    return gaussian(key, jnp.eye(3), jnp.eye(3), shape, left=left)


def mean(
    samples: Array,
    rtol: float = 1e-6,
    atol=1e-6,
    max_steps: int = 100,
    throw: bool = True,
) -> Array:
    def residuals(mean, samples):
        errors = vex3(rminus(samples, mean))
        return softnorm(jnp.sum(errors, axis=0))

    # construct the initial guess to warm-start the solver
    exp_coords = vex3(logm(samples))
    init_mean = jnp.mean(exp_coords, axis=0)
    y0 = expm(skew3(init_mean))

    # find the mean using a root-finder
    sol = optx.root_find(
        residuals,
        optx.Newton(rtol, atol),
        y0,
        args=samples,
        max_steps=max_steps,
        throw=throw,
    )

    return sol.value


def cov(mean: Array, samples: Array) -> Array:
    distances = vex3(rminus(samples, mean))
    sigma = jnp.divide(jnp.einsum("ni,nj->ij", distances, distances), len(samples))
    return sigma
