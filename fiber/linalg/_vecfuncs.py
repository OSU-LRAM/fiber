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
from typing import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._custom_types import RealScalarLike
from .._epsilon import ε

_Axis = None | int | Sequence[int]


def softnorm(g: ArrayLike, axis: _Axis = None) -> ArrayLike:
    return jnp.sqrt(jnp.sum(g**2, axis=axis) + ε)


# @functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def skew3(x: Array) -> Array:
    return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


# @functools.partial(jnp.vectorize, signature="(n)->(m,m)")
def skew2(x: RealScalarLike) -> Array:
    return jnp.array([[0, -x], [x, 0]])


# @functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def vex3(x: Array) -> Array:
    return jnp.array([x[2, 1], x[0, 2], x[1, 0]])


# @functools.partial(jnp.vectorize, signature="(n,n)->(m)")
def vex2(x: Array) -> RealScalarLike:
    return jnp.asarray(x[1, 0])
