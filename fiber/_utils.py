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
from functools import singledispatch

import jax.numpy as jnp
from jaxtyping import Array

from ._elements import Isometry, Twist


def split_state(x, axis: int = 0) -> tuple[Isometry, Twist]:
    g, v = jnp.split(x, (Isometry.size,), axis=axis)  # type: ignore
    return Isometry.unflatten(g), Twist.from_vector(v)


@singledispatch
def join_state(g, v):
    return _join_state(g, v)


@functools.partial(jnp.vectorize, signature="(n),(m)->(k)")
def _join_state(g: Array, v: Array) -> Array:
    return jnp.concatenate([g, v])


@join_state.register
def _(g: Isometry, v: Twist) -> Array:
    return _join_state(g.flatten(), v.as_vector())


@join_state.register
def _(g: Twist, v: Twist) -> Array:
    # this is a special case that shows up in integrators
    return _join_state(g.flatten(), v.as_vector())
