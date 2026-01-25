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

import jax.numpy as jnp
from jaxtyping import Array

from .._linalg import skew3


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def adj(g: Array, h: Array) -> Array:
    return g @ h - h @ g


@functools.partial(jnp.vectorize, signature="(n),(n)->(m,m)")
def adj_op(v: Array, w: Array) -> Array:
    return jnp.block([[skew3(w), skew3(v)], [jnp.zeros((3, 3)), skew3(w)]])


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj(g: Array, h: Array) -> Array:
    return g @ h @ jnp.linalg.inv(g)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adji(g: Array, h: Array) -> Array:
    return jnp.linalg.inv(g) @ h @ g


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dadj(g: Array, p: Array) -> Array:
    return dadj_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dadj_op(g: Array) -> Array:
    return adj_op(g).T


def dAdj(): ...


def dAdj_op(): ...


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dadji(g: Array, p: Array) -> Array:
    return -dadj(g, p)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dadji_op(g: Array) -> Array:
    return -adj_op(g).T


def dAdji(): ...


def dAdji_op(): ...


def rplus(): ...


def rminus(): ...


def dexp(g: Array, h: Array, order: int = 3): ...


def dlog(g: Array, h: Array, order: int = 3): ...
