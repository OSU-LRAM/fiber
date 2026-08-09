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

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._vecfuncs import softclip, softnorm, vex3


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def inv(w: Array) -> Array:
    return w.T


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def adj(w: Array, v: Array) -> Array:
    return w @ v - v @ w


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def adj_op(w: Array) -> Array:
    return w


@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def dadj(w: Array, p: Array):
    return dadj_op(w) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dadj_op(w: Array) -> Array:
    return -adj_op(w).T


@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def dadj_inv(w: Array, p: Array) -> Array:
    return -dadj(w, p)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dadj_inv_op(w: Array):
    return -dadj_op(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj(g: Array, w: Array) -> Array:
    return g @ w @ inv(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def Adj_op(g: Array) -> Array:
    return g


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj_inv(g: Array, w: Array) -> Array:
    return inv(g) @ w @ g


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def Adj_inv_op(g: Array) -> Array:
    return Adj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def dAdj(g: Array, p: Array) -> Array:
    return dAdj_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dAdj_op(g: Array) -> Array:
    return Adj_op(inv(g)).T


@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def dAdj_inv(g: Array, p: Array) -> Array:
    return dAdj_inv_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dAdj_inv_op(g: Array) -> Array:
    return dAdj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def expm(w: Array) -> Array:
    theta = softnorm(vex3(w))
    sin = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 6) + (theta**4 / 120),
        lambda: jnp.sin(theta) / theta,
    )
    cos = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 0.5 - (theta**2 / 24) + (theta**4 / 720),
        lambda: (1 - jnp.cos(theta)) / (theta**2),
    )
    return jnp.eye(3) + sin * w + cos * (w @ w)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dexpm(w: Array) -> Array:
    theta = softnorm(vex3(w))
    a = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 6) + (theta**4 / 120),
        lambda: jnp.sin(theta) / theta,
    )
    b = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 0.5 - (theta**2 / 24) + (theta**4 / 720),
        lambda: (1 - jnp.cos(theta)) / (theta**2),
    )
    c = (1 - a) / (theta**2)
    return jnp.eye(3) + b * w + c * (w @ w)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    cos = (jnp.trace(g) - 1) / 2
    cos = softclip(cos, -1, 1)
    theta = jnp.arccos(cos)
    w_hat = jnp.zeros((3, 3))
    return jax.lax.cond(
        jnp.sin(theta) < 1e-6,
        lambda: w_hat,
        lambda: 0.5 * theta / jnp.sin(theta) * (g - g.T),
    )


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dlogm(w: Array) -> Array:
    theta = softnorm(vex3(w))
    cos = jnp.cos(theta)
    a = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 6) + (theta**4 / 120),
        lambda: jnp.sin(theta) / theta,
    )
    b = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 0.5 - (theta**2 / 24) + (theta**4 / 720),
        lambda: (1 - cos) / (theta**2),
    )
    e = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: (1 / 12) + (theta**2 / 720),
        lambda: (b - 0.5 * a) / (1 - cos),
    )
    return jnp.eye(3) - 0.5 * w + e * (w @ w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def lplus(g: Array, w: Array) -> Array:
    return expm(w) @ g


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def rplus(g: Array, w: Array) -> Array:
    return g @ expm(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def lminus(g: Array, h: Array) -> Array:
    return logm(g @ inv(h))


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def rminus(g: Array, h: Array) -> Array:
    return logm(inv(h) @ g)
