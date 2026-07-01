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

from ..._vecfuncs import vex2
from .. import so2

_J = jnp.array([[0.0, -1.0], [1.0, 0.0]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def inv(w: Array) -> Array:
    return jnp.linalg.inv(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def adj(w: Array, v: Array) -> Array:
    return w @ v - v @ w


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def adj_op(w: Array) -> Array:
    lin, ang = w[:2, 2], w[:2, :2]
    perp = _J @ lin
    return jnp.block([[ang, -perp.reshape(2, 1)], [jnp.zeros((1, 3))]])


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
    pos, rot = g[:2, 2], g[:2, :2]
    perp = _J @ pos
    return jnp.block([[rot, -perp.reshape(2, 1)], [jnp.array([[0.0, 0.0, 1.0]])]])


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
    lin, ang_hat = w[:2, 2], w[:2, :2]
    theta = vex2(ang_hat)
    A = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 6) + (theta**4 / 120),
        lambda: jnp.sin(theta) / theta,
    )
    B = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: theta * (0.5 - (theta**2 / 24) + (theta**4 / 720)),
        lambda: (1 - jnp.cos(theta)) / theta,
    )
    V = A * jnp.eye(2) + B * _J
    return jnp.block([[so2.expm(ang_hat), (V @ lin).reshape(2, 1)], [jnp.zeros(2), 1]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dexpm(w: Array) -> Array:
    A = adj_op(w)
    theta = vex2(w[:2, :2])
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
    c = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: (1 / 6) + (theta**2 / 120),
        lambda: (1 - a) / (theta**2),
    )
    return jnp.eye(3) + b * A + c * (A @ A)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    pos, rot = g[:2, 2], g[:2, :2]
    ang_hat = so2.logm(rot)
    theta = vex2(ang_hat)
    half = 0.5 * theta
    c = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 12) - (theta**4 / 720),
        lambda: half / jnp.tan(half),
    )
    W = c * jnp.eye(2) - half * _J
    return jnp.block([[ang_hat, (W @ pos).reshape(2, 1)], [jnp.zeros(3)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dlogm(w: Array) -> Array:
    A = adj_op(w)
    theta = vex2(w[:2, :2])
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
    e = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: (1 / 12) + (theta**2 / 720),
        lambda: (b - 0.5 * a) / (1 - jnp.cos(theta)),
    )
    return jnp.eye(3) - 0.5 * A + e * (A @ A)


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


def split_bundle(s: Array, axis: int = 0) -> tuple[Array, Array]:
    g, w = jnp.split(s, (6,), axis=axis)
    return g, w
