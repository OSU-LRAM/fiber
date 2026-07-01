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

from ..._vecfuncs import skew3, softnorm, vex3
from .. import so3


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def inv(w: Array) -> Array:
    return jnp.linalg.inv(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def adj(w: Array, v: Array) -> Array:
    return w @ v - v @ w


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def adj_op(w: Array) -> Array:
    lin, ang = w[:3, 3], w[:3, :3]
    return jnp.block([[ang, skew3(lin)], [jnp.zeros((3, 3)), ang]])


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dadj(w: Array, p: Array):
    return dadj_op(w) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dadj_op(w: Array) -> Array:
    return -adj_op(w).T


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dadj_inv(w: Array, p: Array) -> Array:
    return -dadj(w, p)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dadj_inv_op(w: Array):
    return -dadj_op(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj(g: Array, w: Array) -> Array:
    return g @ w @ inv(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def Adj_op(g: Array) -> Array:
    pos, rot = g[:3, 3], g[:3, :3]
    return jnp.block([[rot, skew3(pos)], [jnp.zeros_like(rot), rot]])


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj_inv(g: Array, w: Array) -> Array:
    return inv(g) @ w @ g


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def Adj_inv_op(g: Array) -> Array:
    return Adj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dAdj(g: Array, p: Array) -> Array:
    return dAdj_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dAdj_op(g: Array) -> Array:
    return Adj_op(inv(g)).T


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dAdj_inv(g: Array, p: Array) -> Array:
    return dAdj_inv_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dAdj_inv_op(g: Array) -> Array:
    return dAdj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def expm(w: Array) -> Array:
    lin, ang = w[:3, 3], w[:3, :3]
    theta = softnorm(vex3(ang))
    A = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 1 - (theta**2 / 6) + (theta**4 / 120),
        lambda: jnp.sin(theta) / theta,
    )
    B = jax.lax.cond(
        jnp.isclose(theta, 0.0),  # type: ignore
        lambda: 0.5 - (theta**2 / 24) + (theta**4 / 720),
        lambda: (1 - jnp.cos(theta)) / (theta**2),
    )
    C = (1 - A) / theta**2
    V = jnp.eye(3) + B * ang + C * (ang @ ang)
    return jnp.block([[so3.expm(ang), (V @ lin).reshape(3, 1)], [jnp.zeros(3), 1]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dexpm(w: Array) -> Array:
    lin, ang_hat = w[:3, 3], w[:3, :3]
    lin_hat, ang = skew3(lin), vex3(ang_hat)
    theta = softnorm(ang)
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
    W = (
        (c - b) * jnp.eye(3)
        + ((a - 2 * b) / theta**2) * ang_hat
        + ((b - 3 * c) / theta**2) * (ang_hat @ ang_hat)
    )
    V = b * lin_hat + c * (ang @ lin.T + lin @ ang.T) + (ang.T @ lin * W)
    D = a * jnp.eye(3) + b * ang_hat + c * (ang_hat @ ang_hat)
    return jnp.block([[D, V], [jnp.zeros_like(V), D]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    pos, rot = g[:3, 3], g[:3, :3]
    ang_hat = so3.logm(rot)
    ang = vex3(ang_hat)
    jac = jnp.eye(3)
    V = jax.lax.cond(
        jnp.isclose(softnorm(ang), 0.0),
        lambda: jac,
        lambda: (
            jac
            - 0.5 * ang_hat
            + (
                (1 / (softnorm(ang) ** 2))
                - (1 + jnp.cos(softnorm(ang)))
                / (2 * softnorm(ang) * jnp.sin(softnorm(ang)))
            )
            * (ang_hat @ ang_hat)
        ),
    )
    return jnp.block([[ang_hat, (V @ pos).reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dlogm(w: Array) -> Array:
    lin, ang_hat = w[:3, 3], w[:3, :3]
    lin_hat, ang = skew3(lin), vex3(ang_hat)
    theta = softnorm(ang)
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
    W = (
        (c - b) * jnp.eye(3)
        + ((a - 2 * b) / theta**2) * ang_hat
        + ((b - 3 * c) / theta**2) * (ang_hat @ ang_hat)
    )
    B = b * lin_hat + c * (ang @ lin.T + lin @ ang.T) + (ang.T @ lin) * W
    e = (b - 0.5 * a) / (1 - jnp.cos(theta))
    D = jnp.eye(3) - 0.5 * ang_hat + e * (ang_hat @ ang_hat)
    return jnp.block([[D, -D @ B @ D], [jnp.zeros_like(D), D]])


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
    g, w = jnp.split(s, (12,), axis=axis)
    return g, w
