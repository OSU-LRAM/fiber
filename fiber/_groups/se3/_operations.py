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

from ..._coefficients import cosc, series_or_closed, sinc, sinc3
from ..._vecfuncs import skew3, vex3
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
    return adj_op(w).T


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
    return jnp.block([[rot, skew3(pos) @ rot], [jnp.zeros_like(rot), rot]])


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
    return Adj_op(g).T


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def dAdj_inv(g: Array, p: Array) -> Array:
    return dAdj_inv_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dAdj_inv_op(g: Array) -> Array:
    return dAdj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def expm(w: Array) -> Array:
    lin, ang = w[:3, 3], w[:3, :3]
    V = so3.dexpm(ang)
    return jnp.block([[so3.expm(ang), (V @ lin).reshape(3, 1)], [jnp.zeros(3), 1]])


def _se3_V(lin: Array, ang_hat: Array) -> Array:
    lin_hat = skew3(lin)
    ang = vex3(ang_hat)
    theta2 = jnp.sum(ang**2)
    q1 = series_or_closed(
        theta2,
        lambda x: -1 / 12 + x / 180 - x**2 / 6720 + x**3 / 453600 - x**4 / 47900160,
        lambda x: (sinc(x) - 2 * cosc(x)) / x,
    )
    q2 = series_or_closed(
        theta2,
        lambda x: -1 / 60 + x / 1260 - x**2 / 60480 + x**3 / 4989600 - x**4 / 622702080,
        lambda x: (cosc(x) - 3 * sinc3(x)) / x,
    )
    Q = q1 * ang_hat + q2 * (ang_hat @ ang_hat)
    return (
        cosc(theta2) * lin_hat
        + sinc3(theta2) * (ang_hat @ lin_hat + lin_hat @ ang_hat)
        + jnp.dot(ang, lin) * Q
    )


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dexpm(w: Array) -> Array:
    lin, ang_hat = w[:3, 3], w[:3, :3]
    V = _se3_V(lin, ang_hat)
    D = so3.dexpm(ang_hat)
    return jnp.block([[D, V], [jnp.zeros_like(V), D]])


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    pos, rot = g[:3, 3], g[:3, :3]
    ang_hat = so3.logm(rot)
    V = so3.dlogm(ang_hat)
    return jnp.block([[ang_hat, (V @ pos).reshape(3, 1)], [jnp.zeros(4)]])


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def dlogm(w: Array) -> Array:
    lin, ang_hat = w[:3, 3], w[:3, :3]
    B = _se3_V(lin, ang_hat)
    D = so3.dlogm(ang_hat)
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
