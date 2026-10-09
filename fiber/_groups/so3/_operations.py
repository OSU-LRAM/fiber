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

from ..._coefficients import cosc, dlogc, sinc, sinc3
from ..._vecfuncs import skew3, vex3


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
    return adj_op(w).T


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
    return Adj_op(g).T


@functools.partial(jnp.vectorize, signature="(n,n),(n)->(n)")
def dAdj_inv(g: Array, p: Array) -> Array:
    return dAdj_inv_op(g) @ p


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dAdj_inv_op(g: Array) -> Array:
    return dAdj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def expm(w: Array) -> Array:
    theta2 = jnp.sum(vex3(w) ** 2)
    return jnp.eye(3) + sinc(theta2) * w + cosc(theta2) * (w @ w)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dexpm(w: Array) -> Array:
    theta2 = jnp.sum(vex3(w) ** 2)
    return jnp.eye(3) + cosc(theta2) * w + sinc3(theta2) * (w @ w)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    cos = jnp.clip(0.5 * (jnp.trace(g) - 1), -1.0, 1.0)
    sin_axis = 0.5 * vex3(g - g.T)
    sin2 = jnp.sum(sin_axis**2)

    # guard the square root so its gradient stays finite at the identity
    sin = jnp.where(sin2 > 0.0, jnp.sqrt(jnp.where(sin2 > 0.0, sin2, 1.0)), 0.0)
    theta = jnp.arctan2(sin, cos)

    # past pi / 2 the skew part loses the axis, so recover it from the symmetric part
    obtuse = cos < 0.0
    outer = (0.5 * (g + g.T) - cos * jnp.eye(3)) / jnp.where(obtuse, 1 - cos, 1.0)
    i = jnp.argmax(jnp.diag(outer))
    axis = outer[:, i] / jnp.sqrt(jnp.where(obtuse, outer[i, i], 1.0))
    axis = jnp.where(axis @ sin_axis < 0.0, -axis, axis)

    w = jnp.where(
        obtuse, theta * axis, sin_axis / sinc(jnp.where(obtuse, 0.0, theta**2))
    )
    return skew3(w)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def dlogm(w: Array) -> Array:
    theta2 = jnp.sum(vex3(w) ** 2)
    return jnp.eye(3) - 0.5 * w + dlogc(theta2) * (w @ w)


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
