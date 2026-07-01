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

from ..._custom_types import RealScalarLike
from ..._vecfuncs import skew2, vex2


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def inv(w: Array) -> Array:
    return w.T


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def adj(w: Array, v: Array) -> Array:
    return w @ v - v @ w


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def adj_op(w: Array) -> RealScalarLike:
    del w
    return jnp.zeros(())


@functools.partial(jnp.vectorize, signature="(n,n),()->()")
def dadj(w: Array, p: RealScalarLike) -> RealScalarLike:
    return dadj_op(w) * p


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dadj_op(w: Array) -> RealScalarLike:
    return -adj_op(w)


@functools.partial(jnp.vectorize, signature="(n,n),()->()")
def dadj_inv(w: Array, p: RealScalarLike) -> RealScalarLike:
    return -dadj(w, p)


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dadj_inv_op(w: Array) -> RealScalarLike:
    return -dadj_op(w)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj(g: Array, w: Array) -> Array:
    return g @ w @ inv(g)


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def Adj_op(g: Array) -> RealScalarLike:
    del g
    return jnp.ones(())


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def Adj_inv(g: Array, w: Array) -> Array:
    return inv(g) @ w @ g


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def Adj_inv_op(g: Array) -> RealScalarLike:
    return Adj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n),()->()")
def dAdj(g: Array, p: RealScalarLike) -> RealScalarLike:
    return dAdj_op(g) * p


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dAdj_op(g: Array) -> RealScalarLike:
    return Adj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n),()->()")
def dAdj_inv(g: Array, p: RealScalarLike) -> RealScalarLike:
    return dAdj_inv_op(g) * p


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dAdj_inv_op(g: Array) -> RealScalarLike:
    return dAdj_op(inv(g))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def expm(w: Array) -> Array:
    theta = vex2(w)
    cos, sin = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[cos, -sin], [sin, cos]])


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dexpm(w: Array) -> RealScalarLike:
    del w
    return jnp.ones(())


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def logm(g: Array) -> Array:
    theta = jnp.arctan2(g[1, 0], g[0, 0])
    return skew2(theta)


@functools.partial(jnp.vectorize, signature="(n,n)->()")
def dlogm(w: Array) -> RealScalarLike:
    del w
    return jnp.ones(())


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
