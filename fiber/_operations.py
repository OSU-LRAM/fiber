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

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._elements import Isometry, Twist
from .linalg import skew3, softnorm, vex3


@singledispatch
def inv(g):
    return _inv(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _inv(g: Array) -> Array:
    return jnp.linalg.inv(g)


@inv.register
def _inv_type(g: Isometry) -> Isometry:
    return Isometry.from_matrix(_inv(g.coordinates))


@singledispatch
def adj(g, h):
    return _adj(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _adj(g: Array, h: Array) -> Array:
    return g @ h - h @ g


@adj.register
def _adj_type(g: Twist, h: Twist) -> Twist:
    return Twist.from_matrix(_adj(g.coordinates, h.coordinates))


@singledispatch
def adj_op(g):
    return _adj_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _adj_op(g: Array) -> Array:
    v, w = g[:3, 3], g[:3, :3]
    return jnp.block([[w, skew3(v)], [jnp.zeros((3, 3)), w]])


@adj_op.register
def _adj_op_type(g: Twist) -> Array:
    return _adj_op(g.coordinates)


@singledispatch
def dadj(g, p):
    return _dadj(g, p)


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def _dadj(g: Array, p: Array) -> Array:
    return _dadj_op(g) @ p


@dadj.register
def _dadj_type(g: Twist, p: Array) -> Array:
    return _dadj(g.coordinates, p)


@singledispatch
def dadj_op(g):
    return _dadj_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dadj_op(g: Array) -> Array:
    return -_adj_op(g).T


@dadj_op.register
def _dadj_op_type(g: Twist) -> Array:
    return _dadj_op(g.coordinates)


@singledispatch
def dadj_inv(g, p):
    return _dadj_inv(g, p)


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def _dadj_inv(g: Array, p: Array) -> Array:
    return -_dadj(g, p)


@dadj_inv.register
def _dadj_inv_type(g: Twist, p: Array) -> Array:
    return _dadj_inv(g.coordinates, p)


@singledispatch
def dadj_inv_op(g):
    return _dadj_inv_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dadj_inv_op(g: Array) -> Array:
    return -_adj_op(g).T


@dadj_inv_op.register
def _dadj_inv_op_type(g: Twist) -> Array:
    return _dadj_inv_op(g.coordinates)


@singledispatch
def Adj(g, h):
    return _Adj(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _Adj(g: Array, h: Array) -> Array:
    return g @ h @ jnp.linalg.inv(g)


@Adj.register
def _Adj_type(g: Isometry, h: Twist) -> Twist:
    return Twist.from_matrix(_Adj(g.coordinates, h.coordinates))


@singledispatch
def Adj_op(g):
    return _Adj_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _Adj_op(g: Array) -> Array:
    p, rot = g[:3, 3], g[:3, :3]
    return jnp.block([[rot, skew3(p) @ rot], [jnp.zeros_like(rot), rot]])


@Adj_op.register
def _Adj_op_type(g: Isometry) -> Array:
    return _Adj_op(g.coordinates)


@singledispatch
def Adj_inv(g, h):
    return _Adj_inv(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _Adj_inv(g: Array, h: Array) -> Array:
    return jnp.linalg.inv(g) @ h @ g


@Adj_inv.register
def _Adj_inv_type(g: Isometry, h: Twist) -> Twist:
    return Twist.from_matrix(_Adj_inv(g.coordinates, h.coordinates))


@singledispatch
def Adj_inv_op(g):
    return _Adj_inv_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _Adj_inv_op(g: Array) -> Array:
    return _Adj_op(jnp.linalg.inv(g))


@Adj_inv_op.register
def _Adj_inv_op_type(g: Isometry) -> Array:
    return _Adj_inv_op(g.coordinates)


@singledispatch
def dAdj(g, p):
    return _dAdj(g, p)


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def _dAdj(g: Array, p: Array) -> Array:
    return _dAdj_op(g) @ p


@dAdj.register
def _dAdj_type(g: Isometry, p: Array) -> Array:
    return _dAdj(g.coordinates, p)


@singledispatch
def dAdj_op(g):
    return _dAdj_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dAdj_op(g: Array) -> Array:
    return _Adj_op(jnp.linalg.inv(g)).T


@dAdj_op.register
def _dAdj_op_type(g: Isometry) -> Array:
    return _dAdj_op(g.coordinates)


@singledispatch
def dAdj_inv(g, p):
    return _dAdj_inv(g, p)


@functools.partial(jnp.vectorize, signature="(n,n),(m)->(m)")
def _dAdj_inv(g: Array, p: Array) -> Array:
    return _dAdj_inv_op(g) @ p


@dAdj_inv.register
def _dAdj_inv_type(g: Isometry, p: Array) -> Array:
    return _dAdj_inv(g.coordinates, p)


@singledispatch
def dAdj_inv_op(g):
    return _dAdj_inv_op(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dAdj_inv_op(g: Array) -> Array:
    return _dAdj_op(jnp.linalg.inv(g))


@dAdj_inv_op.register
def _dAdj_inv_op_type(g: Isometry) -> Array:
    return _dAdj_inv_op(g.coordinates)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _so3_expm(w: Array) -> Array:
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
def _so3_dexpm(w: Array) -> Array:
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
    return a * jnp.eye(3) + b * w + c * (w @ w)


@singledispatch
def expm(g):
    return _expm(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _expm(g: Array) -> Array:
    p, w_hat = g[:3, 3], g[:3, :3]
    theta = softnorm(vex3(w_hat))
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
    V = jnp.eye(3) + B * w_hat + C * (w_hat @ w_hat)
    return jnp.block([[_so3_expm(w_hat), (V @ p).reshape(3, 1)], [jnp.zeros(3), 1]])


@expm.register
def _expm_type(g: Twist) -> Isometry:
    return Isometry.from_matrix(_expm(g.coordinates))


@singledispatch
def dexpm(g):
    return _dexpm(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dexpm(g: Array) -> Array:
    v, w_hat = g[:3, 3], g[:3, :3]
    v_hat, w = skew3(v), vex3(w_hat)
    theta = softnorm(w)
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
        + ((a - 2 * b) / theta**2) * w_hat
        + ((b - 3 * c) / theta**2) * (w_hat @ w_hat)
    )
    V = b * v_hat + c * (w @ v.T + v @ w.T) + (w.T @ v * W)
    D = a * jnp.eye(3) + b * w_hat + c * (w_hat @ w_hat)
    return jnp.block([[D, V], [jnp.zeros_like(V), D]])


@dexpm.register
def _dexpm_type(g: Twist) -> Array:
    return _dexpm(g.coordinates)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _so3_logm(w: Array) -> Array:
    cos = (jnp.trace(w) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)
    w_hat = jnp.zeros((3, 3))
    return jax.lax.cond(
        jnp.sin(theta) < 1e-6,
        lambda: w_hat,
        lambda: 0.5 * theta / jnp.sin(theta) * (w - w.T),
    )


@singledispatch
def logm(g):
    return _logm(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _logm(g: Array) -> Array:
    p, rot = g[:3, 3], g[:3, :3]
    w_hat = _so3_logm(rot)
    w = vex3(w_hat)
    jac = jnp.eye(3)
    V = jax.lax.cond(
        jnp.isclose(softnorm(w), 0.0),
        lambda: jac,
        lambda: jac
        - 0.5 * w_hat
        + (
            (1 / (softnorm(w) ** 2))
            - (1 + jnp.cos(softnorm(w))) / (2 * softnorm(w) * jnp.sin(softnorm(w)))
        )
        * (w_hat @ w_hat),
    )
    return jnp.block([[w_hat, (V @ p).reshape(3, 1)], [jnp.zeros(4)]])


@logm.register
def _logm_type(g: Isometry) -> Twist:
    return Twist.from_matrix(_logm(g.coordinates))


@functools.partial(jnp.vectorize, signature="(n,n)->(n,n)")
def _so3_dlogm(w: Array) -> Array:
    cos = (jnp.trace(w) - 1) / 2
    cos = jnp.clip(cos, -1, 1)
    theta = jnp.arccos(cos)
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
    e = (b - 0.5 * a) / (1 - cos)
    return jnp.eye(3) - 0.5 * w + e * (w @ w)


@singledispatch
def dlogm(g):
    return _dlogm(g)


@functools.partial(jnp.vectorize, signature="(n,n)->(m,m)")
def _dlogm(A: Array) -> Array:
    v, w_hat = A[:3, 3], A[:3, :3]
    v_hat, w = skew3(v), vex3(w_hat)
    theta = softnorm(w)
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
        + ((a - 2 * b) / theta**2) * w_hat
        + ((b - 3 * c) / theta**2) * (w_hat @ w_hat)
    )
    B = b * v_hat + c * (w @ v.T + v @ w.T) + (w.T @ v) * W
    e = (b - 0.5 * a) / (1 - jnp.cos(theta))
    D = jnp.eye(3) - 0.5 * w_hat + e * (w_hat @ w_hat)
    return jnp.block([[D, -D @ B @ D], [jnp.zeros_like(D), D]])


@dlogm.register
def _dlogm_type(g: Twist) -> Array:
    return _dlogm(g.coordinates)


@singledispatch
def lplus(g, h):
    return _lplus(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _lplus(g: Array, h: Array) -> Array:
    return _expm(h) @ g


@lplus.register
def _lplus_type(g: Isometry, h: Twist) -> Isometry:
    return Isometry.from_matrix(_lplus(g.coordinates, h.coordinates))


@singledispatch
def rplus(g, h):
    return _rplus(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _rplus(g: Array, h: Array) -> Array:
    return g @ _expm(h)


@rplus.register
def _rplus_type(g: Isometry, h: Twist) -> Isometry:
    return Isometry.from_matrix(_rplus(g.coordinates, h.coordinates))


@singledispatch
def lminus(g, h):
    return _lminus(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _lminus(g: Array, h: Array) -> Array:
    return _logm(g @ jnp.linalg.inv(h))


@lminus.register
def _lminus_type(g: Isometry, h: Isometry) -> Twist:
    return Twist.from_matrix(_lminus(g.coordinates, h.coordinates))


@singledispatch
def rminus(g, h):
    return _rminus(g, h)


@functools.partial(jnp.vectorize, signature="(n,n),(n,n)->(n,n)")
def _rminus(g: Array, h: Array) -> Array:
    return _logm(jnp.linalg.inv(h) @ g)


@rminus.register
def _rminus_type(g: Isometry, h: Isometry) -> Twist:
    return Twist.from_matrix(_rminus(g.coordinates, h.coordinates))
