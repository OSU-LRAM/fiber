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

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._linalg import skew3, vex3
from ..so3._operations import _left_jac_inv as _left_jac_inv_so3
from ..so3._operations import _logm as _logm_so3
from ._group import SE3, se3


def inv(A):
    return eqx.tree_at(lambda _A: _A.coordinates, A, jnp.linalg.inv(A.coordinates))


def expm(A: se3) -> SE3: ...


def logm(A: SE3) -> se3:
    return se3.from_matrix(_logm(A.coordinates))


def adj(g: se3, h: se3) -> se3:
    return g @ h - h @ g


def adj_op(g: se3) -> Array:
    v, w = g.linear, g.angular
    return jnp.block([[skew3(w), skew3(v)], [jnp.zeros((3, 3)), skew3(w)]])


def Adj(g: SE3, h: se3) -> se3:
    return g @ h @ inv(g)


def Adji(g: SE3, h: se3) -> se3:
    return inv(g) @ h @ g


def dadj(g: se3, p: Array) -> Array:
    return adj_op(g) @ p


def dadj_op(g: se3) -> Array:
    return adj_op(g).T


def dAdj(): ...


def dAdj_op(): ...


def dadji(): ...


def dadji_op(): ...


def dAdji(): ...


def dAdji_op(): ...


def rplus(): ...


def rminus(): ...


def dexp(g: se3, h: se3, order: int = 3): ...


def dlog(order: int = 3): ...


def _logm(g: Array) -> Array:
    t, rot = g[:3, 3], g[:3, :3]
    w_hat = _logm_so3(rot)
    return jnp.block([[w_hat, _left_jac_inv_so3(vex3(w_hat)) @ t], [jnp.zeros(4)]])
