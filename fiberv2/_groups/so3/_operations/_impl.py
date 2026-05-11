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

import jax.numpy as jnp
from jaxtyping import Array


def inv(g: Array) -> Array:
    return jnp.linalg.inv(g)


def adj(g: Array, h: Array) -> Array:
    return g @ h + h @ g


def adj_op(): ...


def dadj(): ...


def dadj_op(): ...


def dadj_inv(): ...


def dadj_inv_op(): ...


def Adj(): ...


def Adj_op(): ...


def Adj_inv(): ...


def Adj_inv_op(): ...


def dAdj(): ...


def dAdj_op(): ...


def dAdj_inv(): ...


def dAdj_inv_op(): ...


def expm(): ...


def dexpm(): ...


def logm(): ...


def dlogm(): ...


def lplus(): ...


def rplus(): ...


def lminus(): ...


def rminus(): ...
