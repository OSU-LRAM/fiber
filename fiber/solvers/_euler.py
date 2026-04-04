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

from typing import Callable, ClassVar

import jax.numpy as jnp
from diffrax import RESULTS, AbstractItoSolver, AbstractTerm
from jaxtyping import Array
from typing_extensions import TypeAlias

from .._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
from .._elements import Isometry, Twist
from .._interpolations._local_interpolation import LocalPartitionedGeodesicInterpolation
from .._operations import rplus
from .._utils import join_state, split_state

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


class LieEuler(AbstractItoSolver):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[
        Callable[..., LocalPartitionedGeodesicInterpolation]
    ] = LocalPartitionedGeodesicInterpolation

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Args,
    ) -> _SolverState:
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Array,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Array, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        g0, v0 = split_state(y0)

        vf = terms.vf_prod(t0, y0, args, terms.contr(t0, t1))
        g_tilde, v_tilde = jnp.split(vf, (Isometry.size,))

        g1 = rplus(g0, Twist.unflatten(g_tilde))
        v1 = v0 + v_tilde

        y1 = join_state(g1, v1)
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Array,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
