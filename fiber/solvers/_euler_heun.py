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

from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
from diffrax import RESULTS, AbstractStratonovichSolver, AbstractTerm, MultiTerm
from jaxtyping import Array

from .._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
from .._groups import AbstractCotangentVector, AbstractTangentVector
from .._local_interpolation import LocalLeftBundleInterpolation as LocalInterpolation
from .._operations import rplus
from ._term import SharpTerm

type _ErrorEstimate = None
type _SolverState = None
type _Terms = MultiTerm[tuple[SharpTerm, AbstractTerm]]


class EulerHeun[VectorT: AbstractTangentVector, CovectorT: AbstractCotangentVector](
    AbstractStratonovichSolver
):
    term_structure: ClassVar = _Terms
    interpolation_cls: ClassVar[Callable[..., LocalInterpolation]] = LocalInterpolation

    def order(self, terms):
        del terms
        return 1

    def strong_order(self, terms):
        del terms
        return 0.5

    def init(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: VectorT,
        args: Args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: VectorT,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[VectorT, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        drift, diffusion = terms.terms
        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)

        f0 = drift.vf_prod(t0, y0, args, dt)
        h0 = diffusion.prod(diffusion.vf(t0, y0, args), dw)
        h0 = drift.term.cometric(y0, h0)  # type: ignore

        y_prime = y0 + h0
        h_prime = diffusion.vf_prod(t0, y_prime, args, dw)
        h_prime = drift.term.cometric(y_prime, h_prime)  # type: ignore

        vf = f0 + 0.5 * (h0 + h_prime)
        y1 = y0 + vf
        y1 = eqx.tree_at(lambda w: w.point.value, y1, rplus(y0.point, f0).value)

        dense_info = {"y0": y0, "y1": y1}
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        y0: Array,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
