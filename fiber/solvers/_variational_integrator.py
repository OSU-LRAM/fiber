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

from typing import Callable, ClassVar, Generic, TypeAlias, cast

import optimistix as optx
from diffrax import (
    RESULTS,
    AbstractImplicitSolver,
    AbstractStratonovichSolver,
    AbstractTerm,
    MultiTerm,
)
from diffrax._heuristics import is_sde
from diffrax._term import WrapTerm
from jaxtyping import Array
from optimistix import AbstractRootFinder

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike
from .._local_interpolation import LocalLeftBundleInterpolation as LocalInterpolation
from ._vector_field import _Covector, _Vector

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None


def _implicit_relation(v: Array, solver_args: Args) -> Array:
    implicit_step, y_prime, t1, vector_cls, integrator_args = solver_args
    y = vector_cls.from_vector(v, point=y_prime.point)
    diff = implicit_step(t1, y, integrator_args) - y_prime
    return diff.value


class VariationalIntegrator(
    AbstractImplicitSolver, AbstractStratonovichSolver, Generic[_Vector, _Covector]
):
    term_structure: ClassVar = MultiTerm
    interpolation_cls: ClassVar[Callable[..., LocalInterpolation]] = LocalInterpolation
    root_finder: AbstractRootFinder = optx.Chord(rtol=1e-2, atol=1e-2)  # type: ignore
    root_find_max_steps: int = 100  # type: ignore

    def order(self, terms):
        del terms
        return 1

    def error_order(self, terms):
        del terms
        return 2

    def init(
        self,
        terms: MultiTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _Vector,
        args: Args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _Vector,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[_Vector, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        if is_sde(terms):
            terms = cast(MultiTerm, terms)

            drift, diffusion = terms.terms
            dt = drift.contr(t0, t1)
            dw = diffusion.contr(t0, t1)

            f0 = drift.vf_prod(t0, y0, args, dt)
            h0 = diffusion.vf_prod(t0, y0, args, dw)
            y_prime = f0 + h0

            t = t1 * drift.direction
            implicit_step = drift.term.implicit_step
        else:
            terms = cast(WrapTerm, terms)
            y_prime = terms.vf_prod(t0, y0, args, terms.contr(t0, t1))

            t = t1 * terms.direction
            implicit_step = terms.term.implicit_step  # type: ignore

        solver_args = (implicit_step, y_prime, t, type(y0), args)
        nonlinear_sol = optx.root_find(
            _implicit_relation,
            self.root_finder,
            y0.as_vector(),
            solver_args,
            throw=False,
            max_steps=self.root_find_max_steps,
        )

        y1 = type(y0).from_vector(nonlinear_sol.value, point=y_prime.point)
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = RESULTS.promote(nonlinear_sol.result)

        return y1, None, dense_info, solver_state, result

    def func(
        self,
        terms: MultiTerm,
        t0: RealScalarLike,
        y0: _Vector,
        args: Args,
    ) -> _Covector:
        raise NotImplementedError
