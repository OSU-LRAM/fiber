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
from typing import ClassVar, cast

import jax.tree_util as jtu
import optimistix as optx
from diffrax import (
    RESULTS,
    AbstractImplicitSolver,
    AbstractStratonovichSolver,
    MultiTerm,
)
from diffrax._term import WrapTerm
from jaxtyping import Array
from optimistix import AbstractRootFinder

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike
from .._groups._element import AbstractCotangentVector, AbstractTangentVector
from .._local_interpolation import LocalLeftBundleInterpolation as LocalInterpolation
from ._term import ImplicitVariationalTerm, VariationalDiffusionTerm

type _ErrorEstimate = None
type _SolverState = None
type _V = AbstractTangentVector
type _CV = AbstractCotangentVector

_Terms = MultiTerm[
    tuple[
        ImplicitVariationalTerm[AbstractCotangentVector],
        VariationalDiffusionTerm[AbstractCotangentVector],
    ]
]


def _implicit_relation(v: Array, solver_args: Args) -> Array:
    implicit_step, y_prime, t1, vector_cls, integrator_args, dt = solver_args
    y = vector_cls.from_vector(v, point=y_prime.point)
    diff = implicit_step(t1, y, integrator_args, dt) - y_prime
    return diff.value


class LieSVI(AbstractImplicitSolver, AbstractStratonovichSolver):
    term_structure: ClassVar = _Terms
    interpolation_cls: ClassVar[Callable[..., LocalInterpolation]] = LocalInterpolation
    root_finder: AbstractRootFinder = optx.Chord(rtol=1e-3, atol=1e-4)  # type: ignore
    root_find_max_steps: int = 10  # type: ignore

    def order(self, terms):
        del terms
        return 1

    def error_order(self, terms):
        del terms
        return 2

    def init(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _V,
        args: Args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _V,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[_V, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        drift, diffusion = jtu.tree_map(lambda t: cast(WrapTerm, t), terms.terms)

        dt = drift.contr(t0, t1)
        dw = diffusion.contr(t0, t1)

        f0 = drift.vf_prod(t0, y0, args, dt)
        h0 = diffusion.vf_prod(t0, y0, args, dw)
        k0 = f0 + h0

        t = t1 * drift.direction
        implicit_step = drift.term.relation

        solver_args = (implicit_step, k0, t, type(y0), args, dt)
        nonlinear_sol = optx.root_find(
            _implicit_relation,
            self.root_finder,
            y0.as_vector(),
            solver_args,
            throw=False,
            max_steps=self.root_find_max_steps,
        )
        k1 = nonlinear_sol.value

        y1 = type(y0).from_vector(k1, point=k0.point)
        dense_info = {"y0": y0, "y1": y1}
        solver_state = None
        result = RESULTS.promote(nonlinear_sol.result)

        return y1, None, dense_info, solver_state, result

    def func(
        self,
        terms: _Terms,
        t0: RealScalarLike,
        y0: _V,
        args: Args,
    ) -> _CV:
        raise NotImplementedError
