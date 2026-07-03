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

from typing import Callable, ClassVar, Generic, TypeVar

import equinox as eqx
from diffrax import RESULTS, AbstractItoSolver, AbstractTerm
from typing_extensions import TypeAlias

from .._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike
from .._groups._element import AbstractTangentVector
from .._local_interpolation import LocalLeftBundleInterpolation as LocalInterpolation
from .._operations import rplus

_ErrorEstimate: TypeAlias = None
_SolverState: TypeAlias = None

_TangentVector = TypeVar("_TangentVector", bound=AbstractTangentVector)


class LieEuler(AbstractItoSolver, Generic[_TangentVector]):
    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalInterpolation]] = LocalInterpolation

    def order(self, terms):
        del terms
        return 1

    def strong_order(self, terms):
        del terms
        return 0.5

    def init(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _TangentVector,
        args: Args,
    ) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: _TangentVector,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[_TangentVector, _ErrorEstimate, DenseInfo, _SolverState, RESULTS]:
        del solver_state, made_jump

        vf = terms.vf_prod(t0, y0, args, terms.contr(t0, t1))
        y1 = y0 + vf
        y1 = eqx.tree_at(lambda w: w.point.value, y1, rplus(y0.point, vf.point).value)

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: _TangentVector,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)
