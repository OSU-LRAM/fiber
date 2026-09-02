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
from typing import TypeVar, cast

from diffrax import AbstractTerm

from .._custom_types import Args, Control, RealScalarLike
from .._groups._element import AbstractCotangentVector, AbstractTangentVector

_CV = TypeVar("_CV", bound=AbstractCotangentVector)
type _V = AbstractTangentVector


class SharpTerm(AbstractTerm[_CV, Control]):
    vector_field: Callable[[RealScalarLike, _V, Args], _CV]
    dual_metric_fn: Callable[[_V, _CV], _V]

    def vf(self, t: RealScalarLike, y: _V, args: Args) -> _CV:
        return self.vector_field(t, y, args)

    def dual_metric(self, y0: _V, k1: _CV) -> _V:
        return self.dual_metric_fn(y0, k1)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return cast(Control, t1 - t0)

    def prod(self, vf: _CV, control: RealScalarLike) -> _CV:
        return vf * control  # type: ignore

    def vf_prod(self, t: RealScalarLike, y: _V, args: Args, control: Control) -> _CV:
        return self.prod(self.vf(t, y, args), control)


class ImplicitVariationalTerm(AbstractTerm[_CV, Control]):
    vector_field: Callable[[RealScalarLike, _V, Args, Control], _CV]
    implicit_f: Callable[[RealScalarLike, _V, Args, Control], _CV]

    def vf(self, t: RealScalarLike, y: _V, args: Args, control: Control = 0.0) -> _CV:
        return self.vector_field(t, y, args, control)

    def relation(self, t: RealScalarLike, y: _V, args: Args, control: Control) -> _CV:
        return self.implicit_f(t, y, args, control)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return cast(Control, t1 - t0)

    def prod(self, vf: _CV, control: RealScalarLike) -> _V:
        raise NotImplementedError

    def vf_prod(self, t: RealScalarLike, y: _V, args: Args, control: Control) -> _CV:
        return self.vf(t, y, args, control)


class VariationalControlTerm(AbstractTerm[_CV, Control]):
    vector_field: Callable[[RealScalarLike, _V, Args, Control], _CV]
    control: Control

    def vf(self, t: RealScalarLike, y: _V, args: Args, control: Control = 0.0) -> _CV:
        return self.vector_field(t, y, args, control)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs) -> Control:
        return self.control.evaluate(t0, t1, **kwargs)

    def prod(self, vf: _CV, control: RealScalarLike) -> _V:
        raise NotImplementedError

    def vf_prod(self, t: RealScalarLike, y: _V, args: Args, control: Control) -> _CV:
        return self.vf(t, y, args, control)
