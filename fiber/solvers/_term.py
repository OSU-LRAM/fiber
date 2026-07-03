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

from typing import Generic, override

import equinox as eqx
import jax.tree_util as jtu
from diffrax import AbstractTerm
from diffrax import ControlTerm as _ControlTerm
from diffrax._misc import upcast_or_raise

from .._custom_types import VF, Args, Control, RealScalarLike
from ._vector_field import AbstractImplicitVectorField, _Covector, _Vector


class ImplicitTerm(AbstractTerm, Generic[_Vector, _Covector]):
    vector_field: AbstractImplicitVectorField[_Vector, _Covector]

    def _vf(self, fn: VF, t: RealScalarLike, y: _Vector, args: Args):
        out = fn(t, y, args)

        def _upcast(oi, yi):
            oi = upcast_or_raise(
                oi,
                yi,
                "the vector field passed to `ImplicitTerm`",
                "the corresponding leaf of `y`",
            )
            return oi

        point = jtu.tree_map(_upcast, out.point, y.point)
        value = _upcast(out.value, y.value)

        return eqx.tree_at(lambda w: (w.point, w.value), out, (point, value))

    def vf(self, t: RealScalarLike, y: _Vector, args: Args) -> _Covector:
        return self._vf(self.vector_field, t, y, args)

    def implicit_step(self, t: RealScalarLike, y: _Vector, args: Args) -> _Covector:
        return self._vf(self.vector_field.implicit_step, t, y, args)

    def contr(self, t0: RealScalarLike, t1: RealScalarLike, **kwargs):
        return t1 - t0

    def prod(self, vf: VF, control: Control):
        raise NotImplementedError

    def vf_prod(
        self, t: RealScalarLike, y: _Vector, args: Args, control: Control
    ) -> _Covector:
        return self.vf(t, y, args)


class UnsafeControlTerm(_ControlTerm, Generic[_Vector]):
    @override
    def vf_prod(self, t: RealScalarLike, y: _Vector, args: Args, control):
        vf = self.vf(t, y, args)
        return self.prod(vf, control)
