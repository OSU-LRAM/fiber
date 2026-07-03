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

from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
from jaxtyping import ScalarLike

from .._custom_types import Args
from .._groups._element import AbstractCotangentVector, AbstractTangentVector

_Vector = TypeVar("_Vector", bound=AbstractTangentVector)
_Covector = TypeVar("_Covector", bound=AbstractCotangentVector)


class AbstractImplicitVectorField(eqx.Module, Generic[_Vector, _Covector]):
    """Base class for a vector field described implicitly."""

    @abstractmethod
    def implicit_step(self, t: ScalarLike, y: _Vector, args: Args) -> _Covector:
        raise NotImplementedError
