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
from typing import Generic, Optional, TypeVar

import equinox as eqx
from equinox import AbstractVar
from jaxtyping import Array

from .._custom_types import ArrayLike


class AbstractGroupElement(eqx.Module):
    value: AbstractVar[Array]

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def as_matrix(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def eye(cls):
        raise NotImplementedError

    @property
    def single(self) -> bool:
        return self.value.ndim == 1


_G = TypeVar("_G", bound=AbstractGroupElement)


class AbstractTangentVector(eqx.Module, Generic[_G]):
    point: AbstractVar[_G]
    value: AbstractVar[Array]

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat: Array, point: Optional[_G] = None):
        raise NotImplementedError

    @abstractmethod
    def as_matrix(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_vector(cls, vec: Array, point: Optional[_G] = None):
        raise NotImplementedError

    @abstractmethod
    def as_vector(self) -> Array:
        raise NotImplementedError

    @property
    def single(self) -> bool:
        return self.value.ndim == 1
