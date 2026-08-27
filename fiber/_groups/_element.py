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
from collections.abc import Sequence

import equinox as eqx
from equinox import AbstractVar
from jaxtyping import Array

from .._custom_types import ArrayLike, RealScalarLike


class AbstractGroupElement(eqx.Module):
    value: AbstractVar[Array]
    nq: AbstractVar[int]

    @property
    def shape(self):
        return self.value.shape

    @property
    def single(self) -> bool:
        return self.value.ndim == 2

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def as_matrix(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unravel(cls, params: Array):
        raise NotImplementedError

    @abstractmethod
    def ravel(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def eye(cls):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, elements: Sequence):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(cls, elements: Sequence):
        raise NotImplementedError

    def __len__(self) -> int:
        if self.single:
            raise TypeError("Single element has no len().")
        return self.value.shape[0]

    @abstractmethod
    def __getitem__(self, indexer):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class AbstractTangentVector[GroupElement: AbstractGroupElement](eqx.Module):
    point: AbstractVar[GroupElement]
    value: AbstractVar[Array]
    nv: AbstractVar[int]
    nx: AbstractVar[int]

    @property
    def shape(self):
        return self.value.shape

    @property
    def single(self) -> bool:
        return self.value.ndim == 2

    @classmethod
    @abstractmethod
    def from_matrix(cls, mat: ArrayLike, point: GroupElement | None = None):
        raise NotImplementedError

    @abstractmethod
    def as_matrix(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_vector(cls, vec: ArrayLike, point: GroupElement | None = None):
        raise NotImplementedError

    @abstractmethod
    def as_vector(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_coords(cls, coords: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def as_coords(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unravel(cls, params: ArrayLike, point: GroupElement | None = None):
        raise NotImplementedError

    @abstractmethod
    def ravel(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def zeros(cls, point: GroupElement | None = None):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, vectors: Sequence):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(cls, vectors: Sequence):
        raise NotImplementedError

    def __len__(self) -> int:
        if self.single:
            raise TypeError("Single vector has no len().")
        return self.value.shape[0]

    @abstractmethod
    def __getitem__(self, indexer):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class AbstractCotangentVector[GroupElement: AbstractGroupElement](eqx.Module):
    point: AbstractVar[GroupElement]
    value: AbstractVar[Array]
    nf: AbstractVar[int]
    nx: AbstractVar[int]

    @property
    def shape(self):
        return self.value.shape

    @property
    def single(self) -> bool:
        return self.value.ndim == 1

    @classmethod
    @abstractmethod
    def from_vector(cls, vec: ArrayLike, point: GroupElement | None = None):
        raise NotImplementedError

    @abstractmethod
    def as_vector(self) -> Array:
        raise NotImplementedError

    @abstractmethod
    def pair(self, tangent) -> RealScalarLike:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_coords(cls, coords: ArrayLike):
        raise NotImplementedError

    @abstractmethod
    def as_coords(self) -> Array:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def concatenate(cls, vectors: Sequence):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def stack(cls, vectors: Sequence):
        raise NotImplementedError

    def __len__(self) -> int:
        if self.single:
            raise TypeError("Single vector has no len().")
        return self.value.shape[0]

    @abstractmethod
    def __getitem__(self, indexer):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
