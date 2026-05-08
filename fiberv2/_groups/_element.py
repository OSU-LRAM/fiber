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
from equinox import AbstractClassVar, AbstractVar
from jaxtyping import Array

from .._custom_types import ArrayLike


class Element(eqx.Module):
    """Base class for an element (either a group element or tangent vector)."""

    coordinates: AbstractVar[Array]
    dim: AbstractClassVar[int]
    size: AbstractClassVar[int]

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: ArrayLike) -> Element: ...

    @abstractmethod
    def as_matrix(self) -> Array: ...

    @abstractmethod
    def flatten(self) -> Array: ...

    @classmethod
    @abstractmethod
    def unflatten(cls, flat: ArrayLike): ...

    @classmethod
    @abstractmethod
    def eye(cls) -> Element: ...

    @classmethod
    @abstractmethod
    def pack(cls, elements: Sequence[Element]) -> Element: ...

    @abstractmethod
    def unpack(self) -> Sequence[Element]: ...

    @property
    def single(self) -> bool:
        """Whether this instance represents a single transformation."""
        return self.coordinates.ndim == 1

    def __len__(self) -> int:
        """The total number of transformations contained in this instance."""
        if self.single:
            raise ValueError("Single element has no len().")
        return self.coordinates.shape[0]
