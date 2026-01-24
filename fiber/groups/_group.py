from abc import abstractmethod
from typing import Any, Sequence

from equinox import AbstractClassVar, AbstractVar, Module
from jaxtyping import Array

from .._custom_types import ArrayLike


class GroupElement(Module):
    coordinates: AbstractVar[Array]
    size: AbstractClassVar[int]
    shape: AbstractClassVar[Sequence[int]]

    @classmethod
    @abstractmethod
    def from_vector(cls, vector: ArrayLike) -> Any: ...

    @classmethod
    @abstractmethod
    def from_matrix(cls, matrix: ArrayLike) -> Any: ...

    @classmethod
    @abstractmethod
    def from_flat(cls, flat: ArrayLike) -> Any: ...

    @abstractmethod
    def as_matrix(self) -> Array: ...

    @abstractmethod
    def as_vector(self) -> Array: ...

    @abstractmethod
    def as_flat(self) -> Array: ...

    @classmethod
    @abstractmethod
    def eye(cls) -> Any: ...
