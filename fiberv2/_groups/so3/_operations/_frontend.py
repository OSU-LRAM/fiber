from jaxtyping import Array
from plum import dispatch

from .._element import Rotation3d, Spin3d
from . import _impl as impl


@dispatch
def inv(g: Rotation3d) -> Rotation3d:
    return Rotation3d.from_matrix(impl.inv(g.coordinates))


@dispatch
def inv(g: Array) -> Array:
    return impl.inv(g)


@dispatch
def adj(g: Spin3d, h: Spin3d) -> Spin3d:
    return Spin3d.from_matrix(impl.adj(g.coordinates, h.coordinates))


@dispatch
def adj(g: Array, h: Array) -> Array:
    return impl.adj(g, h)
