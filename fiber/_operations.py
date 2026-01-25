from jaxtyping import Array

from ._core.elements import Isometry, Twist


def adj(g: Twist, h: Twist) -> Twist:
    return Twist.from_matrix(_adj(g.coordinates, h.coordinates))


def adj_op(g: Twist) -> Array:
    return _adj_op(g.linear, g.angular)


def Adj(g: Isometry, h: Twist) -> Twist:
    return Twist.from_matrix(_Adj(g.coordinates, h.coordinates))


def Adji(g: Isometry, h: Twist) -> Twist:
    return Twist.from_matrix(_Adji(g.coordinates, h.coordinates))


def dadj(g: Twist, p: Array) -> Array:
    return _dadj(g.coordinates, p)


def dadj_op(g: Twist) -> Array:
    return _dadj_op(g.coordinates)


def dadji(g: Twist, p: Array) -> Array:
    return _dadji(g.coordinates, p)


def dadji_op(g: Twist) -> Array:
    return _dadji_op(g.coordinates)
