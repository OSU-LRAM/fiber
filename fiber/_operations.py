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

from jaxtyping import Array
from plum import dispatch

from ._groups import se2, se3, so2, so3
from ._groups.se2 import Isometry2d, Twist2d, Wrench2d
from ._groups.se3 import Isometry3d, Twist3d, Wrench3d
from ._groups.so2 import Moment2d, Rotation2d, Spin2d
from ._groups.so3 import Moment3d, Rotation3d, Spin3d


@dispatch
def inv(w: Rotation2d) -> Rotation2d:  # type: ignore[reportRedeclaration]
    return Rotation2d.from_matrix(so2.inv(w.value))


@dispatch
def inv(w: Rotation3d) -> Rotation3d:
    return Rotation3d.from_matrix(so3.inv(w.value))


@dispatch
def inv(w: Isometry2d) -> Isometry2d:
    return Isometry2d.from_matrix(se2.inv(w.value))


@dispatch
def inv(w: Isometry3d) -> Isometry3d:
    return Isometry3d.from_matrix(se3.inv(w.value))


@dispatch
def adj(w: Spin2d, v: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.adj(w.value, v.value))


@dispatch
def adj(w: Spin3d, v: Spin3d) -> Spin3d:
    return Spin3d.from_matrix(so3.adj(w.value, v.value))


@dispatch
def adj(w: Twist2d, v: Twist2d) -> Twist2d:
    return Twist2d.from_matrix(se2.adj(w.value, v.value))


@dispatch
def adj(w: Twist3d, v: Twist3d) -> Twist3d:
    return Twist3d.from_matrix(se3.adj(w.value, v.value))


@dispatch
def adj_op(w: Spin2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.adj_op(w.value)


@dispatch
def adj_op(w: Spin3d) -> Array:
    return so3.adj_op(w.value)


@dispatch
def adj_op(w: Twist2d) -> Array:
    return se2.adj_op(w.value)


@dispatch
def adj_op(w: Twist3d) -> Array:
    return se3.adj_op(w.value)


@dispatch
def dadj(w: Spin2d, p: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
    return Moment2d.from_vector(so2.dadj(w.value, p.value))


@dispatch
def dadj(w: Spin3d, p: Moment3d) -> Moment3d:
    return Moment3d.from_vector(so3.dadj(w.value, p.value))


@dispatch
def dadj(w: Twist2d, p: Wrench2d) -> Wrench2d:
    return Wrench2d.from_vector(se2.dadj(w.value, p.value))


@dispatch
def dadj(w: Twist3d, p: Wrench3d) -> Wrench3d:
    return Wrench3d.from_vector(se3.dadj(w.value, p.value))


@dispatch
def dadj_op(w: Spin2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dadj_op(w.value)


@dispatch
def dadj_op(w: Spin3d) -> Array:
    return so3.dadj_op(w.value)


@dispatch
def dadj_op(w: Twist2d) -> Array:
    return se2.dadj_op(w.value)


@dispatch
def dadj_op(w: Twist3d) -> Array:
    return se3.dadj_op(w.value)


@dispatch
def dadj_inv(w: Spin2d, p: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
    return Moment2d.from_vector(so2.dadj_inv(w.value, p.value))


@dispatch
def dadj_inv(w: Spin3d, p: Moment3d) -> Moment3d:
    return Moment3d.from_vector(so3.dadj_inv(w.value, p.value))


@dispatch
def dadj_inv(w: Twist2d, p: Wrench2d) -> Wrench2d:
    return Wrench2d.from_vector(se2.dadj_inv(w.value, p.value))


@dispatch
def dadj_inv(w: Twist3d, p: Wrench3d) -> Wrench3d:
    return Wrench3d.from_vector(se3.dadj_inv(w.value, p.value))


@dispatch
def dadj_inv_op(w: Spin2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dadj_inv_op(w.value)


@dispatch
def dadj_inv_op(w: Spin3d) -> Array:
    return so3.dadj_inv_op(w.value)


@dispatch
def dadj_inv_op(w: Twist2d) -> Array:
    return se2.dadj_inv_op(w.value)


@dispatch
def dadj_inv_op(w: Twist3d) -> Array:
    return se3.dadj_inv_op(w.value)


@dispatch
def Adj(g: Rotation2d, w: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.Adj(g.value, w.value))


@dispatch
def Adj(g: Rotation3d, w: Spin3d) -> Spin3d:
    return Spin3d.from_matrix(so3.Adj(g.value, w.value))


@dispatch
def Adj(g: Isometry2d, w: Twist2d) -> Twist2d:
    return Twist2d.from_matrix(se2.Adj(g.value, w.value))


@dispatch
def Adj(g: Isometry3d, w: Twist3d) -> Twist3d:
    return Twist3d.from_matrix(se3.Adj(g.value, w.value))


@dispatch
def Adj_op(g: Rotation2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.Adj_op(g.value)


@dispatch
def Adj_op(g: Rotation3d) -> Array:
    return so3.Adj_op(g.value)


@dispatch
def Adj_op(g: Isometry2d) -> Array:
    return se2.Adj_op(g.value)


@dispatch
def Adj_op(g: Isometry3d) -> Array:
    return se3.Adj_op(g.value)


@dispatch
def Adj_inv(g: Rotation2d, w: Spin2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.Adj_inv(g.value, w.value))


@dispatch
def Adj_inv(g: Rotation3d, w: Spin3d) -> Spin3d:
    return Spin3d.from_matrix(so3.Adj_inv(g.value, w.value))


@dispatch
def Adj_inv(g: Isometry2d, w: Twist2d) -> Twist2d:
    return Twist2d.from_matrix(se2.Adj_inv(g.value, w.value))


@dispatch
def Adj_inv(g: Isometry3d, w: Twist3d) -> Twist3d:
    return Twist3d.from_matrix(se3.Adj_inv(g.value, w.value))


@dispatch
def Adj_inv_op(g: Rotation2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.Adj_inv_op(g.value)


@dispatch
def Adj_inv_op(g: Rotation3d) -> Array:
    return so3.Adj_inv_op(g.value)


@dispatch
def Adj_inv_op(g: Isometry2d) -> Array:
    return se2.Adj_inv_op(g.value)


@dispatch
def Adj_inv_op(g: Isometry3d) -> Array:
    return se3.Adj_inv_op(g.value)


@dispatch
def dAdj(g: Rotation2d, p: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
    return Moment2d.from_vector(so2.dAdj(g.value, p.value))


@dispatch
def dAdj(g: Rotation3d, p: Moment3d) -> Moment3d:
    return Moment3d.from_vector(so3.dAdj(g.value, p.value))


@dispatch
def dAdj(g: Isometry2d, p: Wrench2d) -> Wrench2d:
    return Wrench2d.from_vector(se2.dAdj(g.value, p.value))


@dispatch
def dAdj(g: Isometry3d, p: Wrench3d) -> Wrench3d:
    return Wrench3d.from_vector(se3.dAdj(g.value, p.value))


@dispatch
def dAdj_op(g: Rotation2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dAdj_op(g.value)


@dispatch
def dAdj_op(g: Rotation3d) -> Array:
    return so3.dAdj_op(g.value)


@dispatch
def dAdj_op(g: Isometry2d) -> Array:
    return se2.dAdj_op(g.value)


@dispatch
def dAdj_op(g: Isometry3d) -> Array:
    return se3.dAdj_op(g.value)


@dispatch
def dAdj_inv(g: Rotation2d, p: Moment2d) -> Moment2d:  # type: ignore[reportRedeclaration]
    return Moment2d.from_vector(so2.dAdj_inv(g.value, p.value))


@dispatch
def dAdj_inv(g: Rotation3d, p: Moment3d) -> Moment3d:
    return Moment3d.from_vector(so3.dAdj_inv(g.value, p.value))


@dispatch
def dAdj_inv(g: Isometry2d, p: Wrench2d) -> Wrench2d:
    return Wrench2d.from_vector(se2.dAdj_inv(g.value, p.value))


@dispatch
def dAdj_inv(g: Isometry3d, p: Wrench3d) -> Wrench3d:
    return Wrench3d.from_vector(se3.dAdj_inv(g.value, p.value))


@dispatch
def dAdj_inv_op(g: Rotation2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dAdj_inv_op(g.value)


@dispatch
def dAdj_inv_op(g: Rotation3d) -> Array:
    return so3.dAdj_inv_op(g.value)


@dispatch
def dAdj_inv_op(g: Isometry2d) -> Array:
    return se2.dAdj_inv_op(g.value)


@dispatch
def dAdj_inv_op(g: Isometry3d) -> Array:
    return se3.dAdj_inv_op(g.value)


@dispatch
def expm(w: Spin2d) -> Rotation2d:  # type: ignore[reportRedeclaration]
    return Rotation2d.from_matrix(so2.expm(w.value))


@dispatch
def expm(w: Spin3d) -> Rotation3d:
    return Rotation3d.from_matrix(so3.expm(w.value))


@dispatch
def expm(w: Twist2d) -> Isometry2d:
    return Isometry2d.from_matrix(se2.expm(w.value))


@dispatch
def expm(w: Twist3d) -> Isometry3d:
    return Isometry3d.from_matrix(se3.expm(w.value))


@dispatch
def dexpm(w: Spin2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dexpm(w.value)


@dispatch
def dexpm(w: Spin3d) -> Array:
    return so3.dexpm(w.value)


@dispatch
def dexpm(w: Twist2d) -> Array:
    return se2.dexpm(w.value)


@dispatch
def dexpm(w: Twist3d) -> Array:
    return se3.dexpm(w.value)


@dispatch
def logm(g: Rotation2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.logm(g.value))


@dispatch
def logm(g: Rotation3d) -> Spin3d:
    return Spin3d.from_matrix(so3.logm(g.value))


@dispatch
def logm(g: Isometry2d) -> Twist2d:
    return Twist2d.from_matrix(se2.logm(g.value))


@dispatch
def logm(g: Isometry3d) -> Twist3d:
    return Twist3d.from_matrix(se3.logm(g.value))


@dispatch
def dlogm(w: Spin2d) -> Array:  # type: ignore[reportRedeclaration]
    return so2.dlogm(w.value)


@dispatch
def dlogm(w: Spin3d) -> Array:
    return so3.dlogm(w.value)


@dispatch
def dlogm(w: Twist2d) -> Array:
    return se2.dlogm(w.value)


@dispatch
def dlogm(w: Twist3d) -> Array:
    return se3.dlogm(w.value)


@dispatch
def lplus(g: Rotation2d, w: Spin2d) -> Rotation2d:  # type: ignore[reportRedeclaration]
    return Rotation2d.from_matrix(so2.lplus(g.value, w.value))


@dispatch
def lplus(g: Rotation3d, w: Spin3d) -> Rotation3d:
    return Rotation3d.from_matrix(so3.lplus(g.value, w.value))


@dispatch
def lplus(g: Isometry2d, w: Twist2d) -> Isometry2d:
    return Isometry2d.from_matrix(se2.lplus(g.value, w.value))


@dispatch
def lplus(g: Isometry3d, w: Twist3d) -> Isometry3d:
    return Isometry3d.from_matrix(se3.lplus(g.value, w.value))


@dispatch
def rplus(g: Rotation2d, w: Spin2d) -> Rotation2d:  # type: ignore[reportRedeclaration]
    return Rotation2d.from_matrix(so2.rplus(g.value, w.value))


@dispatch
def rplus(g: Rotation3d, w: Spin3d) -> Rotation3d:
    return Rotation3d.from_matrix(so3.rplus(g.value, w.value))


@dispatch
def rplus(g: Isometry2d, w: Twist2d) -> Isometry2d:
    return Isometry2d.from_matrix(se2.rplus(g.value, w.value))


@dispatch
def rplus(g: Isometry3d, w: Twist3d) -> Isometry3d:
    return Isometry3d.from_matrix(se3.rplus(g.value, w.value))


@dispatch
def lminus(g: Rotation2d, h: Rotation2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.lminus(g.value, h.value))


@dispatch
def lminus(g: Rotation3d, h: Rotation3d) -> Spin3d:
    return Spin3d.from_matrix(so3.lminus(g.value, h.value))


@dispatch
def lminus(g: Isometry2d, h: Isometry2d) -> Twist2d:
    return Twist2d.from_matrix(se2.lminus(g.value, h.value))


@dispatch
def lminus(g: Isometry3d, h: Isometry3d) -> Twist3d:
    return Twist3d.from_matrix(se3.lminus(g.value, h.value))


@dispatch
def rminus(g: Rotation2d, h: Rotation2d) -> Spin2d:  # type: ignore[reportRedeclaration]
    return Spin2d.from_matrix(so2.rminus(g.value, h.value))


@dispatch
def rminus(g: Rotation3d, h: Rotation3d) -> Spin3d:
    return Spin3d.from_matrix(so3.rminus(g.value, h.value))


@dispatch
def rminus(g: Isometry2d, h: Isometry2d) -> Twist2d:
    return Twist2d.from_matrix(se2.rminus(g.value, h.value))


@dispatch
def rminus(g: Isometry3d, h: Isometry3d) -> Twist3d:
    return Twist3d.from_matrix(se3.rminus(g.value, h.value))
