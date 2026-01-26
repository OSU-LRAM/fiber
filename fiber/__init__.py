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

from . import linalg, random
from ._elements import Isometry, Twist, is_algebra_element, is_group_element
from ._interpolation import (
    AugmentedDirectInterpolation,
    AugmentedGeodesicInterpolation,
    DirectInterpolation,
    GeodesicInterpolation,
)
from ._operations import (
    Adj,
    Adj_inv,
    Adj_inv_op,
    Adj_op,
    adj,
    adj_op,
    dAdj,
    dadj,
    dAdj_inv,
    dadj_inv,
    dAdj_inv_op,
    dadj_inv_op,
    dAdj_op,
    dadj_op,
    dexpm,
    dlogm,
    expm,
    inv,
    lminus,
    logm,
    lplus,
    rminus,
    rplus,
)
from ._utils import join_state, split_state

__all__ = [
    "Twist",
    "Isometry",
    "is_group_element",
    "is_algebra_element",
    "linalg",
    "Adj",
    "Adj_op",
    "Adj_inv_op",
    "Adj_inv",
    "adj",
    "adj_op",
    "dAdj",
    "dadj",
    "dAdj_op",
    "dadj_op",
    "dAdj_inv",
    "dadj_inv",
    "dAdj_inv_op",
    "dadj_inv_op",
    "rminus",
    "rplus",
    "expm",
    "logm",
    "dexpm",
    "dlogm",
    "lminus",
    "lplus",
    "join_state",
    "split_state",
    "random",
    "inv",
    "DirectInterpolation",
    "AugmentedDirectInterpolation",
    "GeodesicInterpolation",
    "AugmentedGeodesicInterpolation",
]
