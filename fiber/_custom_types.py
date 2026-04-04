from typing import TYPE_CHECKING, Any

import numpy as np
from jaxtyping import Array, Bool, Float, Int, PyTree, Real, Shaped
from jaxtyping import ArrayLike as _ArrayLike

# the following types come from:
# https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_custom_types.py
if TYPE_CHECKING:
    type BoolScalarLike = bool | Array | np.ndarray
    type FloatScalarLike = float | Array | np.ndarray
    type IntScalarLike = int | Array | np.ndarray
    type RealScalarLike = bool | int | float | Array | np.ndarray
else:
    type BoolScalarLike = Bool[_ArrayLike, ""]
    FloatScalarLike = Float[_ArrayLike, ""]
    IntScalarLike = Int[_ArrayLike, ""]
    RealScalarLike = Real[_ArrayLike, ""]

# this matches the numpy `ArrayLike` type
type ArrayLike = _ArrayLike | Any

# the following types come from:
# https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_custom_types.py
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]
DenseInfo = dict[str, PyTree[Array]]
