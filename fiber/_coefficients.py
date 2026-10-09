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

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array


def series_or_closed(
    theta2: Array, series: Callable[[Array], Array], closed: Callable[[Array], Array]
) -> Array:
    small = theta2 < 1e-2
    return jnp.where(small, series(theta2), closed(jnp.where(small, 1.0, theta2)))


def sinc(theta2: Array) -> Array:
    return series_or_closed(
        theta2,
        lambda x: 1 - x / 6 + x**2 / 120 - x**3 / 5040 + x**4 / 362880,
        lambda x: jnp.sin(jnp.sqrt(x)) / jnp.sqrt(x),
    )


def cosc(theta2: Array) -> Array:
    return series_or_closed(
        theta2,
        lambda x: 1 / 2 - x / 24 + x**2 / 720 - x**3 / 40320 + x**4 / 3628800,
        lambda x: 2 * jnp.sin(jnp.sqrt(x) / 2) ** 2 / x,
    )


def sinc3(theta2: Array) -> Array:
    return series_or_closed(
        theta2,
        lambda x: 1 / 6 - x / 120 + x**2 / 5040 - x**3 / 362880 + x**4 / 39916800,
        lambda x: (1 - sinc(x)) / x,
    )


def dlogc(theta2: Array) -> Array:
    return series_or_closed(
        theta2,
        lambda x: 1 / 12 + x / 720 + x**2 / 30240 + x**3 / 1209600 + x**4 / 47900160,
        lambda x: (1 - jnp.sqrt(x) / 2 / jnp.tan(jnp.sqrt(x) / 2)) / x,
    )
