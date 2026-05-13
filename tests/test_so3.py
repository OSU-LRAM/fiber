import os
import time

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["EQX_ON_ERROR"] = "nan"

import jax
import jax.numpy as jnp
import jax.random as jr

import fiberv3 as fiber
from fiberv3 import Isometry3d, Twist3d

jnp.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    g = Isometry3d.from_euclidean("xyz", [0, 0, 0, 45, 0, 0])
    h = Isometry3d.from_euclidean("xyz", [1, 0, 0, 0, 0, 0])

    w = Twist3d.from_vector([0, 0, 0, 0, 0, 0], point=g)
    v = Twist3d.from_vector([1, 0, 0, 0, 0, 0], point=h)

    interp = fiber.LocalBundleInterpolation(0.0, 1.0, w, v)

    print(interp.evaluate(0.5, 0.75).point)
