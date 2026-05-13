import os
import time

os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["EQX_ON_ERROR"] = "nan"

import jax
import jax.numpy as jnp
import jax.random as jr

import fiberv3 as fiber
from fiberv3 import Isometry3d

jnp.set_printoptions(precision=3, suppress=True)

if __name__ == "__main__":
    g, w = fiber.random.gaussian(jr.key(0), Isometry3d.eye(), jnp.eye(6), shape=(32,))

    @jax.jit
    def mean(samples):
        return fiber.random.mean(samples, rtol=1e-1, atol=1e-1)

    @jax.jit
    def cov(mean, samples):
        return fiber.random.cov(mean, samples)

    start_t = time.time()
    m = mean(g)
    # cov(m, g)
    print(time.time() - start_t)

    start_t = time.time()
    m = mean(g)
    # c = cov(m, g)
    print(time.time() - start_t)
    # print(c)
