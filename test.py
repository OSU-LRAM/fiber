import jax.numpy as jnp

import fiber
from fiber import Isometry, Twist

g = Isometry.from_matrix(jnp.eye(4))

v = Twist.from_vector([1, 2, 3, 4, 5, 6]) * 0.1

print(Twist.size)
print(fiber.logm(fiber.expm(v)))

print(fiber.Adj(g, v))
