import jax.numpy as jnp

from fiber.groups.se3 import SE3, se3

g = se3.from_vector([1, 0, 0, 0, 0, 0])
h = SE3.from_matrix(jnp.eye(4))

print(h @ g)
