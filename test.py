import jax.numpy as jnp
import jax.random as jr

import fiber
from fiber import Isometry

g = Isometry.from_matrix(jnp.eye(4))

gs, vs = fiber.random.right_gaussian(g, key=jr.key(0))

print(gs)
print(vs)
