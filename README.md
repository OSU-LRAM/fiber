# fiber

fiber is a JAX library for performing SE(3) operations, targeting geometric mechanics 
applications. Some of the operations implemented include: `expm`, `logm`, `dexpm`,
`dlogm`, and `adj`.

## Installation

fiber can be installed using pip or uv,

```bash
# pip installation
pip install https://github.com/OSU-LRAM/fiber.git

# uv installation
uv pip install https://github.com/OSU-LRAM/jaxgm.git

# or add to your uv project using,
uv add git+ssh://git@github.com:OSU-LRAM/fiber
```

## Example

```python
import fiber
from fiber import Isometry, Twist
import jax.numpy as jnp

# construct an SE(3) element and an se(3) element
# 
# The `Isometry` and `Twist` classes are PyTrees that support various JAX operations
# like `vmap`
g = Isometry.from_matrix(jnp.eye(4))
v = Twist.from_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

# compose the elements
g @ v

# or perform various operations using them
fiber.Adj(g, v)

# the fiber API doesn't force you to use the `Isometry` or `Twist` PyTrees. You can
# also use plain JAX arrays
g_matrix = g.as_matrix()
v_matrix = v.as_matrix()
fiber.Adj(g, v)
```

## License

fiber is released under the MIT license.
