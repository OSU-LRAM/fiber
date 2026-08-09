# fiber

fiber is a JAX library for Lie group operations, targeting robotics
applications. It provides support for `SO(2)`, `SE(2)`, `SO(3)`, and `SE(3)`
along with the differential-geometric operators needed to work with them (e.g.,
`expm`, `logm`, `dexp`, etc.).

## Main Features

- Rigid-body transformations and Lie algebra operations for `SO(2)`, `SE(2)`,
  `SO(3)`, and `SE(3)`
- A consistent set of differential-geometric operators (exponential/log maps,
  adjoint actions, and more) for building controllers, estimators, and
  planners that respect the geometry of a robot's configuration space
- GPU-accelerated implementation for fast hardware deployments
- Lie group integrators and interpolation schemes for simulating dynamics

## Installation

fiber requires Python 3.14+, and can be installed using pip or uv

```bash
# pip installation
pip install git+https://github.com/OSU-LRAM/fiber.git

# uv installation
uv pip install git+https://github.com/OSU-LRAM/fiber.git

# or add to your uv project using
uv add git+ssh://git@github.com/OSU-LRAM/fiber.git
```

## Usage

```python
import jax
import jax.numpy as jnp
import fiber
from fiber import Rotation3d, Spin3d, Isometry3d, Twist3d

# Build a rotation from an axis-angle (algebra) vector via the exponential map
w_vec = jnp.array([0.1, -0.2, 0.3])
R0 = Rotation3d.eye()
w = Spin3d.from_vector(w_vec, point=R0)
R1 = fiber.expm(w)

# Compose, invert, and recover the algebra element
R2 = R1 @ R1
w_back = fiber.logm(R2)
theta = R2.angle

# SE(3) rigid transform driven forward by a twist
T = Isometry3d.from_pq(jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
xi = Twist3d.from_vector(jnp.concatenate([jnp.zeros(3), w_vec]), point=T)
T1 = fiber.expm(xi) @ T

# Adjoint and coadjoint actions
Ad_w = fiber.Adj(R1, w)
wrench = fiber.Wrench3d.from_vector(jnp.ones(6), point=T)
coAd_wrench = fiber.dAdj(T, wrench)

# Sample a Gaussian distribution on the group
key = jax.random.key(0)
mean_sample, tangent_sample = fiber.random.gaussian(key, R0, 0.01 * jnp.eye(3))
```

Every operation above also has a raw-array counterpart under `fiber.numpy`,
grouped by group (`fiber.numpy.so3`, `fiber.numpy.se3`, ...) for use without
the typed wrapper classes:

```python
import fiber.numpy as fnp

w_hat = fnp.skew3(w_vec)     # 3x3 skew-symmetric matrix
R_mat = fnp.so3.expm(w_hat)  # 3x3 rotation matrix
```

## License

fiber is released under the MIT license.
