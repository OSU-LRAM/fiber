import diffrax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import matplotlib.pyplot as plt
from jaxtyping import Array, PRNGKeyArray, ScalarLike

import fiber
from fiber import Twist3d, Wrench3d
from fiber.internal import Args
from fiber.solvers import (
    AbstractImplicitVectorField,
    ImplicitTerm,
    UnsafeControlTerm,
    VariationalIntegrator,
)


class Func(AbstractImplicitVectorField[Twist3d, Wrench3d]):
    M: Array
    D1: Array
    D2: Array
    fg: Array
    fb: Array
    cog: Array
    cob: Array
    u: Array

    def __init__(self, control: Array):
        """Initialize the drift for the stochastic hydrodynamic model.

        Parameters
        ----------
        - `control`: The control input to apply for this simulation.
        """
        mass = 13.5
        self.M = self.build_mass_matrix(mass)
        self.D1, self.D2 = self.build_drag_matrices()
        self.fg, self.fb, self.cog, self.cob = self.build_hydrostatics(mass)
        self.u = control

    @staticmethod
    def build_mass_matrix(mass: float) -> Array:
        """Build the mass matrix, including the rigid-body and added mass matrices."""
        ixx, iyy, izz = 0.26, 0.23, 0.37
        xdu, ydv, zdw, kdp, mdq, ndr = 6.36, 7.12, 18.68, 0.189, 0.135, 0.222

        M_rb = jnp.diag(jnp.array([mass, mass, mass, ixx, iyy, izz]))
        M_a = jnp.diag(jnp.array([xdu, ydv, zdw, kdp, mdq, ndr]))

        return M_rb + M_a

    @staticmethod
    def build_drag_matrices() -> tuple[Array, Array]:
        """Build the linear and quadratic drag matrices."""
        # the referenced paper does not include drag in the y axis, which results in
        # wonky behavior. i've duplicated the x-axis drag and added some
        xu, yv, zw, kp, mq, nr = 13.7, 15.7, 33.0, 0.0, 0.8, 0.0
        xuu, yvv, zww, kpp, mqq, nrr = 141.0, 217.0, 190.0, 1.19, 0.47, 1.5

        D1 = jnp.diag(jnp.array([xu, yv, zw, kp, mq, nr]))
        D2 = jnp.diag(jnp.array([xuu, yvv, zww, kpp, mqq, nrr]))

        return D1, D2

    @staticmethod
    def build_hydrostatics(mass: float) -> tuple[Array, Array, Array, Array]:
        """Build the hydrostatic terms, including the weight, bouyancy, center of
        gravity, and center of bouyancy."""
        cog = jnp.array([0.0, 0.0, 0.0])
        cob = jnp.array([0.0, 0.0, -0.01])

        gravity = 9.81  # z points downwards
        weight = gravity * mass
        buoyancy = gravity * 0.0134 * 1000.0  # volume * fluid density * gravity

        fg = jnp.array([0.0, 0.0, weight])
        fb = jnp.array([0.0, 0.0, -buoyancy])

        return fg, fb, cog, cob

    def implicit_step(self, t: ScalarLike, y: Twist3d, args: Args) -> Wrench3d:
        (control,) = args
        p = self.M @ y.as_vector()
        return Wrench3d.from_vector(fiber.dlogm(y * control).T @ p)

    def __call__(self, t: ScalarLike, y: Twist3d, args: Args) -> Wrench3d:
        (control,) = args
        v = y.as_vector()

        p = self.M @ v

        r = jnp.zeros((6,))
        R = y.point.rotation.as_matrix()
        r = r.at[:3].set(R.T @ (self.fg + self.fb))
        tq = jnp.cross(self.cog, R.T @ self.fg) + jnp.cross(self.cob, R.T @ self.fb)
        r = r.at[3:].set(tq)
        r *= -1

        D1 = self.D1 @ v
        D2 = self.D2 @ (v * jnp.abs(v))
        D = D1 + D2

        g_next = fiber.rplus(y.point, control * y)
        residual = fiber.dlogm(-control * y) @ p + control * (self.u - D - r)

        return Wrench3d(g_next, residual)


def cvf(t: ScalarLike, y: Twist3d, args: Args):
    """Computes the state-independent diffusion matrix for the system.

    Parameters
    ----------
    - `t`: Ignored; the timepoint to evaluate the system at.
    - `y`: Ignored; A JAX array representing the state at which the diffusion should be
        evaluated at.
    - `args`: Ignored; additional arguments passed by the solver.

    Returns
    -------
    An (18, 12) diffusion matrix with the top-left part (12, 6) representing the
    configuration diffusion and the bottom-right (6, 6) part representing the velocity
    diffusion. We assume that there is no irreducible noise in the kinematics and set
    the configuration diffusion to zero accordingly.
    """
    H = jnp.sqrt(jnp.array([20.0, 20.0, 0.00001, 0.00001, 0.00001, 0.001]))
    return lx.DiagonalLinearOperator(H)


class VariationalSimulator(eqx.Module):
    vf: Func

    def __init__(self, control: Array):
        self.vf = Func(control)

    def __call__(
        self, key: PRNGKeyArray, ts: Array, y0: Twist3d, samples: int, dt: float = 0.05
    ) -> Array:
        """Simulate the dynamics of the underwater vehicle.

        Parameters
        ----------
        - `key`: A `jax.random.key` used to generate the Brownian motion.
            (Keyword only argument.)
        - `ts`: The timepoints to evaluate the solution at.
        - `y0`: The initial conditions.
        - `samples: The total number of forward rollouts to compute per solve.
        - `dt`: The time step. Defaults to 0.05s.

        Returns
        -------
        A JAX array of `(samples, ...)` forward simulations.
        """

        @eqx.filter_vmap
        def _sample(_key: PRNGKeyArray) -> Array:
            t0, t1 = ts[0], ts[-1]
            noise_shape = (Twist3d.nparams,)
            control = diffrax.VirtualBrownianTree(t0, t1, dt / 2, noise_shape, _key)
            drift = ImplicitTerm(self.vf)
            diffusion = UnsafeControlTerm(cvf, control)

            sol = diffrax.diffeqsolve(
                diffrax.MultiTerm(drift, diffusion),
                VariationalIntegrator(),
                t0=t0,
                t1=t1,
                y0=y0,
                dt0=dt,
                saveat=diffrax.SaveAt(ts=ts),
                args=(dt,),
            )

            return sol.ys  # type: ignore

        keys = jr.split(key, samples)
        return _sample(keys)


@eqx.filter_jit
def simulate(
    key: PRNGKeyArray,
    y0: Twist3d,
    control: Array,
    duration: float = 10.0,
    sampling_rate: float = 20.0,
    dt: float = 0.05,
    rollouts: int = 16,
) -> tuple[Array, Array, Array]:
    """Simulate an underwater vehicle subject to nonconservative stochastic forces.

    Parameters
    ----------
    - `key`: A `jax.random.key` used to generate the Brownian motion.
        (Keyword only argument.)
    - `y0`: The initial conditions.
    - `control: The control input to apply during the simulation.
    - `duration`: The rollout duration (s). Defaults to 10.0s.
    - `sampling_rate`: The rate at which to sample the dynamics. This is set to 20 Hz
        to match the rate that we would expect to get on a real vehicle.
    - `dt`: The integration timestep. Defaults to 0.05s.
    - `rollouts`: The total number of simulation rollouts to generate. Defaults to 16.

    Returns
    -------
    The evaluation times, the corresponding states, and the corresponding controls.
    """
    samples = int(duration * sampling_rate)
    ts = jnp.linspace(0.0, duration, samples)
    us = jnp.broadcast_to(control, (rollouts, ts.shape[0], control.shape[0]))
    simulator = VariationalSimulator(control)
    ys = simulator(key, ts, y0, rollouts, dt)
    return ts, ys, us


def draw_states(
    ax,
    states: Array,
    *,
    color=None,
    linewidth: float = 1.2,
    alpha: float = 0.5,
):
    if color is None:
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    for tr in range(states.shape[0]):
        ax.plot(
            states[tr, :, 0],
            states[tr, :, 1],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )


if __name__ == "__main__":
    y0 = Twist3d.from_vector(jnp.zeros(6))
    control = jnp.array([60.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    ts, ys, us = simulate(jr.key(0), y0, control, rollouts=100)

    fig, ax = plt.subplots()

    red, black = (1, 0, 0, 0.8), (0, 0, 0, 1.0)

    draw_states(ax, ys.point.flatten(), color="r")
    indices = [i * int(ys.shape[1] / 3) for i in range(1, 4)]

    ax.set_title("Simulation Distribution")
    ax.set_aspect("equal", "datalim")
    fig.savefig("simulation_dataset.pdf")
