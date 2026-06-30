"""JAX field-transformed HMC for 2D U(1)."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from tqdm import tqdm

from nthmc.u1.u1_observables import action, plaq_mean_from_field, regularize, topo_from_field

Array = Any


class JaxFTHMCResult(NamedTuple):
    theta: Array
    therm_plaq: Array
    therm_acceptance_rate: Array
    plaq: Array
    acceptance_rate: Array
    topo: Array
    hamiltonians: Array


def build_fthmc_chain(
    transform: Any,
    *,
    beta: float,
    n_thermalization: int,
    n_configs: int,
    n_steps: int,
    step_size: float,
) -> Any:
    lattice_size = transform.lattice_size
    lam = 0.1931833
    params = transform.params

    def new_action(theta: Array) -> Array:
        return transform.new_action_with_params(params, theta, beta)

    force = jax.grad(new_action)

    def omelyan(theta: Array, pi: Array) -> tuple[Array, Array]:
        pi = pi - lam * step_size * force(theta)

        def body(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            theta_i, pi_i = carry
            theta_i = theta_i + 0.5 * step_size * pi_i
            pi_i = pi_i - (1 - 2 * lam) * step_size * force(theta_i)
            theta_i = theta_i + 0.5 * step_size * pi_i
            pi_i = jax.lax.cond(i != n_steps - 1, lambda p: p - 2 * lam * step_size * force(theta_i), lambda p: p, pi_i)
            return theta_i, pi_i

        theta, pi = jax.lax.fori_loop(0, n_steps, body, (theta, pi))
        return regularize(theta), pi - lam * step_size * force(theta)

    def metropolis_step(theta: Array, key: Array) -> tuple[Array, Array, Array, Array]:
        key_pi, key_accept, key_next = jax.random.split(key, 3)
        pi = jax.random.normal(key_pi, theta.shape, dtype=theta.dtype)
        h_old = new_action(theta) + 0.5 * jnp.sum(pi**2)
        theta_new, pi_new = omelyan(theta, pi)
        h_new = new_action(theta_new) + 0.5 * jnp.sum(pi_new**2)
        accept_prob = jnp.minimum(1.0, jnp.exp(-(h_new - h_old)))
        accepted = jax.random.uniform(key_accept, (), dtype=theta.dtype) < accept_prob
        return jnp.where(accepted, theta_new, theta), key_next, accepted, jnp.where(accepted, h_new, h_old)

    def observable(theta: Array) -> tuple[Array, Array]:
        theta_ori = regularize(transform.forward_with_params(params, theta[jnp.newaxis, ...])[0])
        return plaq_mean_from_field(theta_ori), topo_from_field(theta_ori)

    def chain(key: Array) -> JaxFTHMCResult:
        theta0 = jnp.zeros((2, lattice_size, lattice_size), dtype=jnp.float32)

        def therm_body(carry: tuple[Array, Array], _: Array):
            theta, key_i = carry
            plaq, _ = observable(theta)
            theta, key_i, accepted, _ = metropolis_step(theta, key_i)
            return (theta, key_i), (plaq, accepted)

        (theta, key), (therm_plaq, therm_accepted) = jax.lax.scan(therm_body, (theta0, key), xs=None, length=n_thermalization)

        def run_body(carry: tuple[Array, Array], _: Array):
            theta, key_i = carry
            theta, key_i, accepted, h = metropolis_step(theta, key_i)
            plaq, topo = observable(theta)
            return (theta, key_i), (plaq, accepted, topo, h)

        (theta, _), (plaq, accepted, topo, hamiltonians) = jax.lax.scan(run_body, (theta, key), xs=None, length=n_configs)
        return JaxFTHMCResult(theta, therm_plaq, jnp.mean(therm_accepted.astype(jnp.float32)), plaq, jnp.mean(accepted.astype(jnp.float32)), topo, hamiltonians)

    return jax.jit(chain)


class HMCU1FT:
    def __init__(
        self,
        lattice_size: int,
        beta: float,
        n_thermalization_steps: int,
        n_steps: int,
        step_size: float,
        *,
        field_transformation,
        compute_jac_logdet,
        observable_field_transformation=None,
        device: str = "cpu",
        tune_step_size: bool = True,
        seed: int = 0,
    ) -> None:
        self.lattice_size = lattice_size
        self.beta = float(beta)
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = float(step_size)
        self.field_transformation = field_transformation
        self.observable_field_transformation = observable_field_transformation or field_transformation
        self.compute_jac_logdet = compute_jac_logdet
        self.key = jax.random.PRNGKey(seed)
        self._compiled_step = None
        self.tune_step_size_enabled = tune_step_size

    def initialize(self) -> Array:
        return jnp.zeros((2, self.lattice_size, self.lattice_size), dtype=jnp.float32)

    def original_action(self, theta: Array) -> Array:
        return action(theta, self.beta)

    def new_action(self, theta_new: Array) -> Array:
        theta_ori = self.field_transformation(theta_new)
        return self.original_action(theta_ori) - self.compute_jac_logdet(theta_new[jnp.newaxis, ...])[0]

    def new_force(self, theta_new: Array) -> Array:
        return jax.grad(self.new_action)(theta_new)

    def _make_step(self):
        n_steps = self.n_steps
        dt = self.dt
        lam = 0.1931833

        def omelyan(theta: Array, pi: Array) -> tuple[Array, Array]:
            pi = pi - lam * dt * self.new_force(theta)

            def body(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                theta_i, pi_i = carry
                theta_i = theta_i + 0.5 * dt * pi_i
                pi_i = pi_i - (1 - 2 * lam) * dt * self.new_force(theta_i)
                theta_i = theta_i + 0.5 * dt * pi_i
                pi_i = jax.lax.cond(i != n_steps - 1, lambda p: p - 2 * lam * dt * self.new_force(theta_i), lambda p: p, pi_i)
                return theta_i, pi_i

            theta, pi = jax.lax.fori_loop(0, n_steps, body, (theta, pi))
            return regularize(theta), pi - lam * dt * self.new_force(theta)

        def step(theta: Array, key: Array) -> tuple[Array, Array, Array, Array]:
            key_pi, key_accept, key_next = jax.random.split(key, 3)
            pi = jax.random.normal(key_pi, theta.shape, dtype=theta.dtype)
            h_old = self.new_action(theta) + 0.5 * jnp.sum(pi**2)
            theta_new, pi_new = omelyan(theta, pi)
            h_new = self.new_action(theta_new) + 0.5 * jnp.sum(pi_new**2)
            accept_prob = jnp.minimum(1.0, jnp.exp(-(h_new - h_old)))
            accepted = jax.random.uniform(key_accept, (), dtype=theta.dtype) < accept_prob
            return jnp.where(accepted, theta_new, theta), key_next, accepted, jnp.where(accepted, h_new, h_old)

        return jax.jit(step)

    def metropolis_step(self, theta: Array) -> tuple[Array, bool, float]:
        if self._compiled_step is None:
            self._compiled_step = self._make_step()
        theta, self.key, accepted, h = self._compiled_step(theta, self.key)
        return theta, bool(accepted), float(h)

    def tune_step_size(self, **_: Any) -> None:
        print(f">>> Using FT step size: {self.dt:.6f}")

    def thermalize(self, *, n_tune_steps: int = 1000) -> tuple[Array, list[float], float]:
        theta = self.initialize()
        if self.tune_step_size_enabled:
            self.tune_step_size(n_tune_steps=n_tune_steps, theta=theta)
        plaq_values: list[float] = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing FT-HMC"):
            theta_ori = regularize(self.observable_field_transformation(theta))
            plaq_values.append(float(plaq_mean_from_field(theta_ori)))
            theta, accepted, _ = self.metropolis_step(theta)
            acceptance_count += int(accepted)
        return theta, plaq_values, acceptance_count / max(self.n_thermalization_steps, 1)

    def run(self, n_iterations: int, theta: Array, *, store_interval: int = 1, save_config: bool = False):
        configs: list[Array] = []
        plaq_values: list[float] = []
        hamiltonians: list[float] = []
        topological_charges: list[float] = []
        acceptance_count = 0
        for index in tqdm(range(n_iterations), desc="Running FT-HMC"):
            theta, accepted, h = self.metropolis_step(theta)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                theta_ori = regularize(self.observable_field_transformation(theta))
                if save_config:
                    configs.append(theta_ori)
                plaq_values.append(float(plaq_mean_from_field(theta_ori)))
                hamiltonians.append(h)
                topological_charges.append(float(topo_from_field(theta_ori)))
        return configs, plaq_values, acceptance_count / max(n_iterations, 1), topological_charges, hamiltonians
