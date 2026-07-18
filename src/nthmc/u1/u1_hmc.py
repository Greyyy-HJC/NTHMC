"""JAX standard HMC for 2D U(1) lattice gauge theory."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from tqdm import tqdm

from nthmc.core.hmc_tuning import tune_step_size as tune_hmc_step_size
from nthmc.u1.u1_observables import action, plaq_mean_from_field, regularize, topo_from_field

Array = Any


class HMCU1:
    def __init__(
        self,
        lattice_size: int,
        beta: float,
        n_thermalization_steps: int,
        n_steps: int,
        step_size: float,
        *,
        device: str = "cpu",
        tune_step_size: bool = True,
        seed: int = 0,
    ) -> None:
        self.lattice_size = lattice_size
        self.beta = float(beta)
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = float(step_size)
        self.device = device
        self.tune_step_size_enabled = tune_step_size
        self.key = jax.random.PRNGKey(seed)
        self._compiled_step = None

    def initialize(self) -> Array:
        return jnp.zeros((2, self.lattice_size, self.lattice_size), dtype=jnp.float32)

    def action(self, theta: Array) -> Array:
        return action(theta, self.beta)

    def force(self, theta: Array) -> Array:
        return jax.grad(self.action)(theta)

    def _make_step(self):
        beta = self.beta
        n_steps = self.n_steps
        lam = 0.1931833

        def energy(theta: Array, pi: Array) -> Array:
            return action(theta, beta) + 0.5 * jnp.sum(pi**2)

        force = jax.grad(lambda x: action(x, beta))

        def omelyan(theta: Array, pi: Array, dt: Array) -> tuple[Array, Array]:
            pi = pi - lam * dt * force(theta)

            def body(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                theta_i, pi_i = carry
                theta_i = theta_i + 0.5 * dt * pi_i
                pi_i = pi_i - (1 - 2 * lam) * dt * force(theta_i)
                theta_i = theta_i + 0.5 * dt * pi_i
                pi_i = jax.lax.cond(i != n_steps - 1, lambda p: p - 2 * lam * dt * force(theta_i), lambda p: p, pi_i)
                return theta_i, pi_i

            theta, pi = jax.lax.fori_loop(0, n_steps, body, (theta, pi))
            return regularize(theta), pi - lam * dt * force(theta)

        def step(theta: Array, key: Array, dt: Array) -> tuple[Array, Array, Array, Array]:
            key_pi, key_accept, key_next = jax.random.split(key, 3)
            pi = jax.random.normal(key_pi, theta.shape, dtype=theta.dtype)
            h_old = energy(theta, pi)
            theta_new, pi_new = omelyan(theta, pi, dt)
            h_new = energy(theta_new, pi_new)
            accept_prob = jnp.minimum(1.0, jnp.exp(-(h_new - h_old)))
            accepted = jax.random.uniform(key_accept, (), dtype=theta.dtype) < accept_prob
            return jnp.where(accepted, theta_new, theta), key_next, accepted, jnp.where(accepted, h_new, h_old)

        return jax.jit(step)

    def _metropolis_step_at(self, theta: Array, step_size: float) -> tuple[Array, bool, float]:
        if self._compiled_step is None:
            self._compiled_step = self._make_step()
        dt = jnp.asarray(step_size, dtype=theta.dtype)
        theta, self.key, accepted, hamiltonian = self._compiled_step(theta, self.key, dt)
        return theta, bool(accepted), float(hamiltonian)

    def metropolis_step(self, theta: Array) -> tuple[Array, bool, float]:
        return self._metropolis_step_at(theta, self.dt)

    def tune_step_size(
        self,
        *,
        n_tune_steps: int = 2000,
        target_rate: float = 0.70,
        target_tolerance: float = 0.15,
        max_attempts: int = 10,
        theta: Array | None = None,
    ) -> None:
        theta = self.initialize() if theta is None else jnp.asarray(theta)
        self.dt = tune_hmc_step_size(
            theta,
            self.dt,
            self._metropolis_step_at,
            n_tune_steps=n_tune_steps,
            target_rate=target_rate,
            target_tolerance=target_tolerance,
            max_attempts=max_attempts,
        )
        print(f">>> Using step size: {self.dt:.6f}")

    def thermalize(self, *, n_tune_steps: int = 2000) -> tuple[Array, list[float], float]:
        theta = self.initialize()
        if self.tune_step_size_enabled:
            self.tune_step_size(n_tune_steps=n_tune_steps, theta=theta)
        plaq_values = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            theta = regularize(theta)
            plaq_values.append(float(plaq_mean_from_field(theta)))
            theta, accepted, _ = self.metropolis_step(theta)
            acceptance_count += int(accepted)
        denom = max(self.n_thermalization_steps, 1)
        return theta, plaq_values, acceptance_count / denom

    def run(
        self,
        n_iterations: int,
        theta: Array,
        *,
        store_interval: int = 1,
        save_config: bool = True,
    ) -> tuple[list[Array], list[float], float, list[float], list[float]]:
        configs: list[Array] = []
        plaq_values: list[float] = []
        hamiltonians: list[float] = []
        topological_charges: list[float] = []
        acceptance_count = 0
        for index in tqdm(range(n_iterations), desc="Running HMC"):
            theta, accepted, hamiltonian = self.metropolis_step(theta)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                theta = regularize(theta)
                if save_config:
                    configs.append(theta)
                plaq_values.append(float(plaq_mean_from_field(theta)))
                hamiltonians.append(hamiltonian)
                topological_charges.append(float(topo_from_field(theta)))
        return configs, plaq_values, acceptance_count / max(n_iterations, 1), topological_charges, hamiltonians
