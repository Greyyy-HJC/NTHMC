"""JAX standard HMC for 2D U(2) lattice gauge theory."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from tqdm import tqdm

from nthmc.u2.u2_observables import (
    action_from_field,
    force_from_field,
    identity_field,
    plaquette_mean_from_field,
    topology_from_field,
    u2_exp,
    u2_mul,
    u2_normalize,
)

Array = Any


class HMCU2:
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
        return identity_field(self.lattice_size)

    def action(self, links: Array) -> Array:
        return action_from_field(links, self.beta)

    def force(self, links: Array) -> Array:
        return force_from_field(links, self.beta)

    def update_links(self, links: Array, momenta: Array, coefficient: float) -> Array:
        return u2_mul(u2_exp(coefficient * self.dt * momenta), links)

    def _make_step(self):
        beta = self.beta
        n_steps = self.n_steps
        dt = self.dt
        lam = 0.1931833
        force = jax.grad(lambda algebra, links: action_from_field(u2_mul(u2_exp(algebra), links), beta), argnums=0)

        def force_links(links: Array) -> Array:
            return force(jnp.zeros((*links.shape[:-1], 4), dtype=links.dtype), links)

        def update(links: Array, momenta: Array, coefficient: float) -> Array:
            return u2_mul(u2_exp(coefficient * dt * momenta), links)

        def omelyan(links: Array, momenta: Array) -> tuple[Array, Array]:
            momenta = momenta - lam * dt * force_links(links)

            def body(i: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
                links_i, mom_i = carry
                links_i = update(links_i, mom_i, 0.5)
                mom_i = mom_i - (1 - 2 * lam) * dt * force_links(links_i)
                links_i = update(links_i, mom_i, 0.5)
                mom_i = jax.lax.cond(i != n_steps - 1, lambda p: p - 2 * lam * dt * force_links(links_i), lambda p: p, mom_i)
                return links_i, mom_i

            links, momenta = jax.lax.fori_loop(0, n_steps, body, (links, momenta))
            return u2_normalize(links), momenta - lam * dt * force_links(links)

        def step(links: Array, key: Array) -> tuple[Array, Array, Array, Array]:
            key_p, key_accept, key_next = jax.random.split(key, 3)
            momenta = jax.random.normal(key_p, (*links.shape[:-1], 4), dtype=links.dtype)
            h_old = action_from_field(links, beta) + 0.5 * jnp.sum(momenta**2)
            new_links, new_momenta = omelyan(links, momenta)
            h_new = action_from_field(new_links, beta) + 0.5 * jnp.sum(new_momenta**2)
            accept_prob = jnp.minimum(1.0, jnp.exp(-(h_new - h_old)))
            accepted = jax.random.uniform(key_accept, (), dtype=links.dtype) < accept_prob
            return jnp.where(accepted, new_links, links), key_next, accepted, jnp.where(accepted, h_new, h_old)

        return jax.jit(step)

    def metropolis_step(self, links: Array) -> tuple[Array, bool, float]:
        if self._compiled_step is None:
            self._compiled_step = self._make_step()
        links, self.key, accepted, h = self._compiled_step(links, self.key)
        return links, bool(accepted), float(h)

    def tune_step_size(self, **_: Any) -> None:
        print(f">>> Using step size: {self.dt:.6f}")

    def thermalize(self, *, n_tune_steps: int = 2000) -> tuple[Array, list[float], float]:
        links = self.initialize()
        if self.tune_step_size_enabled:
            self.tune_step_size(n_tune_steps=n_tune_steps, links=links)
        plaq_values: list[float] = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing"):
            links = u2_normalize(links)
            plaq_values.append(float(plaquette_mean_from_field(links)))
            links, accepted, _ = self.metropolis_step(links)
            acceptance_count += int(accepted)
        return links, plaq_values, acceptance_count / max(self.n_thermalization_steps, 1)

    def run(self, n_iterations: int, links: Array, *, store_interval: int = 1, save_config: bool = True):
        configs: list[Array] = []
        plaq_values: list[float] = []
        hamiltonians: list[float] = []
        topological_charges: list[float] = []
        acceptance_count = 0
        for index in tqdm(range(n_iterations), desc="Running HMC"):
            links, accepted, h = self.metropolis_step(links)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                links = u2_normalize(links)
                if save_config:
                    configs.append(links)
                plaq_values.append(float(plaquette_mean_from_field(links)))
                hamiltonians.append(h)
                topological_charges.append(float(topology_from_field(links)))
        return configs, plaq_values, acceptance_count / max(n_iterations, 1), topological_charges, hamiltonians
