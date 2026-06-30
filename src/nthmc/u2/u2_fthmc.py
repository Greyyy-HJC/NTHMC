"""JAX field-transformed HMC for 2D U(2)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from tqdm import tqdm

from nthmc.u2.u2_observables import (
    action_from_field,
    identity_field,
    plaquette_mean_from_field,
    topology_from_field,
    u2_exp,
    u2_mul,
    u2_normalize,
)

Array = Any


class HMCU2FT:
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
        force_field_transformation=None,
        force_compute_jac_logdet=None,
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
        self.tune_step_size_enabled = tune_step_size
        self.key = jax.random.PRNGKey(seed)
        self._compiled_step = None

    def initialize(self) -> Array:
        return identity_field(self.lattice_size)

    def original_action(self, links: Array) -> Array:
        return action_from_field(links, self.beta)

    def new_action(self, links_new: Array) -> Array:
        links_ori = self.field_transformation(links_new)
        return self.original_action(links_ori) - self.compute_jac_logdet(links_new[jnp.newaxis, ...])[0]

    def _make_step(self):
        n_steps = self.n_steps
        dt = self.dt
        lam = 0.1931833

        def force_links(links: Array) -> Array:
            def varied_action(algebra: Array) -> Array:
                return self.new_action(u2_mul(u2_exp(algebra), links))

            return jax.grad(varied_action)(jnp.zeros((*links.shape[:-1], 4), dtype=links.dtype))

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
            h_old = self.new_action(links) + 0.5 * jnp.sum(momenta**2)
            new_links, new_momenta = omelyan(links, momenta)
            h_new = self.new_action(new_links) + 0.5 * jnp.sum(new_momenta**2)
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
        print(f">>> Using FT step size: {self.dt:.6f}")

    def thermalize(self, *, n_tune_steps: int = 1000) -> tuple[Array, list[float], float]:
        links = self.initialize()
        if self.tune_step_size_enabled:
            self.tune_step_size(n_tune_steps=n_tune_steps, links=links)
        plaq_values: list[float] = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing FT-HMC"):
            links_ori = u2_normalize(self.observable_field_transformation(links))
            plaq_values.append(float(plaquette_mean_from_field(links_ori)))
            links, accepted, _ = self.metropolis_step(links)
            acceptance_count += int(accepted)
        return links, plaq_values, acceptance_count / max(self.n_thermalization_steps, 1)

    def run(self, n_iterations: int, links: Array, *, store_interval: int = 1, save_config: bool = False):
        configs: list[Array] = []
        plaq_values: list[float] = []
        hamiltonians: list[float] = []
        topological_charges: list[float] = []
        acceptance_count = 0
        for index in tqdm(range(n_iterations), desc="Running FT-HMC"):
            links, accepted, h = self.metropolis_step(links)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                links_ori = u2_normalize(self.observable_field_transformation(links))
                if save_config:
                    configs.append(links_ori)
                plaq_values.append(float(plaquette_mean_from_field(links_ori)))
                hamiltonians.append(h)
                topological_charges.append(float(topology_from_field(links_ori)))
        return configs, plaq_values, acceptance_count / max(n_iterations, 1), topological_charges, hamiltonians
