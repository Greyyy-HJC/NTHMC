"""Field-transformed HMC for 2D U(2) lattice gauge theory."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from nthmc.u2.u2_observables import (
    identity_field,
    plaquette_mean_from_field,
    real_dtype_from_links,
    topology_from_field,
    u2_exp,
    u2_mul,
    u2_normalize,
)


class HMCU2FT:
    """HMC sampler in transformed U(2) coordinates."""

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
    ) -> None:
        self.lattice_size = lattice_size
        self.beta = beta
        self.n_thermalization_steps = n_thermalization_steps
        self.n_steps = n_steps
        self.dt = step_size
        self.field_transformation = field_transformation
        self.observable_field_transformation = observable_field_transformation or field_transformation
        self.compute_jac_logdet = compute_jac_logdet
        self.force_field_transformation = force_field_transformation or field_transformation
        self.force_compute_jac_logdet = force_compute_jac_logdet or compute_jac_logdet
        self.device = torch.device(device)
        self.tune_step_size_enabled = tune_step_size

    def initialize(self) -> torch.Tensor:
        return identity_field(self.lattice_size, device=self.device)

    def original_action(self, links: torch.Tensor) -> torch.Tensor:
        plaq_mean = plaquette_mean_from_field(links)
        volume = self.lattice_size**2
        action_value = self.beta * volume * (1 - plaq_mean)
        assert action_value.dim() == 0
        return action_value

    def new_action(self, links_new: torch.Tensor, *, for_force: bool = False) -> torch.Tensor:
        field_transformation = self.force_field_transformation if for_force else self.field_transformation
        compute_jac_logdet = self.force_compute_jac_logdet if for_force else self.compute_jac_logdet
        links_ori = field_transformation(links_new)
        original_action_value = self.original_action(links_ori)
        jacobian_log_det = compute_jac_logdet(links_new.unsqueeze(0)).squeeze(0)
        action_value = original_action_value - jacobian_log_det
        assert action_value.dim() == 0
        return action_value

    def new_force(self, links_new: torch.Tensor) -> torch.Tensor:
        algebra = torch.zeros(
            (*links_new.shape[:-1], 4),
            device=self.device,
            dtype=real_dtype_from_links(links_new),
            requires_grad=True,
        )
        varied_links = u2_mul(u2_exp(algebra), links_new.detach())
        action_value = self.new_action(varied_links, for_force=True)
        return torch.autograd.grad(action_value, algebra)[0].detach()

    def update_links(self, links: torch.Tensor, momenta: torch.Tensor, coefficient: float) -> torch.Tensor:
        delta = u2_exp(coefficient * self.dt * momenta)
        return u2_mul(delta, links)

    def omelyan(self, links: torch.Tensor, momenta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = 0.1931833
        links_next = links
        momenta_next = momenta - lam * self.dt * self.new_force(links_next)
        for step_index in range(self.n_steps):
            links_next = self.update_links(links_next, momenta_next, 0.5)
            momenta_next = momenta_next - (1 - 2 * lam) * self.dt * self.new_force(links_next)
            links_next = self.update_links(links_next, momenta_next, 0.5)
            if step_index != self.n_steps - 1:
                momenta_next = momenta_next - 2 * lam * self.dt * self.new_force(links_next)
        momenta_next = momenta_next - lam * self.dt * self.new_force(links_next)
        return u2_normalize(links_next), momenta_next

    def metropolis_step(self, links: torch.Tensor) -> tuple[torch.Tensor, bool, float]:
        momenta = torch.randn((*links.shape[:-1], 4), device=self.device, dtype=real_dtype_from_links(links))
        h_old = self.new_action(links) + 0.5 * torch.sum(momenta**2)
        new_links, new_momenta = self.omelyan(links.clone(), momenta.clone())
        h_new = self.new_action(new_links) + 0.5 * torch.sum(new_momenta**2)
        accept_prob = torch.exp(-(h_new - h_old)).clamp(max=1)
        if torch.rand([], device=self.device) < accept_prob:
            return new_links, True, float(h_new.detach().cpu())
        return links, False, float(h_old.detach().cpu())

    def tune_step_size(
        self,
        *,
        n_tune_steps: int = 1000,
        target_rate: float = 0.75,
        target_tolerance: float = 0.15,
        max_attempts: int = 10,
        links: torch.Tensor | None = None,
    ) -> None:
        links = self.initialize() if links is None else links.clone()
        step_min = 1e-6
        step_max = 1.0
        best_dt = self.dt
        best_rate_diff = float("inf")
        current_rate = 0.0

        for attempt in range(max_attempts):
            acceptance_count = 0
            for _ in tqdm(range(n_tune_steps), desc=f"Tuning FT step size ({attempt + 1}/{max_attempts})"):
                links, accepted, _ = self.metropolis_step(links)
                acceptance_count += int(accepted)
            current_rate = acceptance_count / n_tune_steps
            rate_diff = abs(current_rate - target_rate)
            print(f"Step size: {self.dt:.6f}, acceptance rate: {current_rate:.2%}")
            if rate_diff < best_rate_diff:
                best_dt = self.dt
                best_rate_diff = rate_diff
            if rate_diff <= target_tolerance:
                break
            if current_rate > target_rate:
                step_min = self.dt
                self.dt = min((self.dt + step_max) / 2, step_max)
            else:
                step_max = self.dt
                self.dt = max((self.dt + step_min) / 2, step_min)

        if abs(current_rate - target_rate) > target_tolerance:
            self.dt = best_dt
        print(f">>> Using FT step size: {self.dt:.6f}")

    def thermalize(self, *, n_tune_steps: int = 1000) -> tuple[torch.Tensor, list[float], float]:
        links = self.initialize()
        if self.tune_step_size_enabled:
            print(">>> Initial FT thermalization for step-size tuning")
            initial_plaq = []
            for _ in tqdm(range(self.n_thermalization_steps), desc="Initial FT thermalization"):
                links, _, _ = self.metropolis_step(links)
                with torch.no_grad():
                    links_ori = u2_normalize(self.observable_field_transformation(links))
                initial_plaq.append(round(float(plaquette_mean_from_field(links_ori).detach().cpu()), 4))
            if len(initial_plaq) >= 10:
                n_complete = (len(initial_plaq) // 10) * 10
                means = np.mean(np.array(initial_plaq[-n_complete:]).reshape(-1, 10), axis=1)
                print(f"Initial FT thermalization plaquette means: {means}")
            self.tune_step_size(n_tune_steps=n_tune_steps, links=links)
        else:
            print(f">>> Using FT step size without tuning: {self.dt:.6f}")

        links = self.initialize()
        plaq_values = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing FT-HMC"):
            links = u2_normalize(links)
            with torch.no_grad():
                links_ori = u2_normalize(self.observable_field_transformation(links))
            plaq_values.append(float(plaquette_mean_from_field(links_ori).detach().cpu()))
            links, accepted, _ = self.metropolis_step(links)
            acceptance_count += int(accepted)

        return links, plaq_values, acceptance_count / self.n_thermalization_steps

    def run(
        self,
        n_iterations: int,
        links: torch.Tensor,
        *,
        store_interval: int = 1,
        save_config: bool = False,
    ) -> tuple[list[torch.Tensor], list[float], float, list[float], list[float]]:
        configs = []
        plaq_values = []
        hamiltonians = []
        topological_charges = []
        acceptance_count = 0

        for index in tqdm(range(n_iterations), desc="Running FT-HMC"):
            links, accepted, hamiltonian = self.metropolis_step(links)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                links = u2_normalize(links)
                with torch.no_grad():
                    links_ori = u2_normalize(self.observable_field_transformation(links))
                if save_config:
                    configs.append(links_ori.detach().cpu().clone())
                plaq_values.append(float(plaquette_mean_from_field(links_ori).detach().cpu()))
                hamiltonians.append(hamiltonian)
                topological_charges.append(float(topology_from_field(links_ori).detach().cpu()))

        return configs, plaq_values, acceptance_count / n_iterations, topological_charges, hamiltonians
