"""Field-transformed HMC for 2D U(1)."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from nthmc.u1.u1_observables import plaq_from_field, plaq_mean_from_field, regularize, topo_from_field


class HMCU1FT:
    """HMC sampler in transformed coordinates."""

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
        return torch.zeros([2, self.lattice_size, self.lattice_size], device=self.device)

    def original_action(self, theta: torch.Tensor) -> torch.Tensor:
        theta_p = regularize(plaq_from_field(theta))
        action_value = -self.beta * torch.sum(torch.cos(theta_p))
        assert action_value.dim() == 0
        return action_value

    def new_action(self, theta_new: torch.Tensor, *, for_force: bool = False) -> torch.Tensor:
        field_transformation = self.force_field_transformation if for_force else self.field_transformation
        compute_jac_logdet = self.force_compute_jac_logdet if for_force else self.compute_jac_logdet
        theta_ori = field_transformation(theta_new)
        original_action_value = self.original_action(theta_ori)
        jacobian_log_det = compute_jac_logdet(theta_new.unsqueeze(0)).squeeze(0)
        action_value = original_action_value - jacobian_log_det
        assert action_value.dim() == 0
        return action_value

    def new_force(self, theta_new: torch.Tensor) -> torch.Tensor:
        theta_copy = theta_new.detach().clone().requires_grad_(True)
        action_value = self.new_action(theta_copy, for_force=True)
        return torch.autograd.grad(action_value, theta_copy)[0].detach()

    def omelyan(self, theta: torch.Tensor, pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lam = 0.1931833
        theta_next = theta
        pi_next = pi - lam * self.dt * self.new_force(theta_next)
        for step_index in range(self.n_steps):
            theta_next = theta_next + 0.5 * self.dt * pi_next
            pi_next = pi_next - (1 - 2 * lam) * self.dt * self.new_force(theta_next)
            theta_next = theta_next + 0.5 * self.dt * pi_next
            if step_index != self.n_steps - 1:
                pi_next = pi_next - 2 * lam * self.dt * self.new_force(theta_next)
        pi_next = pi_next - lam * self.dt * self.new_force(theta_next)
        return regularize(theta_next), pi_next

    def metropolis_step(self, theta: torch.Tensor) -> tuple[torch.Tensor, bool, float]:
        pi = torch.randn_like(theta, device=self.device)
        h_old = self.new_action(theta) + 0.5 * torch.sum(pi**2)
        new_theta, new_pi = self.omelyan(theta.clone(), pi.clone())
        h_new = self.new_action(new_theta) + 0.5 * torch.sum(new_pi**2)
        accept_prob = torch.exp(-(h_new - h_old))
        if torch.rand([], device=self.device) < accept_prob:
            return new_theta, True, float(h_new.detach().cpu())
        return theta, False, float(h_old.detach().cpu())

    def tune_step_size(
        self,
        *,
        n_tune_steps: int = 1000,
        target_rate: float = 0.75,
        target_tolerance: float = 0.15,
        max_attempts: int = 10,
        theta: torch.Tensor | None = None,
    ) -> None:
        theta = self.initialize() if theta is None else theta.clone()
        step_min = 1e-6
        step_max = 1.0
        best_dt = self.dt
        best_rate_diff = float("inf")
        current_rate = 0.0

        for attempt in range(max_attempts):
            acceptance_count = 0
            for _ in tqdm(range(n_tune_steps), desc=f"Tuning FT step size ({attempt + 1}/{max_attempts})"):
                theta, accepted, _ = self.metropolis_step(theta)
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
        theta = self.initialize()
        if self.tune_step_size_enabled:
            print(">>> Initial FT thermalization for step-size tuning")
            initial_plaq = []
            for _ in tqdm(range(self.n_thermalization_steps), desc="Initial FT thermalization"):
                theta, _, _ = self.metropolis_step(theta)
                with torch.no_grad():
                    theta_ori = regularize(self.observable_field_transformation(theta))
                initial_plaq.append(round(float(plaq_mean_from_field(theta_ori).detach().cpu()), 4))
            if len(initial_plaq) >= 10:
                n_complete = (len(initial_plaq) // 10) * 10
                means = np.mean(np.array(initial_plaq[-n_complete:]).reshape(-1, 10), axis=1)
                print(f"Initial FT thermalization plaquette means: {means}")
            self.tune_step_size(n_tune_steps=n_tune_steps, theta=theta)
        else:
            print(f">>> Using FT step size without tuning: {self.dt:.6f}")

        theta = self.initialize()
        plaq_values = []
        acceptance_count = 0
        for _ in tqdm(range(self.n_thermalization_steps), desc="Thermalizing FT-HMC"):
            theta = regularize(theta)
            with torch.no_grad():
                theta_ori = regularize(self.observable_field_transformation(theta))
            plaq_values.append(float(plaq_mean_from_field(theta_ori).detach().cpu()))
            theta, accepted, _ = self.metropolis_step(theta)
            acceptance_count += int(accepted)

        return theta, plaq_values, acceptance_count / self.n_thermalization_steps

    def run(
        self,
        n_iterations: int,
        theta: torch.Tensor,
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
            theta, accepted, hamiltonian = self.metropolis_step(theta)
            acceptance_count += int(accepted)
            if index % store_interval == 0:
                theta = regularize(theta)
                with torch.no_grad():
                    theta_ori = regularize(self.observable_field_transformation(theta))
                if save_config:
                    configs.append(theta_ori.detach().cpu().clone())
                plaq_values.append(float(plaq_mean_from_field(theta_ori).detach().cpu()))
                hamiltonians.append(hamiltonian)
                topological_charges.append(float(topo_from_field(theta_ori).detach().cpu()))

        return configs, plaq_values, acceptance_count / n_iterations, topological_charges, hamiltonians
