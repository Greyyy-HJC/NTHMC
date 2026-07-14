"""Optimized neural field transformation for 2D U(1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nthmc.core.training import fixed_batches, local_batch, load_jax_npz, save_jax_npz, unwrap_model
from nthmc.u1.training_models import choose_model
from nthmc.u1.training_observables import (
    format_beta,
    get_field_mask,
    get_plaq_mask,
    get_rect_mask,
    plaq_from_field_batch,
    rect_from_field_batch,
)


class FieldTransformation:
    """Optimized field transformation with optional torch.compile."""

    def __init__(
        self,
        lattice_size: int,
        *,
        device: str = "cpu",
        n_subsets: int = 8,
        if_check_jac: bool = False,
        num_workers: int = 0,
        identity_init: bool = True,
        model_tag: str = "base",
        save_tag: str | None = None,
        model_dir: str | Path = "artifacts/models",
        plot_dir: str | Path = "plots",
        dump_dir: str | Path = "dumps",
        hyperparams: dict[str, float] | None = None,
        fabric=None,
        backend: str = "eager",
        compile_enabled: bool = False,
    ) -> None:
        self.lattice_size = lattice_size
        self.device = torch.device(device)
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.num_workers = num_workers
        self.model_tag = model_tag
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.plot_dir = Path(plot_dir)
        self.dump_dir = Path(dump_dir)
        self.train_beta: float | None = None
        self.fabric = fabric
        self.print = fabric.print if fabric is not None else print
        self.backward = fabric.backward if fabric is not None else torch.autograd.backward
        self.backend = backend
        self.compile_enabled = compile_enabled

        self.hyperparams = {
            "init_std": 0.001,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "factor": 0.5,
            "patience": 5,
            "max_grad_norm": 10.0,
            "inverse_max_iters": 200,
            "inverse_tol": 1e-6,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        model_cls = choose_model(model_tag)
        raw_models = nn.ModuleList([model_cls().to(self.device) for _ in range(n_subsets)])

        if identity_init:
            for model in raw_models:
                nn.init.normal_(model.conv_input.weight, mean=0.0, std=self.hyperparams["init_std"])
                nn.init.normal_(model.conv_input.bias, mean=0.0, std=self.hyperparams["init_std"])
                nn.init.normal_(model.conv_output.weight, mean=0.0, std=self.hyperparams["init_std"])
                nn.init.normal_(model.conv_output.bias, mean=0.0, std=self.hyperparams["init_std"])
                model.out_scale.scale.data.zero_()

        raw_optimizer = torch.optim.AdamW(
            raw_models.parameters(), lr=self.hyperparams["lr"], weight_decay=self.hyperparams["weight_decay"]
        )
        self.models = [self.fabric.setup_module(model) for model in raw_models] if self.fabric is not None else list(raw_models)
        optimizer = self.fabric.setup_optimizers(raw_optimizer) if self.fabric is not None else raw_optimizer
        self.optimizers = [optimizer]
        self.schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hyperparams["factor"],
                patience=int(self.hyperparams["patience"]),
            )
            for optimizer in self.optimizers
        ]

        self._init_compiled_functions()

    def _clip_gradients(self) -> None:
        max_norm = float(self.hyperparams.get("max_grad_norm", 0.0))
        if max_norm <= 0:
            return
        params: list[torch.nn.Parameter] = []
        for model in self.models:
            params.extend(p for p in model.parameters() if p.requires_grad)
        if not params:
            return
        torch.nn.utils.clip_grad_norm_(params, max_norm)

    def _init_compiled_functions(self) -> None:
        """Prepare compiled callables and fall back to regular methods if unavailable."""
        self.ft_phase_compiled = self.ft_phase
        self.forward_compiled = self.forward
        self.inverse_compiled = self.inverse
        self.compute_jac_logdet_compiled = self.compute_jac_logdet
        self.compute_action_compiled = self.compute_action

        if not self.compile_enabled:
            self.print("torch.compile disabled; using standard functions")
            return
        if not hasattr(torch, "compile"):
            self.print("torch.compile not available; using standard functions")
            return

        compile_options = {"backend": self.backend, "fullgraph": False, "dynamic": True}
        try:
            self.ft_phase_compiled = torch.compile(self.ft_phase, **compile_options)
            self.forward_compiled = torch.compile(self._forward_using_compiled_phase, **compile_options)
            self.inverse_compiled = torch.compile(self._inverse_using_compiled_phase, **compile_options)
            self.compute_jac_logdet_compiled = torch.compile(self.compute_jac_logdet, **compile_options)
            self.compute_action_compiled = torch.compile(self.compute_action, **compile_options)
            self.print(f"Initialized torch.compile wrappers with backend={self.backend!r}")
        except Exception as exc:
            self.print(f"Warning: torch.compile initialization failed: {exc}")
            self.print("Falling back to standard functions")

    def freeze_models_for_eval(self) -> None:
        """Freeze model parameters before evaluation-only compiled execution."""
        for model in self.models:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

    def enable_eval_compile(self, *, backend: str = "inductor") -> None:
        """Enable torch.compile after models are loaded and frozen for evaluation."""
        if self.if_check_jac:
            raise RuntimeError("Evaluation compile is not supported with if_check_jac=True")
        self.backend = backend
        self.compile_enabled = True
        self._init_compiled_functions()

    def compute_k0_k1(
        self,
        theta: torch.Tensor,
        index: int,
        plaq: torch.Tensor,
        rect: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return plaquette and rectangle coefficients for one active link subset."""
        batch_size = theta.shape[0]
        plaq_mask = get_plaq_mask(index, batch_size, self.lattice_size, self.device)
        rect_mask = get_rect_mask(index, batch_size, self.lattice_size, self.device)

        plaq_masked = plaq * plaq_mask
        rect_masked = rect * rect_mask
        plaq_features = torch.stack([torch.sin(plaq_masked), torch.cos(plaq_masked)], dim=1)
        rect_features = torch.cat([torch.sin(rect_masked), torch.cos(rect_masked)], dim=1)

        output = self.models[index](plaq_features, rect_features)
        if not isinstance(output, tuple):
            raise ValueError("field_transform expects models to return (plaq_coeffs, rect_coeffs)")
        k0, k1 = output
        if k0.shape[1] != 8 or k1.shape[1] != 16:
            raise ValueError("field_transform expects 8 plaquette and 16 rectangle coefficient channels")
        return k0, k1

    def _plaq_angle_stack(self, plaq: torch.Tensor) -> torch.Tensor:
        """Stack the four plaquette angles touching each active link."""
        return torch.stack(
            [
                plaq,
                torch.roll(plaq, shifts=1, dims=2),
                plaq,
                torch.roll(plaq, shifts=1, dims=1),
            ],
            dim=1,
        )

    def _rect_angle_stack(self, rect: torch.Tensor) -> torch.Tensor:
        """Stack the eight rectangle angles touching each active link."""
        rect0 = rect[:, 0]
        rect1 = rect[:, 1]
        return torch.stack(
            [
                torch.roll(rect0, shifts=1, dims=1),
                torch.roll(rect0, shifts=(1, 1), dims=(1, 2)),
                rect0,
                torch.roll(rect0, shifts=1, dims=2),
                torch.roll(rect1, shifts=1, dims=2),
                torch.roll(rect1, shifts=(1, 1), dims=(1, 2)),
                rect1,
                torch.roll(rect1, shifts=1, dims=1),
            ],
            dim=1,
        )

    def _plaq_phase_shift(self, k0: torch.Tensor, plaq_angles: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        sin_plaq_signs = torch.tensor([-1, 1, 1, -1], device=self.device, dtype=theta.dtype)
        cos_plaq_signs = -sin_plaq_signs
        plaq_stack = torch.cat(
            [
                torch.sin(plaq_angles) * sin_plaq_signs.view(1, 4, 1, 1),
                torch.cos(plaq_angles) * cos_plaq_signs.view(1, 4, 1, 1),
            ],
            dim=1,
        )
        temp = k0 * plaq_stack
        return torch.stack(
            [
                temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5],
                temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7],
            ],
            dim=1,
        )

    def _plaq_jac_shift(self, k0: torch.Tensor, plaq_angles: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        temp = k0 * torch.cat([-torch.cos(plaq_angles), -torch.sin(plaq_angles)], dim=1)
        return torch.stack(
            [
                temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5],
                temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7],
            ],
            dim=1,
        )

    def _rect_phase_shift(self, k1: torch.Tensor, rect_angles: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1], device=self.device, dtype=theta.dtype)
        rect_stack = torch.cat(
            [
                torch.sin(rect_angles) * signs.view(1, 8, 1, 1),
                torch.cos(rect_angles) * (-signs).view(1, 8, 1, 1),
            ],
            dim=1,
        )
        temp = k1 * rect_stack
        return torch.stack(
            [
                temp[:, 0:4].sum(dim=1) + temp[:, 8:12].sum(dim=1),
                temp[:, 4:8].sum(dim=1) + temp[:, 12:16].sum(dim=1),
            ],
            dim=1,
        )

    def _rect_jac_shift(self, k1: torch.Tensor, rect_angles: torch.Tensor) -> torch.Tensor:
        temp = k1 * torch.cat([-torch.cos(rect_angles), -torch.sin(rect_angles)], dim=1)
        return torch.stack(
            [
                temp[:, 0:4].sum(dim=1) + temp[:, 8:12].sum(dim=1),
                temp[:, 4:8].sum(dim=1) + temp[:, 12:16].sum(dim=1),
            ],
            dim=1,
        )

    def ft_phase(self, theta: torch.Tensor, index: int) -> torch.Tensor:
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)

        plaq_angles = self._plaq_angle_stack(plaq)
        rect_angles = self._rect_angle_stack(rect)
        k0, k1 = self.compute_k0_k1(theta, index, plaq, rect)
        ft_phase_plaq = self._plaq_phase_shift(k0, plaq_angles, theta)
        ft_phase_rect = self._rect_phase_shift(k1, rect_angles, theta)

        field_mask = get_field_mask(index, batch_size, self.lattice_size, self.device)
        return (ft_phase_plaq + ft_phase_rect) * field_mask

    def _forward_using_compiled_phase(self, theta: torch.Tensor) -> torch.Tensor:
        theta_curr = theta.clone()
        for index in range(self.n_subsets):
            theta_curr = theta_curr + self.ft_phase_compiled(theta_curr, index)
        return theta_curr

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        theta_curr = theta.clone()
        for index in range(self.n_subsets):
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)
        return theta_curr

    def field_transformation(self, theta: torch.Tensor) -> torch.Tensor:
        return self.forward(theta.unsqueeze(0)).squeeze(0)

    def field_transformation_compiled(self, theta: torch.Tensor) -> torch.Tensor:
        return self.forward_compiled(theta.unsqueeze(0)).squeeze(0)

    def _inverse_using_compiled_phase(
        self,
        theta: torch.Tensor,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        return self.inverse(theta, max_iter=max_iter, tol=tol)

    def inverse(
        self,
        theta: torch.Tensor,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
        sample_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
        max_iter = int(self.hyperparams["inverse_max_iters"] if max_iter is None else max_iter)
        tol = float(self.hyperparams["inverse_tol"] if tol is None else tol)
        theta_curr = theta.clone()
        valid = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device) if sample_mask is None else sample_mask.bool()
        subset_final_diffs: list[torch.Tensor] = []
        subset_iterations: list[torch.Tensor] = []
        n_not_converged = 0
        for index in reversed(range(self.n_subsets)):
            theta_iter = theta_curr.clone()
            active = valid.clone()
            diff = torch.full((theta.shape[0],), float("inf"), device=theta.device, dtype=theta.dtype)
            iterations = torch.zeros(theta.shape[0], device=theta.device, dtype=torch.int32)
            for _ in range(max_iter):
                theta_next = theta_curr - self.ft_phase(theta_iter, index)
                flat_iter = theta_iter.reshape(theta.shape[0], -1)
                flat_update = (theta_next - theta_iter).reshape(theta.shape[0], -1)
                next_diff = torch.linalg.vector_norm(flat_update, dim=1) / torch.clamp(
                    torch.linalg.vector_norm(flat_iter, dim=1), min=1e-12
                )
                broadcast_active = active.reshape((-1,) + (1,) * (theta.ndim - 1))
                theta_iter = torch.where(broadcast_active, theta_next, theta_iter)
                diff = torch.where(active, next_diff, diff)
                iterations = iterations + active.to(iterations.dtype)
                active = active & ((next_diff >= tol) | ~torch.isfinite(next_diff))
                if not bool(active.any().item()):
                    break
            subset_final_diffs.append(diff[valid].detach())
            subset_iterations.append(iterations[valid].detach())
            n_not_converged += int(active.sum().detach().cpu())
            theta_curr = theta_iter
        if return_diagnostics:
            final_diffs = torch.cat(subset_final_diffs) if subset_final_diffs else torch.zeros(1, device=theta.device)
            iteration_counts = torch.cat(subset_iterations) if subset_iterations else torch.zeros(1, device=theta.device)
            if final_diffs.numel() == 0:
                final_diffs = torch.zeros(1, device=theta.device, dtype=theta.dtype)
                iteration_counts = torch.zeros(1, device=theta.device, dtype=torch.int32)
            diag: dict[str, float | int] = {
                "max_final_diff": float(final_diffs.max().cpu()),
                "mean_final_diff": float(final_diffs.mean().cpu()),
                "mean_iterations": float(iteration_counts.float().mean().cpu()),
                "max_iterations": int(iteration_counts.max().cpu()),
                "n_not_converged": n_not_converged,
            }
            return theta_curr, diag
        return theta_curr

    def inverse_field_transformation(self, theta: torch.Tensor) -> torch.Tensor:
        return self.inverse(theta.unsqueeze(0)).squeeze(0)

    def inverse_field_transformation_compiled(self, theta: torch.Tensor) -> torch.Tensor:
        return self.inverse_compiled(theta.unsqueeze(0)).squeeze(0)

    def compute_jac_logdet(self, theta: torch.Tensor) -> torch.Tensor:
        batch_size = theta.shape[0]
        log_det = torch.zeros(batch_size, device=self.device)
        theta_curr = theta.clone()

        for index in range(self.n_subsets):
            field_mask = get_field_mask(index, batch_size, self.lattice_size, self.device)
            plaq = plaq_from_field_batch(theta_curr)
            rect = rect_from_field_batch(theta_curr)

            plaq_angles = self._plaq_angle_stack(plaq)
            rect_angles = self._rect_angle_stack(rect)
            k0, k1 = self.compute_k0_k1(theta_curr, index, plaq, rect)
            plaq_jac_shift = self._plaq_jac_shift(k0, plaq_angles, theta_curr) * field_mask
            rect_jac_shift = self._rect_jac_shift(k1, rect_angles) * field_mask

            log_det = log_det + torch.log(1 + plaq_jac_shift + rect_jac_shift).sum(dim=(1, 2, 3))
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)

        return log_det

    def compute_jac_logdet_autograd(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute the full Jacobian log determinant for the first batch item."""
        theta_single = theta[0].unsqueeze(0)
        jacobian = torch.autograd.functional.jacobian(
            self.forward_compiled,
            theta_single,
            create_graph=True,
        )
        jacobian_2d = jacobian.reshape(theta_single.shape[0], theta_single.numel(), theta_single.numel())
        _, logabsdet = torch.linalg.slogdet(jacobian_2d)
        return logabsdet

    def compute_action(self, theta: torch.Tensor, beta: float) -> torch.Tensor:
        plaq = plaq_from_field_batch(theta)
        return -beta * torch.sum(torch.cos(plaq), dim=(1, 2))

    def compute_force(self, theta: torch.Tensor, beta: float, *, transformed: bool = False) -> torch.Tensor:
        if not theta.requires_grad:
            theta = theta.clone().requires_grad_(True)

        if transformed:
            theta_ori = self.forward_compiled(theta)
            action = self.compute_action_compiled(theta_ori, beta)
            jac_logdet = self.compute_jac_logdet_compiled(theta)
            if self.if_check_jac:
                jac_logdet_autograd = self.compute_jac_logdet_autograd(theta)
                abs_diff = torch.abs(jac_logdet_autograd[0] - jac_logdet[0])
                denominator = torch.clamp(torch.abs(jac_logdet[0]), min=1e-12)
                relative_diff = abs_diff / denominator
                is_close = torch.isclose(jac_logdet_autograd[0], jac_logdet[0], rtol=1e-4, atol=1e-6)
                if not is_close.item():
                    self.print(
                        "\nWarning: Jacobian log determinant difference "
                        f"abs={abs_diff.item():.2e}, rel={relative_diff.item():.2e}"
                    )
                    self.print(">>> Jacobian is not correct!")
                else:
                    self.print(
                        "\nJacobian log det "
                        f"(manual): {jac_logdet[0].item():.2e}, "
                        f"(autograd): {jac_logdet_autograd[0].item():.2e}"
                    )
                    self.print(">>> Jacobian is all good!")
            total_action = action - jac_logdet
        else:
            total_action = self.compute_action_compiled(theta, beta)

        return torch.autograd.grad(total_action.sum(), theta, create_graph=True)[0]

    def loss_fn(self, theta_ori: torch.Tensor, sample_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        valid = torch.ones(theta_ori.shape[0], dtype=torch.bool, device=theta_ori.device) if sample_mask is None else sample_mask.bool()
        theta_new = self.inverse(theta_ori, sample_mask=valid)
        force_new = self.compute_force(theta_new, self.train_beta, transformed=True)
        volume = self.lattice_size * self.lattice_size
        force_flat = force_new.reshape(force_new.shape[0], -1)
        loss_per_config = (
            torch.linalg.vector_norm(force_flat, ord=2, dim=1) / (volume**0.5)
            + torch.linalg.vector_norm(force_flat, ord=4, dim=1) / (volume**0.25)
            + torch.linalg.vector_norm(force_flat, ord=6, dim=1) / (volume ** (1 / 6))
            + torch.linalg.vector_norm(force_flat, ord=8, dim=1) / (volume ** (1 / 8))
        )
        return (loss_per_config * valid.to(loss_per_config.dtype)).sum() / valid.sum().clamp_min(1)

    def _maybe_log_inverse_diagnostics(
        self,
        test_data: torch.Tensor,
        batch_size: int,
        epoch_display: int,
        n_epochs: int,
    ) -> None:
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        n = min(8, int(test_data.shape[0]), int(batch_size))
        if n <= 0:
            return
        probe = test_data[:n].to(self.device)
        with torch.no_grad():
            inv, diag = self.inverse(probe, return_diagnostics=True)
            recon = self.forward(inv)
            rt_err = (recon - probe).abs().mean().item()
        self.print(
            f"Epoch {epoch_display}/{n_epochs} inverse_diag: "
            f"max_final_diff={diag['max_final_diff']:.2e} "
            f"mean_final_diff={diag['mean_final_diff']:.2e} "
            f"mean_iterations={diag['mean_iterations']:.2f} "
            f"max_iterations={diag['max_iterations']} "
            f"n_subsets_not_converged={diag['n_not_converged']} "
            f"round_trip_mean_abs_err={rt_err:.2e}"
        )

    def train_step(self, theta_ori: torch.Tensor, sample_mask: torch.Tensor) -> tuple[float, int]:
        theta_ori = theta_ori.to(self.device)
        sample_mask = sample_mask.to(self.device)
        loss = self.loss_fn(theta_ori, sample_mask)
        local_count = sample_mask.sum().to(loss.dtype)
        global_count = self.fabric.all_reduce(local_count.detach(), reduce_op="sum") if self.fabric is not None else local_count
        world_size = int(self.fabric.world_size) if self.fabric is not None else 1
        backward_loss = loss * local_count * world_size / global_count.clamp_min(1)
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.backward(backward_loss)
        self._clip_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        return float(loss.detach().cpu()), int(local_count.detach().cpu())

    def evaluate_step(self, theta_ori: torch.Tensor, sample_mask: torch.Tensor) -> tuple[float, int]:
        theta_ori = theta_ori.to(self.device).requires_grad_(True)
        sample_mask = sample_mask.to(self.device)
        loss = self.loss_fn(theta_ori, sample_mask)
        return float(loss.detach().cpu()), int(sample_mask.sum().cpu())

    def train(self, train_data: torch.Tensor, test_data: torch.Tensor, train_beta: float, *, n_epochs: int, batch_size: int) -> None:
        self.train_beta = train_beta
        rank = int(self.fabric.global_rank) if self.fabric is not None else 0
        world_size = int(self.fabric.world_size) if self.fabric is not None else 1
        if batch_size % world_size:
            raise ValueError(f"global batch size {batch_size} must be divisible by world_size={world_size}")

        train_losses: list[float] = []
        test_losses: list[float] = []
        best_loss = float("inf")

        progress_disabled = self.fabric is not None and self.fabric.global_rank != 0
        for epoch in tqdm(range(n_epochs), desc="Training epochs", disable=progress_disabled):
            self._set_models_mode(True)
            epoch_losses = []
            for batch, mask in fixed_batches(train_data, batch_size, shuffle=True, seed=epoch):
                local, local_mask = local_batch(batch, mask, rank=rank, world_size=world_size)
                epoch_losses.append(self.train_step(local, local_mask))
            train_loss = self._global_weighted_epoch_loss(epoch_losses)
            train_losses.append(train_loss)

            self._set_models_mode(False)
            test_epoch_losses = []
            for batch, mask in fixed_batches(test_data, batch_size, shuffle=False, seed=0):
                local, local_mask = local_batch(batch, mask, rank=rank, world_size=world_size)
                test_epoch_losses.append(self.evaluate_step(local, local_mask))
            test_loss = self._global_weighted_epoch_loss(test_epoch_losses)
            test_losses.append(test_loss)

            self.print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Train Loss: {train_loss:.6f} - Test Loss: {test_loss:.6f}"
            )
            self._maybe_log_inverse_diagnostics(test_data, batch_size, epoch + 1, n_epochs)
            if test_loss < best_loss:
                self.save_best_model(epoch, test_loss)
                best_loss = test_loss
            for scheduler in self.schedulers:
                scheduler.step(test_loss)

        self.plot_training_history(train_losses, test_losses)
        if self.fabric is not None:
            self.fabric.barrier()
        self.load_best_model(train_beta)

    def _set_models_mode(self, is_train: bool) -> None:
        for model in self.models:
            model.train() if is_train else model.eval()

    @staticmethod
    def _weighted_epoch_loss(losses_and_counts: list[tuple[float, int]]) -> float:
        total_count = sum(count for _, count in losses_and_counts)
        if total_count == 0:
            return float("nan")
        return float(sum(loss * count for loss, count in losses_and_counts) / total_count)

    def _global_weighted_epoch_loss(self, losses_and_counts: list[tuple[float, int]]) -> float:
        totals = torch.tensor(
            [sum(loss * count for loss, count in losses_and_counts), sum(count for _, count in losses_and_counts)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.fabric is not None:
            totals = self.fabric.all_reduce(totals, reduce_op="sum")
        return float((totals[0] / totals[1].clamp_min(1)).item())

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.pt"

    def jax_checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.npz"

    def save_best_model(self, epoch: int, loss: float) -> None:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {"epoch": epoch, "loss": loss, "hyperparams": self.hyperparams}
        for index, model in enumerate(self.models):
            save_dict[f"model_state_dict_{index}"] = unwrap_model(model, self.fabric).state_dict()
        for index, optimizer in enumerate(self.optimizers):
            save_dict[f"optimizer_state_dict_{index}"] = optimizer.state_dict()
        for index, scheduler in enumerate(self.schedulers):
            save_dict[f"scheduler_state_dict_{index}"] = scheduler.state_dict()
        torch.save(save_dict, self.checkpoint_path(self.train_beta))
        save_jax_npz(
            self.jax_checkpoint_path(self.train_beta),
            self.models,
            {
                "system": "2du1",
                "transform": "neural_u1_torch_training",
                "model_tag": self.model_tag,
                "n_subsets": self.n_subsets,
                "lattice_size": self.lattice_size,
                "train_beta": float(self.train_beta),
                "epoch": int(epoch),
                "loss": float(loss),
                "hyperparams": self.hyperparams,
            },
            self.fabric,
        )

    def load_best_model(self, train_beta: float) -> None:
        path = self.checkpoint_path(train_beta)
        if not path.exists():
            metadata = load_jax_npz(self.jax_checkpoint_path(train_beta), self.models, self.fabric)
            self.print(
                f"Loaded JAX NPZ weights from epoch {metadata.get('epoch')} with loss {metadata.get('loss')}; "
                "optimizer and scheduler were reinitialized"
            )
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        for index, model in enumerate(self.models):
            state_dict = checkpoint[f"model_state_dict_{index}"]
            unwrap_model(model, self.fabric).load_state_dict(state_dict)
        for index, optimizer in enumerate(self.optimizers):
            key = f"optimizer_state_dict_{index}"
            if key in checkpoint:
                optimizer.load_state_dict(checkpoint[key])
        for index, scheduler in enumerate(self.schedulers):
            key = f"scheduler_state_dict_{index}"
            if key in checkpoint:
                scheduler.load_state_dict(checkpoint[key])
        self.print(f"Loaded best model from epoch {checkpoint['epoch'] + 1} with loss {checkpoint['loss']:.6f}")

    def plot_training_history(self, train_losses: list[float], test_losses: list[float]) -> None:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        beta_tag = format_beta(self.train_beta)

        epochs_axis = np.arange(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_axis, train_losses, label="Train")
        plt.plot(epochs_axis, test_losses, label="Test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"cnn_loss_train_beta{beta_tag}_{self.save_tag}.pdf", transparent=True)
        plt.close()

        np.savetxt(self.dump_dir / f"train_loss_train_beta{beta_tag}_{self.save_tag}.csv", train_losses, fmt="%.6e")
        np.savetxt(self.dump_dir / f"test_loss_train_beta{beta_tag}_{self.save_tag}.csv", test_losses, fmt="%.6e")
