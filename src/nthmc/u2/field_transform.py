"""Base neural field transformation for 2D U(2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from nthmc.u2.models import choose_model
from nthmc.u2.u2_observables import (
    action_from_field_batch,
    format_beta,
    get_link_mask,
    get_plaq_mask,
    get_rect_mask,
    plaquette_from_field_batch,
    rectangle_from_field_batch,
    identity_like,
    loop_sin_cos_features,
    quaternion_conj,
    quaternion_mul,
    u2_conj,
    u2_exp,
    u2_log,
    u2_mul,
    u2_normalize,
)

LoopToken = tuple[int, int | tuple[int, int] | None, int | tuple[int, int] | None, bool]


class FieldTransformation:
    """Loop-term U(2) field transformation for FT-HMC."""

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
        self.save_tag = save_tag or "base"
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
            "lr": 0.0001,
            "weight_decay": 1e-5,
            "factor": 0.5,
            "patience": 1,
            "max_grad_norm": 10.0,
            "early_stop_patience": 3,
            "loss_weights": (1.0, 1.0, 1.0, 1.0),
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)

        model_cls = choose_model(model_tag)
        raw_models = nn.ModuleList([model_cls().to(self.device) for _ in range(n_subsets)])
        if identity_init:
            for model in raw_models:
                for param in model.parameters():
                    nn.init.normal_(param, mean=0.0, std=self.hyperparams["init_std"])

        raw_optimizers = [
            torch.optim.AdamW(
                model.parameters(),
                lr=self.hyperparams["lr"],
                weight_decay=self.hyperparams["weight_decay"],
            )
            for model in raw_models
        ]
        self.models = []
        self.optimizers = []
        for model, optimizer in zip(raw_models, raw_optimizers):
            if self.fabric is not None:
                model, optimizer = self.fabric.setup(model, optimizer)
            self.models.append(model)
            self.optimizers.append(optimizer)
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

    def _gradient_norm(self) -> float:
        norm_sq = 0.0
        for model in self.models:
            for param in model.parameters():
                if param.grad is not None:
                    norm_sq += float(torch.sum(param.grad.detach() ** 2).cpu())
        return norm_sq**0.5

    def _init_compiled_functions(self) -> None:
        """Prepare compiled callables and fall back to regular methods if unavailable."""
        self.compute_delta_compiled = self.compute_delta
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
            self.compute_delta_compiled = torch.compile(self.compute_delta, **compile_options)
            self.ft_phase_compiled = torch.compile(self.ft_phase, **compile_options)
            self.forward_compiled = torch.compile(self._forward_using_compiled_phase, **compile_options)
            self.inverse_compiled = torch.compile(self._inverse_using_compiled_delta, **compile_options)
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

    def _masked_loops(self, loops: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, loops, identity_like(loops))

    def _scalar_loop_features(self, loops: torch.Tensor) -> torch.Tensor:
        """Return compact gauge-invariant scalar loop features."""
        loops = u2_normalize(loops)
        phase = loops[..., :1]
        q0 = loops[..., 1:2]
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        trace_sq_factor = 2 * (2 * q0**2 - 1)
        return torch.cat(
            [
                q0 * cos_phase,
                q0 * sin_phase,
                cos_phase,
                sin_phase,
                trace_sq_factor * torch.cos(2 * phase),
                trace_sq_factor * torch.sin(2 * phase),
            ],
            dim=-1,
        )

    def compute_coefficients(
        self,
        links: torch.Tensor,
        index: int,
        plaq: torch.Tensor,
        rect: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return plaquette and rectangle coefficients for one active link subset."""
        batch_size = links.shape[0]
        plaq_mask = get_plaq_mask(index, batch_size, self.lattice_size, self.device)
        rect_mask = get_rect_mask(index, batch_size, self.lattice_size, self.device)

        plaq_features = self._scalar_loop_features(self._masked_loops(plaq, plaq_mask)).permute(0, 3, 1, 2)
        rect_features = self._scalar_loop_features(self._masked_loops(rect, rect_mask))
        rect_features = rect_features.permute(0, 1, 4, 2, 3).reshape(batch_size, 12, self.lattice_size, self.lattice_size)

        output = self.models[index](plaq_features, rect_features)
        if not isinstance(output, tuple):
            raise ValueError("field_transform expects models to return (plaq_coeffs, rect_coeffs)")
        plaq_coeffs, rect_coeffs = output
        if plaq_coeffs.shape[1] != 16 or rect_coeffs.shape[1] != 32:
            raise ValueError("field_transform expects 16 plaquette and 32 rectangle coefficient channels")
        return plaq_coeffs, rect_coeffs

    def _loop_product(self, parts: list[torch.Tensor]) -> torch.Tensor:
        value = parts[0]
        for part in parts[1:]:
            value = u2_mul(value, part)
        return value

    def _loop_product_with_tangent(
        self,
        parts: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, value_tangent = parts[0]
        for next_value, next_tangent in parts[1:]:
            value, value_tangent = self._mul_with_tangent(value, value_tangent, next_value, next_tangent)
        return value, value_tangent

    def _resolve_loop_token(
        self,
        link0: torch.Tensor,
        link1: torch.Tensor,
        token: LoopToken,
    ) -> torch.Tensor:
        direction, shifts, dims, is_inverse = token
        value = link0 if direction == 0 else link1
        if shifts is not None and dims is not None:
            value = torch.roll(value, shifts=shifts, dims=dims)
        return u2_conj(value) if is_inverse else value

    def _resolve_loop_token_with_tangent(
        self,
        link0: torch.Tensor,
        link1: torch.Tensor,
        tangent0: torch.Tensor,
        tangent1: torch.Tensor,
        token: LoopToken,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        direction, shifts, dims, is_inverse = token
        value = link0 if direction == 0 else link1
        tangent = tangent0 if direction == 0 else tangent1
        if shifts is not None and dims is not None:
            value = torch.roll(value, shifts=shifts, dims=dims)
            tangent = torch.roll(tangent, shifts=shifts, dims=dims)
        return self._conj_with_tangent(value, tangent) if is_inverse else (value, tangent)

    def _stack_loop_specs(self, links: torch.Tensor, specs: list[list[LoopToken]]) -> torch.Tensor:
        links = u2_normalize(links)
        link0, link1 = links[:, 0], links[:, 1]
        loops = [
            self._loop_product([self._resolve_loop_token(link0, link1, token) for token in loop_spec])
            for loop_spec in specs
        ]
        return torch.stack(loops, dim=1)

    def _stack_loop_specs_with_tangent(
        self,
        links: torch.Tensor,
        tangent: torch.Tensor,
        specs: list[list[LoopToken]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        links = u2_normalize(links)
        link0, link1 = links[:, 0], links[:, 1]
        tangent0, tangent1 = tangent[:, 0], tangent[:, 1]
        loop_values = []
        loop_tangents = []
        for loop_spec in specs:
            value, value_tangent = self._loop_product_with_tangent(
                [
                    self._resolve_loop_token_with_tangent(link0, link1, tangent0, tangent1, token)
                    for token in loop_spec
                ]
            )
            loop_values.append(value)
            loop_tangents.append(value_tangent)
        return torch.stack(loop_values, dim=1), torch.stack(loop_tangents, dim=1)

    def _plaq_loop_specs(self) -> list[list[LoopToken]]:
        """Return plaquette loops touching active links, all based at the active site x."""
        return [
            [(0, None, None, False), (1, -1, 1, False), (0, -1, 2, True), (1, None, None, True)],
            [(1, 1, 2, True), (0, 1, 2, False), (1, (-1, 1), (1, 2), False), (0, None, None, True)],
            [(0, None, None, False), (1, -1, 1, False), (0, -1, 2, True), (1, None, None, True)],
            [(1, None, None, False), (0, (1, -1), (1, 2), True), (1, 1, 1, True), (0, 1, 1, False)],
        ]

    def _rect_loop_specs(self) -> list[list[LoopToken]]:
        """Return rectangle loops touching active links, all based at the active site x."""
        return [
            [
                (0, None, None, False),
                (1, -1, 1, False),
                (0, -1, 2, True),
                (0, (1, -1), (1, 2), True),
                (1, 1, 1, True),
                (0, 1, 1, False),
            ],
            [
                (0, 1, 1, True),
                (1, (1, 1), (1, 2), True),
                (0, (1, 1), (1, 2), False),
                (0, 1, 2, False),
                (1, (-1, 1), (1, 2), False),
                (0, None, None, True),
            ],
            [
                (0, None, None, False),
                (0, -1, 1, False),
                (1, -2, 1, False),
                (0, (-1, -1), (1, 2), True),
                (0, -1, 2, True),
                (1, None, None, True),
            ],
            [
                (1, 1, 2, True),
                (0, 1, 2, False),
                (0, (-1, 1), (1, 2), False),
                (1, (-2, 1), (1, 2), False),
                (0, -1, 1, True),
                (0, None, None, True),
            ],
            [
                (1, 1, 2, True),
                (0, 1, 2, False),
                (1, (-1, 1), (1, 2), False),
                (1, -1, 1, False),
                (0, -1, 2, True),
                (1, None, None, True),
            ],
            [
                (1, None, None, False),
                (0, (1, -1), (1, 2), True),
                (1, 1, 1, True),
                (1, (1, 1), (1, 2), True),
                (0, (1, 1), (1, 2), False),
                (1, 1, 2, False),
            ],
            [
                (0, None, None, False),
                (1, -1, 1, False),
                (1, (-1, -1), (1, 2), False),
                (0, -2, 2, True),
                (1, -1, 2, True),
                (1, None, None, True),
            ],
            [
                (1, None, None, False),
                (1, -1, 2, False),
                (0, (1, -2), (1, 2), True),
                (1, (1, -1), (1, 2), True),
                (1, 1, 1, True),
                (0, 1, 1, False),
            ],
        ]

    def _plaq_loop_stack(self, links: torch.Tensor) -> torch.Tensor:
        """Stack based plaquette loops touching each active link."""
        return self._stack_loop_specs(links, self._plaq_loop_specs())

    def _rect_loop_stack(self, links: torch.Tensor) -> torch.Tensor:
        """Stack based rectangle loops touching each active link."""
        return self._stack_loop_specs(links, self._rect_loop_specs())

    def _plaq_loop_stack_with_tangent(self, links: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._stack_loop_specs_with_tangent(links, tangent, self._plaq_loop_specs())

    def _rect_loop_stack_with_tangent(self, links: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._stack_loop_specs_with_tangent(links, tangent, self._rect_loop_specs())

    def _loop_delta(self, coeffs: torch.Tensor, loops: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
        """Return per-loop U(2) algebra contributions."""
        batch_size, n_loops = loops.shape[:2]
        features = loop_sin_cos_features(loops)
        coeffs = coeffs.reshape(batch_size, n_loops, 4, self.lattice_size, self.lattice_size)
        coeffs = coeffs.permute(0, 1, 3, 4, 2)
        signs = signs.to(device=self.device, dtype=loops.dtype).view(1, n_loops, 1, 1, 1)

        phase_delta = coeffs[..., 0:1] * features[..., 0:1] * signs
        phase_delta = phase_delta + coeffs[..., 2:3] * features[..., 4:5]
        traceless_delta = coeffs[..., 1:2] * features[..., 1:4] * signs
        traceless_delta = traceless_delta + coeffs[..., 3:4] * features[..., 5:8]
        return torch.cat([phase_delta, traceless_delta], dim=-1)

    def _plaq_delta(self, k0: torch.Tensor, plaq_loops: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor([-1, 1, 1, -1], device=self.device, dtype=plaq_loops.dtype)
        per_loop = self._loop_delta(k0, plaq_loops, signs)
        return torch.stack([per_loop[:, 0:2].sum(dim=1), per_loop[:, 2:4].sum(dim=1)], dim=1)

    def _rect_delta(self, k1: torch.Tensor, rect_loops: torch.Tensor) -> torch.Tensor:
        signs = torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1], device=self.device, dtype=rect_loops.dtype)
        per_loop = self._loop_delta(k1, rect_loops, signs)
        return torch.stack([per_loop[:, 0:4].sum(dim=1), per_loop[:, 4:8].sum(dim=1)], dim=1)

    def _identity_tangent_like(self, links: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(4, device=links.device, dtype=links.dtype)
        return identity.expand(*links.shape[:-1], 4, 4).clone()

    def _adjoint_algebra(self, links: torch.Tensor, algebra: torch.Tensor) -> torch.Tensor:
        phase = algebra[..., :1]
        q = links[..., 1:].unsqueeze(-2)
        pure = torch.cat([torch.zeros_like(algebra[..., 1:2]), algebra[..., 1:]], dim=-1)
        rotated = quaternion_mul(quaternion_mul(q, pure), quaternion_conj(q))
        return torch.cat([phase, rotated[..., 1:]], dim=-1)

    def _mul_with_tangent(
        self,
        left: torch.Tensor,
        left_tangent: torch.Tensor,
        right: torch.Tensor,
        right_tangent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return u2_mul(left, right), left_tangent + self._adjoint_algebra(left, right_tangent)

    def _conj_with_tangent(self, links: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inverse = u2_conj(links)
        return inverse, -self._adjoint_algebra(inverse, tangent)

    def _plaquette_with_tangent(self, links: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        links = u2_normalize(links)
        link0, link1 = links[:, 0], links[:, 1]
        tangent0, tangent1 = tangent[:, 0], tangent[:, 1]

        rolled_link1 = torch.roll(link1, shifts=-1, dims=1)
        rolled_tangent1 = torch.roll(tangent1, shifts=-1, dims=1)
        rolled_link0 = torch.roll(link0, shifts=-1, dims=2)
        rolled_tangent0 = torch.roll(tangent0, shifts=-1, dims=2)
        conj_rolled_link0, conj_rolled_tangent0 = self._conj_with_tangent(rolled_link0, rolled_tangent0)
        conj_link1, conj_tangent1 = self._conj_with_tangent(link1, tangent1)

        value, value_tangent = self._mul_with_tangent(link0, tangent0, rolled_link1, rolled_tangent1)
        value, value_tangent = self._mul_with_tangent(value, value_tangent, conj_rolled_link0, conj_rolled_tangent0)
        return self._mul_with_tangent(value, value_tangent, conj_link1, conj_tangent1)

    def _rectangle_with_tangent(self, links: torch.Tensor, tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        links = u2_normalize(links)
        link0, link1 = links[:, 0], links[:, 1]
        tangent0, tangent1 = tangent[:, 0], tangent[:, 1]

        def roll_pair(
            value: torch.Tensor,
            value_tangent: torch.Tensor,
            shifts: int | tuple[int, int],
            dims: int | tuple[int, int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.roll(value, shifts=shifts, dims=dims), torch.roll(value_tangent, shifts=shifts, dims=dims)

        def conj_pair(value: torch.Tensor, value_tangent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self._conj_with_tangent(value, value_tangent)

        rect0_parts = [
            (link0, tangent0),
            roll_pair(link0, tangent0, -1, 1),
            roll_pair(link1, tangent1, -2, 1),
            conj_pair(*roll_pair(link0, tangent0, (-1, -1), (1, 2))),
            conj_pair(*roll_pair(link0, tangent0, -1, 2)),
            conj_pair(link1, tangent1),
        ]
        rect1_parts = [
            (link0, tangent0),
            roll_pair(link1, tangent1, -1, 1),
            roll_pair(link1, tangent1, (-1, -1), (1, 2)),
            conj_pair(*roll_pair(link0, tangent0, -2, 2)),
            conj_pair(*roll_pair(link1, tangent1, -1, 2)),
            conj_pair(link1, tangent1),
        ]

        rect_values = []
        rect_tangents = []
        for parts in (rect0_parts, rect1_parts):
            value, value_tangent = parts[0]
            for next_value, next_tangent in parts[1:]:
                value, value_tangent = self._mul_with_tangent(value, value_tangent, next_value, next_tangent)
            rect_values.append(value)
            rect_tangents.append(value_tangent)
        return torch.stack(rect_values, dim=1), torch.stack(rect_tangents, dim=1)

    def _loop_feature_tangent(self, loops: torch.Tensor, loop_tangent: torch.Tensor) -> torch.Tensor:
        loops = u2_normalize(loops)
        phase = loops[..., :1]
        q = loops[..., 1:]
        q0 = q[..., :1]
        qv = q[..., 1:]
        phase_tangent = loop_tangent[..., :1]
        pure_tangent = torch.cat(
            [torch.zeros_like(loop_tangent[..., 1:2]), loop_tangent[..., 1:]],
            dim=-1,
        )
        q_tangent = quaternion_mul(pure_tangent, q.unsqueeze(-2))
        q0_tangent = q_tangent[..., :1]
        qv_tangent = q_tangent[..., 1:]

        sin_phase = torch.sin(phase).unsqueeze(-2)
        cos_phase = torch.cos(phase).unsqueeze(-2)
        q0 = q0.unsqueeze(-2)
        qv = qv.unsqueeze(-2)

        sin_like_phase = q0_tangent * sin_phase + q0 * cos_phase * phase_tangent
        sin_like_traceless = qv_tangent * cos_phase - qv * sin_phase * phase_tangent
        cos_like_phase = q0_tangent * cos_phase - q0 * sin_phase * phase_tangent
        cos_like_traceless = -qv_tangent * sin_phase - qv * cos_phase * phase_tangent
        return torch.cat([sin_like_phase, sin_like_traceless, cos_like_phase, cos_like_traceless], dim=-1)

    def _loop_delta_jac(
        self,
        coeffs: torch.Tensor,
        loops: torch.Tensor,
        loop_tangent: torch.Tensor,
        signs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_loops = loops.shape[:2]
        feature_tangent = self._loop_feature_tangent(loops, loop_tangent)
        coeffs = coeffs.reshape(batch_size, n_loops, 4, self.lattice_size, self.lattice_size)
        coeffs = coeffs.permute(0, 1, 3, 4, 2).unsqueeze(-2)
        signs = signs.to(device=self.device, dtype=loops.dtype).view(1, n_loops, 1, 1, 1, 1)

        phase_jac = coeffs[..., 0:1] * feature_tangent[..., 0:1] * signs
        phase_jac = phase_jac + coeffs[..., 2:3] * feature_tangent[..., 4:5]
        traceless_jac = coeffs[..., 1:2] * feature_tangent[..., 1:4] * signs
        traceless_jac = traceless_jac + coeffs[..., 3:4] * feature_tangent[..., 5:8]
        return torch.cat([phase_jac, traceless_jac], dim=-1)

    def _plaq_delta_jac(
        self,
        k0: torch.Tensor,
        plaq_loops: torch.Tensor,
        plaq_tangent: torch.Tensor,
    ) -> torch.Tensor:
        signs = torch.tensor([-1, 1, 1, -1], device=self.device, dtype=plaq_loops.dtype)
        per_loop = self._loop_delta_jac(k0, plaq_loops, plaq_tangent, signs)
        return torch.stack([per_loop[:, 0:2].sum(dim=1), per_loop[:, 2:4].sum(dim=1)], dim=1)

    def _rect_delta_jac(
        self,
        k1: torch.Tensor,
        rect_loops: torch.Tensor,
        rect_tangent: torch.Tensor,
    ) -> torch.Tensor:
        signs = torch.tensor([-1, 1, -1, 1, 1, -1, 1, -1], device=self.device, dtype=rect_loops.dtype)
        per_loop = self._loop_delta_jac(k1, rect_loops, rect_tangent, signs)
        return torch.stack([per_loop[:, 0:4].sum(dim=1), per_loop[:, 4:8].sum(dim=1)], dim=1)

    def _su2_exp_derivative(self, algebra: torch.Tensor, algebra_tangent: torch.Tensor) -> torch.Tensor:
        r_sq = torch.sum(algebra**2, dim=-1, keepdim=True)
        r = torch.sqrt(torch.clamp(r_sq, min=1e-12))
        dot = torch.sum(algebra.unsqueeze(-2) * algebra_tangent, dim=-1, keepdim=True)

        scale = torch.sin(r) / r
        scale_derivative = (r * torch.cos(r) - torch.sin(r)) / torch.clamp(r**3, min=1e-12)

        scale_small = 1 - r_sq / 6 + r_sq**2 / 120
        scale_derivative_small = -1 / 3 + r_sq / 30 - r_sq**2 / 840
        scale = torch.where(r_sq < 1e-8, scale_small, scale).unsqueeze(-2)
        scale_derivative = torch.where(r_sq < 1e-8, scale_derivative_small, scale_derivative).unsqueeze(-2)

        scalar_tangent = -scale * dot
        vector_tangent = scale * algebra_tangent + scale_derivative * dot * algebra.unsqueeze(-2)
        return torch.cat([scalar_tangent, vector_tangent], dim=-1)

    def _exp_tangent(self, algebra: torch.Tensor, algebra_tangent: torch.Tensor) -> torch.Tensor:
        value = u2_exp(algebra)
        quaternion_tangent = self._su2_exp_derivative(algebra[..., 1:], algebra_tangent[..., 1:])
        left_tangent = quaternion_mul(quaternion_tangent, quaternion_conj(value[..., 1:]).unsqueeze(-2))
        return torch.cat([algebra_tangent[..., :1], left_tangent[..., 1:]], dim=-1)

    def _layer_jacobian_blocks(
        self,
        links: torch.Tensor,
        index: int,
        plaq_coeffs: torch.Tensor,
        rect_coeffs: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = links.shape[0]
        mask = get_link_mask(index, batch_size, self.lattice_size, self.device)
        link_tangent = self._identity_tangent_like(links) * mask.to(links.dtype).unsqueeze(-1)

        plaq_loops, plaq_tangent = self._plaq_loop_stack_with_tangent(links, link_tangent)
        rect_loops, rect_tangent = self._rect_loop_stack_with_tangent(links, link_tangent)

        delta_jac = self._plaq_delta_jac(plaq_coeffs, plaq_loops, plaq_tangent)
        delta_jac = delta_jac + self._rect_delta_jac(rect_coeffs, rect_loops, rect_tangent)
        delta_jac = delta_jac * mask.to(delta_jac.dtype).unsqueeze(-1)

        exp_delta = u2_exp(delta)
        return self._exp_tangent(delta, delta_jac) + self._adjoint_algebra(exp_delta, link_tangent)

    def compute_delta(self, links: torch.Tensor, index: int) -> torch.Tensor:
        batch_size = links.shape[0]
        plaq = plaquette_from_field_batch(links)
        rect = rectangle_from_field_batch(links)
        plaq_loops = self._plaq_loop_stack(links)
        rect_loops = self._rect_loop_stack(links)
        plaq_coeffs, rect_coeffs = self.compute_coefficients(links, index, plaq, rect)
        delta = self._plaq_delta(plaq_coeffs, plaq_loops) + self._rect_delta(rect_coeffs, rect_loops)
        mask = get_link_mask(index, batch_size, self.lattice_size, self.device)
        return delta * mask.to(delta.dtype)

    def ft_phase(self, links: torch.Tensor, index: int) -> torch.Tensor:
        delta = self.compute_delta(links, index)
        return u2_mul(u2_exp(delta), links)

    def _forward_using_compiled_phase(self, links: torch.Tensor) -> torch.Tensor:
        links_curr = u2_normalize(links)
        for index in range(self.n_subsets):
            links_curr = self.ft_phase_compiled(links_curr, index)
        return u2_normalize(links_curr)

    def forward(self, links: torch.Tensor) -> torch.Tensor:
        links_curr = u2_normalize(links)
        for index in range(self.n_subsets):
            links_curr = self.ft_phase(links_curr, index)
        return u2_normalize(links_curr)

    def field_transformation(self, links: torch.Tensor) -> torch.Tensor:
        return self.forward(links.unsqueeze(0)).squeeze(0)

    def field_transformation_compiled(self, links: torch.Tensor) -> torch.Tensor:
        return self.forward_compiled(links.unsqueeze(0)).squeeze(0)

    def _inverse_using_compiled_delta(
        self,
        links: torch.Tensor,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        return self.inverse(links, max_iter=max_iter, tol=tol)

    def inverse(
        self,
        links: torch.Tensor,
        *,
        max_iter: int = 200,
        tol: float = 1e-6,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float | int]]:
        links_curr = u2_normalize(links)
        subset_final_diffs: list[float] = []
        n_not_converged = 0
        for index in reversed(range(self.n_subsets)):
            links_iter = links_curr.clone()
            diff = torch.tensor(float("inf"), device=self.device, dtype=links.dtype)
            for _ in range(max_iter):
                delta = self.compute_delta(links_iter, index)
                links_next = u2_mul(u2_exp(-delta), links_curr)
                relative = u2_log(u2_mul(links_next, u2_conj(links_iter)))
                denominator = torch.clamp(torch.linalg.norm(u2_log(links_iter)), min=1e-12)
                diff = torch.linalg.norm(relative) / denominator
                links_iter = links_next
                if diff < tol:
                    break
            final_diff = float(diff.detach().cpu())
            subset_final_diffs.append(final_diff)
            if diff >= tol:
                n_not_converged += 1
                self.print(f"Warning: inverse iteration for subset {index} did not converge, diff={diff:.2e}")
            links_curr = links_iter
        out = u2_normalize(links_curr)
        if return_diagnostics:
            diag: dict[str, float | int] = {
                "max_final_diff": max(subset_final_diffs) if subset_final_diffs else 0.0,
                "mean_final_diff": float(sum(subset_final_diffs) / len(subset_final_diffs))
                if subset_final_diffs
                else 0.0,
                "n_not_converged": n_not_converged,
            }
            return out, diag
        return out

    def inverse_field_transformation(self, links: torch.Tensor) -> torch.Tensor:
        return self.inverse(links.unsqueeze(0)).squeeze(0)

    def inverse_field_transformation_compiled(self, links: torch.Tensor) -> torch.Tensor:
        return self.inverse_compiled(links.unsqueeze(0)).squeeze(0)

    def compute_jac_logdet(self, links: torch.Tensor) -> torch.Tensor:
        return self.compute_jac_logdet_manual(links)

    def compute_jac_logdet_manual(self, links: torch.Tensor) -> torch.Tensor:
        """Compute active-link 4x4 local Jacobian blocks by analytic tangent propagation."""
        batch_size = links.shape[0]
        log_det = torch.zeros(batch_size, device=self.device, dtype=links.dtype)
        links_curr = u2_normalize(links)

        for index in range(self.n_subsets):
            plaq = plaquette_from_field_batch(links_curr)
            rect = rectangle_from_field_batch(links_curr)
            plaq_loops = self._plaq_loop_stack(links_curr)
            rect_loops = self._rect_loop_stack(links_curr)
            plaq_coeffs, rect_coeffs = self.compute_coefficients(links_curr, index, plaq, rect)
            delta = self._plaq_delta(plaq_coeffs, plaq_loops) + self._rect_delta(rect_coeffs, rect_loops)
            mask = get_link_mask(index, batch_size, self.lattice_size, self.device)
            delta = delta * mask.to(delta.dtype)

            jacobian_blocks = self._layer_jacobian_blocks(
                links_curr,
                index,
                plaq_coeffs,
                rect_coeffs,
                delta,
            )
            active_batches = torch.nonzero(mask.squeeze(-1), as_tuple=False)[:, 0]
            active_blocks = jacobian_blocks[mask.squeeze(-1)]
            _, logabsdet = torch.linalg.slogdet(active_blocks)
            log_det = log_det.scatter_add(0, active_batches, logabsdet)

            links_curr = u2_mul(u2_exp(delta), links_curr)

        return log_det

    def compute_jac_logdet_autograd(self, links: torch.Tensor) -> torch.Tensor:
        """Compute exact active-link 4x4 local Jacobian blocks with autograd."""
        batch_size = links.shape[0]
        log_det = torch.zeros(batch_size, device=self.device, dtype=links.dtype)
        links_curr = u2_normalize(links)

        for index in range(self.n_subsets):
            layer_input = links_curr
            layer_output = self.ft_phase(layer_input, index)
            mask = get_link_mask(index, batch_size, self.lattice_size, self.device).squeeze(-1)
            active_indices = torch.nonzero(mask, as_tuple=False)

            for batch_index, direction, row, col in active_indices:
                batch_int = int(batch_index.item())
                direction_int = int(direction.item())
                row_int = int(row.item())
                col_int = int(col.item())
                base_output_link = layer_output[batch_int, direction_int, row_int, col_int]

                def local_map(x: torch.Tensor) -> torch.Tensor:
                    perturbation = identity_like(layer_input)
                    perturbation = perturbation.clone()
                    perturbation[batch_int, direction_int, row_int, col_int] = u2_exp(x)
                    perturbed_input = u2_mul(perturbation, layer_input)
                    perturbed_output = self.ft_phase(perturbed_input, index)
                    output_delta = u2_mul(
                        perturbed_output[batch_int, direction_int, row_int, col_int],
                        u2_conj(base_output_link),
                    )
                    return u2_log(output_delta)

                x0 = torch.zeros(4, device=self.device, dtype=links.dtype, requires_grad=True)
                jacobian = torch.autograd.functional.jacobian(local_map, x0, create_graph=True)
                _, logabsdet = torch.linalg.slogdet(jacobian)
                log_det[batch_int] = log_det[batch_int] + logabsdet

            links_curr = layer_output

        return log_det

    def _jacobian_check_tolerances(self, links: torch.Tensor) -> tuple[float, float]:
        """Return dtype- and volume-aware tolerances for manual/autograd checks."""
        if links.dtype == torch.float64:
            return 1e-7, 1e-10

        active_blocks = 0
        for index in range(self.n_subsets):
            mask = get_link_mask(index, 1, self.lattice_size, self.device)
            active_blocks += int(mask.sum().item())
        atol = max(1e-6, 2.0 * active_blocks * torch.finfo(links.dtype).eps)
        return 1e-4, float(atol)

    def compute_action(self, links: torch.Tensor, beta: float) -> torch.Tensor:
        return action_from_field_batch(links, beta)

    def compute_force(self, links: torch.Tensor, beta: float, *, transformed: bool = False) -> torch.Tensor:
        algebra = torch.zeros(
            (*links.shape[:-1], 4),
            device=self.device,
            dtype=links.dtype,
            requires_grad=True,
        )
        varied_links = u2_mul(u2_exp(algebra), links.detach())
        if transformed:
            links_ori = self.forward_compiled(varied_links)
            jac_logdet = self.compute_jac_logdet_compiled(varied_links)
            if self.if_check_jac:
                jac_logdet_autograd = self.compute_jac_logdet_autograd(varied_links)
                abs_diff = torch.abs(jac_logdet_autograd - jac_logdet)
                rtol, atol = self._jacobian_check_tolerances(varied_links)
                denominator = torch.maximum(
                    torch.maximum(torch.abs(jac_logdet), torch.abs(jac_logdet_autograd)),
                    torch.tensor(atol, device=self.device, dtype=varied_links.dtype),
                )
                relative_diff = abs_diff / denominator
                is_close = torch.allclose(jac_logdet_autograd, jac_logdet, rtol=rtol, atol=atol)
                if not is_close:
                    self.print(
                        "\nWarning: Jacobian log determinant difference "
                        f"max_abs={abs_diff.max().item():.2e}, "
                        f"max_rel={relative_diff.max().item():.2e}, "
                        f"rtol={rtol:.1e}, atol={atol:.1e}"
                    )
                    self.print(">>> Jacobian is not correct!")
                else:
                    self.print(
                        "\nJacobian log det "
                        f"(manual): {jac_logdet[0].item():.2e}, "
                        f"(autograd): {jac_logdet_autograd[0].item():.2e}"
                    )
                    self.print(">>> Jacobian is all good!")
            total_action = self.compute_action_compiled(links_ori, beta) - jac_logdet
        else:
            total_action = self.compute_action_compiled(varied_links, beta)
        return torch.autograd.grad(total_action.sum(), algebra, create_graph=True)[0]

    def compute_transformed_force_terms(
        self,
        links: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        algebra = torch.zeros(
            (*links.shape[:-1], 4),
            device=self.device,
            dtype=links.dtype,
            requires_grad=True,
        )
        varied_links = u2_mul(u2_exp(algebra), links.detach())
        links_ori = self.forward_compiled(varied_links)
        action = self.compute_action_compiled(links_ori, beta)
        jac_logdet = self.compute_jac_logdet_compiled(varied_links)
        action_force = torch.autograd.grad(action.sum(), algebra, create_graph=True, retain_graph=True)[0]
        jac_force = torch.autograd.grad(jac_logdet.sum(), algebra, create_graph=True)[0]
        total_force = action_force - jac_force
        return total_force, action_force, jac_force, jac_logdet

    def loss_fn(self, links_ori: torch.Tensor) -> torch.Tensor:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        links_new = self.inverse(links_ori)
        force_new, _, _, _ = self.compute_transformed_force_terms(links_new, self.train_beta)
        return self._weighted_force_loss_tensor(force_new)

    def _loss_weights(self) -> tuple[float, float, float, float]:
        weights = self.hyperparams.get("loss_weights", (1.0, 1.0, 1.0, 1.0))
        try:
            values = tuple(float(weight) for weight in weights)
        except TypeError as exc:
            raise ValueError("loss_weights must contain exactly four numeric values") from exc
        if len(values) != 4:
            raise ValueError("loss_weights must contain exactly four numeric values")
        return values

    def _weighted_force_loss(self, components: dict[str, float]) -> float:
        w2, w4, w6, w8 = self._loss_weights()
        return (
            w2 * components["l2"]
            + w4 * components["l4"]
            + w6 * components["l6"]
            + w8 * components["l8"]
        )

    def _weighted_force_loss_tensor(self, force: torch.Tensor) -> torch.Tensor:
        volume = self.lattice_size * self.lattice_size
        force_flat = force.reshape(force.shape[0], -1)
        force_l2 = torch.linalg.vector_norm(force_flat, ord=2, dim=1) / (volume**0.5)
        force_l4 = torch.linalg.vector_norm(force_flat, ord=4, dim=1) / (volume**0.25)
        force_l6 = torch.linalg.vector_norm(force_flat, ord=6, dim=1) / (volume ** (1 / 6))
        force_l8 = torch.linalg.vector_norm(force_flat, ord=8, dim=1) / (volume ** (1 / 8))
        w2, w4, w6, w8 = self._loss_weights()
        return (w2 * force_l2 + w4 * force_l4 + w6 * force_l6 + w8 * force_l8).mean()

    def _force_loss_components(self, force: torch.Tensor) -> dict[str, float]:
        volume = self.lattice_size * self.lattice_size
        force_flat = force.reshape(force.shape[0], -1)
        return {
            "l2": float((torch.linalg.vector_norm(force_flat, ord=2, dim=1) / (volume**0.5)).mean().detach().cpu()),
            "l4": float((torch.linalg.vector_norm(force_flat, ord=4, dim=1) / (volume**0.25)).mean().detach().cpu()),
            "l6": float(
                (torch.linalg.vector_norm(force_flat, ord=6, dim=1) / (volume ** (1 / 6))).mean().detach().cpu()
            ),
            "l8": float(
                (torch.linalg.vector_norm(force_flat, ord=8, dim=1) / (volume ** (1 / 8))).mean().detach().cpu()
            ),
        }

    def _parameter_diagnostics(self) -> tuple[float, float]:
        norm_sq = 0.0
        max_abs = 0.0
        for model in self.models:
            for param in model.parameters():
                detached = param.detach()
                norm_sq += float(torch.sum(detached**2).cpu())
                max_abs = max(max_abs, float(torch.max(torch.abs(detached)).cpu()))
        return norm_sq**0.5, max_abs

    def _delta_diagnostics(self, links: torch.Tensor) -> tuple[float, float]:
        volume = self.lattice_size * self.lattice_size
        links_curr = u2_normalize(links)
        delta_norms = []
        for index in range(self.n_subsets):
            delta = self.compute_delta(links_curr, index)
            delta_norms.append(torch.linalg.vector_norm(delta.reshape(delta.shape[0], -1), dim=1) / (volume**0.5))
            links_curr = u2_mul(u2_exp(delta), links_curr)
        if not delta_norms:
            return 0.0, 0.0
        stacked = torch.stack(delta_norms, dim=1)
        return float(stacked.mean().cpu()), float(stacked.max().cpu())

    def _coefficient_diagnostics(self, links: torch.Tensor) -> dict[str, float]:
        plaq_coeff_abs = []
        rect_coeff_abs = []
        links_curr = u2_normalize(links)
        for index in range(self.n_subsets):
            plaq = plaquette_from_field_batch(links_curr)
            rect = rectangle_from_field_batch(links_curr)
            plaq_loops = self._plaq_loop_stack(links_curr)
            rect_loops = self._rect_loop_stack(links_curr)
            plaq_coeffs, rect_coeffs = self.compute_coefficients(links_curr, index, plaq, rect)
            plaq_coeff_abs.append(torch.abs(plaq_coeffs).reshape(plaq_coeffs.shape[0], -1))
            rect_coeff_abs.append(torch.abs(rect_coeffs).reshape(rect_coeffs.shape[0], -1))
            delta = self._plaq_delta(plaq_coeffs, plaq_loops) + self._rect_delta(rect_coeffs, rect_loops)
            mask = get_link_mask(index, links.shape[0], self.lattice_size, self.device)
            links_curr = u2_mul(u2_exp(delta * mask.to(delta.dtype)), links_curr)
        if not plaq_coeff_abs or not rect_coeff_abs:
            return {
                "k0_abs_mean": 0.0,
                "k0_abs_max": 0.0,
                "k0_sat_frac": 0.0,
                "k1_abs_mean": 0.0,
                "k1_abs_max": 0.0,
                "k1_sat_frac": 0.0,
            }
        k0_all = torch.cat(plaq_coeff_abs, dim=1)
        k1_all = torch.cat(rect_coeff_abs, dim=1)
        k0_cap = 1 / 6
        k1_cap = 1 / 45
        return {
            "k0_abs_mean": float(k0_all.mean().cpu()),
            "k0_abs_max": float(k0_all.max().cpu()),
            "k0_sat_frac": float((k0_all > 0.95 * k0_cap).to(torch.float32).mean().cpu()),
            "k1_abs_mean": float(k1_all.mean().cpu()),
            "k1_abs_max": float(k1_all.max().cpu()),
            "k1_sat_frac": float((k1_all > 0.95 * k1_cap).to(torch.float32).mean().cpu()),
        }

    def _maybe_log_training_diagnostics(
        self,
        test_data: torch.Tensor,
        batch_size: int,
        epoch_display: int,
        n_epochs: int,
        grad_norm: float,
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
            x0 = u2_normalize(probe)
            rt = u2_mul(recon, u2_conj(x0))
            rt_err = torch.linalg.norm(u2_log(rt).reshape(probe.shape[0], -1), dim=1).mean().item()
            delta_mean, delta_max = self._delta_diagnostics(inv)
            coefficient_diag = self._coefficient_diagnostics(inv)
            param_norm, param_max = self._parameter_diagnostics()
        with torch.enable_grad():
            force, action_force, jac_force, jac_logdet = self.compute_transformed_force_terms(
                inv.detach(),
                self.train_beta,
            )
            force_components = self._force_loss_components(force)
            action_force_components = self._force_loss_components(action_force)
            jac_force_components = self._force_loss_components(jac_force)
        force_weighted_loss = self._weighted_force_loss(force_components)
        jac_logdet = jac_logdet.detach()
        jac_std = jac_logdet.std(unbiased=False).item()
        self.print(
            f"Epoch {epoch_display}/{n_epochs} inverse_diag: "
            f"max_final_diff={diag['max_final_diff']:.2e} "
            f"mean_final_diff={diag['mean_final_diff']:.2e} "
            f"n_subsets_not_converged={diag['n_not_converged']} "
            f"round_trip_mean_log_norm={rt_err:.2e}"
        )
        self.print(
            f"Epoch {epoch_display}/{n_epochs} train_diag: "
            f"grad_norm_pre_clip={grad_norm:.2e} "
            f"param_norm={param_norm:.2e} param_max_abs={param_max:.2e} "
            f"delta_norm_mean={delta_mean:.2e} delta_norm_max={delta_max:.2e} "
            f"loss_weights={self._loss_weights()} "
            f"force_weighted_loss={force_weighted_loss:.6f} "
            f"k0_abs_mean={coefficient_diag['k0_abs_mean']:.2e} "
            f"k0_abs_max={coefficient_diag['k0_abs_max']:.2e} "
            f"k0_sat_frac={coefficient_diag['k0_sat_frac']:.3f} "
            f"k1_abs_mean={coefficient_diag['k1_abs_mean']:.2e} "
            f"k1_abs_max={coefficient_diag['k1_abs_max']:.2e} "
            f"k1_sat_frac={coefficient_diag['k1_sat_frac']:.3f} "
            f"jac_logdet_mean={jac_logdet.mean().item():.2e} "
            f"jac_logdet_std={jac_std:.2e} "
            f"jac_logdet_min={jac_logdet.min().item():.2e} "
            f"jac_logdet_max={jac_logdet.max().item():.2e} "
            f"force_l2={force_components['l2']:.6f} "
            f"force_l4={force_components['l4']:.6f} "
            f"force_l6={force_components['l6']:.6f} "
            f"force_l8={force_components['l8']:.6f}"
        )
        self.print(
            f"Epoch {epoch_display}/{n_epochs} force_split_diag: "
            f"action_l2={action_force_components['l2']:.6f} "
            f"action_l4={action_force_components['l4']:.6f} "
            f"action_l6={action_force_components['l6']:.6f} "
            f"action_l8={action_force_components['l8']:.6f} "
            f"jac_l2={jac_force_components['l2']:.6f} "
            f"jac_l4={jac_force_components['l4']:.6f} "
            f"jac_l6={jac_force_components['l6']:.6f} "
            f"jac_l8={jac_force_components['l8']:.6f}"
        )

    def train_step(self, links_ori: torch.Tensor) -> tuple[float, float]:
        links_ori = links_ori.to(self.device)
        loss = self.loss_fn(links_ori)
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.backward(loss)
        grad_norm = self._gradient_norm()
        self._clip_gradients()
        for optimizer in self.optimizers:
            optimizer.step()
        return float(loss.detach().cpu()), grad_norm

    def evaluate_step(self, links_ori: torch.Tensor) -> float:
        links_ori = links_ori.to(self.device)
        loss = self.loss_fn(links_ori)
        return float(loss.detach().cpu())

    def train(self, train_data: torch.Tensor, test_data: torch.Tensor, train_beta: float, *, n_epochs: int, batch_size: int) -> None:
        self.train_beta = train_beta
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=self.num_workers)
        if self.fabric is not None:
            train_loader = self.fabric.setup_dataloaders(train_loader)
            test_loader = self.fabric.setup_dataloaders(test_loader)

        train_losses: list[float] = []
        test_losses: list[float] = []
        best_loss = float("inf")
        best_epoch = -1
        epochs_without_improvement = 0
        early_stop_patience = int(self.hyperparams.get("early_stop_patience", 0))

        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            self._set_models_mode(True)
            epoch_losses = []
            grad_norms = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
                batch_loss, grad_norm = self.train_step(batch)
                epoch_losses.append((batch_loss, len(batch)))
                grad_norms.append(grad_norm)
            train_loss = self._global_weighted_epoch_loss(epoch_losses)
            mean_grad_norm = self._global_mean(grad_norms)
            train_losses.append(train_loss)

            self._set_models_mode(False)
            test_epoch_losses = [
                (self.evaluate_step(batch), len(batch))
                for batch in tqdm(test_loader, desc="Evaluating")
            ]
            test_loss = self._global_weighted_epoch_loss(test_epoch_losses)
            test_losses.append(test_loss)

            improved = test_loss < best_loss
            if improved:
                self.save_best_model(epoch, test_loss)
                best_loss = test_loss
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            for scheduler in self.schedulers:
                scheduler.step(test_loss)

            self.print(
                f"Epoch {epoch + 1}/{n_epochs} - "
                f"Train Loss: {train_loss:.6f} - Test Loss: {test_loss:.6f} - "
                f"LR: {self._format_learning_rates()}"
            )
            self._maybe_log_training_diagnostics(test_data, batch_size, epoch + 1, n_epochs, mean_grad_norm)
            if (
                early_stop_patience > 0
                and best_epoch >= 0
                and epochs_without_improvement >= early_stop_patience
            ):
                self.print(
                    f"Early stopping at epoch {epoch + 1}; "
                    f"best epoch {best_epoch + 1} with test loss {best_loss:.6f}"
                )
                break

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
        local_loss_sum = sum(loss * count for loss, count in losses_and_counts)
        local_count = sum(count for _, count in losses_and_counts)
        totals = torch.tensor([local_loss_sum, local_count], device=self.device, dtype=torch.float64)
        if self.fabric is not None:
            totals = self.fabric.all_reduce(totals, reduce_op="sum")
        total_count = float(totals[1].item())
        if total_count == 0:
            return float("nan")
        return float((totals[0] / totals[1]).item())

    def _global_mean(self, values: list[float]) -> float:
        local_sum = sum(values)
        local_count = len(values)
        totals = torch.tensor([local_sum, local_count], device=self.device, dtype=torch.float64)
        if self.fabric is not None:
            totals = self.fabric.all_reduce(totals, reduce_op="sum")
        total_count = float(totals[1].item())
        if total_count == 0:
            return float("nan")
        return float((totals[0] / totals[1]).item())

    def _format_learning_rates(self) -> str:
        rates = []
        for optimizer in self.optimizers:
            rates.extend(float(group["lr"]) for group in optimizer.param_groups)
        unique_rates = sorted(set(rates))
        return ",".join(f"{rate:.3e}" for rate in unique_rates)

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.pt"

    def save_best_model(self, epoch: int, loss: float) -> None:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        if self.fabric is not None and self.fabric.global_rank != 0:
            return
        self.model_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {"epoch": epoch, "loss": loss}
        for index, model in enumerate(self.models):
            save_dict[f"model_state_dict_{index}"] = model.state_dict()
        for index, optimizer in enumerate(self.optimizers):
            save_dict[f"optimizer_state_dict_{index}"] = optimizer.state_dict()
        torch.save(save_dict, self.checkpoint_path(self.train_beta))

    def load_best_model(self, train_beta: float) -> None:
        checkpoint = torch.load(self.checkpoint_path(train_beta), map_location=self.device, weights_only=False)
        for index, model in enumerate(self.models):
            state_dict = checkpoint[f"model_state_dict_{index}"]
            if any(key.startswith("module.") for key in state_dict):
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            model.load_state_dict(state_dict)
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
