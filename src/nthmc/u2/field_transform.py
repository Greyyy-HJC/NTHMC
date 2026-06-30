"""JAX neural field transformation for 2D U(2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nthmc.core.checkpoint import load_checkpoint, save_checkpoint
from nthmc.u2.models import apply_model, init_transform_params
from nthmc.u2.u2_observables import (
    action_from_field,
    format_beta,
    get_link_mask,
    get_plaq_mask,
    get_rect_mask,
    identity_like,
    loop_sin_cos_features,
    plaquette_from_field_batch,
    quaternion_conj,
    quaternion_mul,
    rectangle_from_field_batch,
    u2_conj,
    u2_exp,
    u2_log,
    u2_mul,
    u2_normalize,
)

Array = Any
Params = dict[str, Any]
LoopToken = tuple[int, int | tuple[int, int] | None, int | tuple[int, int] | None, bool]


class FieldTransformation:
    """JAX U(2) coupling transform with exact active-link Jacobian blocks."""

    def __init__(
        self,
        lattice_size: int,
        *,
        device: str = "cpu",
        n_subsets: int = 8,
        if_check_jac: bool = False,
        model_tag: str = "base",
        save_tag: str | None = None,
        model_dir: str | Path = "artifacts/models",
        plot_dir: str | Path = "plots",
        dump_dir: str | Path = "dumps",
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        self.lattice_size = lattice_size
        self.device = device
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.model_tag = model_tag
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.plot_dir = Path(plot_dir)
        self.dump_dir = Path(dump_dir)
        self.train_beta: float | None = None
        self.hyperparams: dict[str, Any] = {
            "init_std": 0.001,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "max_grad_norm": 10.0,
            "early_stop_patience": 20,
            "inverse_max_iters": 200,
            "inverse_tol": 1e-6,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)
        self.params = init_transform_params(
            jax.random.PRNGKey(0),
            model_tag,
            n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(float(self.hyperparams["max_grad_norm"])),
            optax.adamw(float(self.hyperparams["lr"]), weight_decay=float(self.hyperparams["weight_decay"])),
        )
        self.opt_state = self.optimizer.init(self.params)

    def _checkpoint_template(self) -> Params:
        return init_transform_params(
            jax.random.PRNGKey(0),
            self.model_tag,
            self.n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.npz"

    def save_best_model(self, train_beta: float, epoch: int, loss: float) -> None:
        metadata = {
            "system": "2du2",
            "transform": "neural_u2_jax",
            "model_tag": self.model_tag,
            "n_subsets": self.n_subsets,
            "lattice_size": self.lattice_size,
            "train_beta": float(train_beta),
            "epoch": int(epoch),
            "loss": float(loss),
            "hyperparams": self.hyperparams,
        }
        save_checkpoint(self.checkpoint_path(train_beta), params=self.params, opt_state=self.opt_state, metadata=metadata)

    def load_best_model(self, train_beta: float) -> None:
        self.params, metadata = load_checkpoint(self.checkpoint_path(train_beta), self._checkpoint_template())
        self.opt_state = self.optimizer.init(self.params)
        print(f"Loaded JAX U(2) checkpoint from epoch {metadata.get('epoch')} with loss {metadata.get('loss')}")

    @staticmethod
    def _features_from_loops(loops: Array) -> Array:
        features = loop_sin_cos_features(loops)
        return jnp.moveaxis(features, -1, 1)

    @staticmethod
    def _masked_loops(loops: Array, mask: Array) -> Array:
        return jnp.where(mask, loops, identity_like(loops))

    def _cnn_features(self, links: Array, index: int) -> tuple[Array, Array]:
        batch_size = links.shape[0]
        plaquettes = plaquette_from_field_batch(links)
        rectangles = rectangle_from_field_batch(links)
        plaq_mask = get_plaq_mask(index, batch_size, self.lattice_size)
        rect_mask = get_rect_mask(index, batch_size, self.lattice_size)
        plaq_features = self._features_from_loops(self._masked_loops(plaquettes, plaq_mask))
        rect_features = self._features_from_loops(self._masked_loops(rectangles, rect_mask)).reshape(
            batch_size,
            16,
            self.lattice_size,
            self.lattice_size,
        )
        return plaq_features, rect_features

    @staticmethod
    def _loop_product(parts: list[Array]) -> Array:
        value = parts[0]
        for part in parts[1:]:
            value = u2_mul(value, part)
        return value

    def _loop_product_with_tangent(self, parts: list[tuple[Array, Array]]) -> tuple[Array, Array]:
        value, value_tangent = parts[0]
        for next_value, next_tangent in parts[1:]:
            value, value_tangent = self._mul_with_tangent(value, value_tangent, next_value, next_tangent)
        return value, value_tangent

    @staticmethod
    def _resolve_loop_token(link0: Array, link1: Array, token: LoopToken) -> Array:
        direction, shifts, axes, is_inverse = token
        value = link0 if direction == 0 else link1
        if shifts is not None and axes is not None:
            value = jnp.roll(value, shift=shifts, axis=axes)
        return u2_conj(value) if is_inverse else value

    def _resolve_loop_token_with_tangent(
        self,
        link0: Array,
        link1: Array,
        tangent0: Array,
        tangent1: Array,
        token: LoopToken,
    ) -> tuple[Array, Array]:
        direction, shifts, axes, is_inverse = token
        value = link0 if direction == 0 else link1
        tangent = tangent0 if direction == 0 else tangent1
        if shifts is not None and axes is not None:
            value = jnp.roll(value, shift=shifts, axis=axes)
            tangent = jnp.roll(tangent, shift=shifts, axis=axes)
        return self._conj_with_tangent(value, tangent) if is_inverse else (value, tangent)

    def _stack_loop_specs(self, links: Array, specs: list[list[LoopToken]]) -> Array:
        links = u2_normalize(links)
        link0, link1 = links[:, 0], links[:, 1]
        loops = [
            self._loop_product([self._resolve_loop_token(link0, link1, token) for token in loop_spec])
            for loop_spec in specs
        ]
        return jnp.stack(loops, axis=1)

    def _stack_loop_specs_with_tangent(
        self,
        links: Array,
        tangent: Array,
        specs: list[list[LoopToken]],
    ) -> tuple[Array, Array]:
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
        return jnp.stack(loop_values, axis=1), jnp.stack(loop_tangents, axis=1)

    @staticmethod
    def _plaq_loop_specs() -> list[list[LoopToken]]:
        return [
            [(0, None, None, False), (1, -1, 1, False), (0, -1, 2, True), (1, None, None, True)],
            [(1, 1, 2, True), (0, 1, 2, False), (1, (-1, 1), (1, 2), False), (0, None, None, True)],
            [(0, None, None, False), (1, -1, 1, False), (0, -1, 2, True), (1, None, None, True)],
            [(1, None, None, False), (0, (1, -1), (1, 2), True), (1, 1, 1, True), (0, 1, 1, False)],
        ]

    @staticmethod
    def _rect_loop_specs() -> list[list[LoopToken]]:
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

    def _plaq_loop_stack(self, links: Array) -> Array:
        return self._stack_loop_specs(links, self._plaq_loop_specs())

    def _rect_loop_stack(self, links: Array) -> Array:
        return self._stack_loop_specs(links, self._rect_loop_specs())

    def _plaq_loop_stack_with_tangent(self, links: Array, tangent: Array) -> tuple[Array, Array]:
        return self._stack_loop_specs_with_tangent(links, tangent, self._plaq_loop_specs())

    def _rect_loop_stack_with_tangent(self, links: Array, tangent: Array) -> tuple[Array, Array]:
        return self._stack_loop_specs_with_tangent(links, tangent, self._rect_loop_specs())

    @staticmethod
    def _loop_delta(coeffs: Array, loops: Array, signs: Array) -> Array:
        batch_size, n_loops = loops.shape[:2]
        features = loop_sin_cos_features(loops)
        coeffs = coeffs.reshape(batch_size, n_loops, 4, loops.shape[2], loops.shape[3])
        coeffs = jnp.moveaxis(coeffs, 2, -1)
        signs = signs.astype(loops.dtype).reshape(1, n_loops, 1, 1, 1)

        phase_delta = coeffs[..., 0:1] * features[..., 0:1] * signs
        phase_delta = phase_delta + coeffs[..., 2:3] * features[..., 4:5]
        traceless_delta = coeffs[..., 1:2] * features[..., 1:4] * signs
        traceless_delta = traceless_delta + coeffs[..., 3:4] * features[..., 5:8]
        return jnp.concatenate([phase_delta, traceless_delta], axis=-1)

    def _plaq_delta(self, coeffs: Array, plaq_loops: Array) -> Array:
        signs = jnp.asarray([-1.0, 1.0, 1.0, -1.0], dtype=plaq_loops.dtype)
        per_loop = self._loop_delta(coeffs, plaq_loops, signs)
        return jnp.stack([per_loop[:, 0:2].sum(axis=1), per_loop[:, 2:4].sum(axis=1)], axis=1)

    def _rect_delta(self, coeffs: Array, rect_loops: Array) -> Array:
        signs = jnp.asarray([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=rect_loops.dtype)
        per_loop = self._loop_delta(coeffs, rect_loops, signs)
        return jnp.stack([per_loop[:, 0:4].sum(axis=1), per_loop[:, 4:8].sum(axis=1)], axis=1)

    def _compute_delta(self, params: Params, links: Array, index: int) -> Array:
        batch_size = links.shape[0]
        plaq_features, rect_features = self._cnn_features(links, index)
        plaq_coeffs, rect_coeffs = apply_model(params["subsets"][index], self.model_tag, plaq_features, rect_features)
        plaq_coeffs = plaq_coeffs.reshape(batch_size, 4, 4, self.lattice_size, self.lattice_size)
        rect_coeffs = rect_coeffs.reshape(batch_size, 8, 4, self.lattice_size, self.lattice_size)
        delta = self._plaq_delta(plaq_coeffs, self._plaq_loop_stack(links))
        delta = delta + self._rect_delta(rect_coeffs, self._rect_loop_stack(links))
        return delta * get_link_mask(index, batch_size, self.lattice_size)

    @staticmethod
    def _identity_tangent_like(links: Array) -> Array:
        identity = jnp.eye(4, dtype=links.dtype)
        return jnp.broadcast_to(identity, (*links.shape[:-1], 4, 4))

    @staticmethod
    def _adjoint_algebra(links: Array, algebra: Array) -> Array:
        phase = algebra[..., :1]
        q = links[..., 1:][..., jnp.newaxis, :]
        pure = jnp.concatenate([jnp.zeros_like(algebra[..., 1:2]), algebra[..., 1:]], axis=-1)
        rotated = quaternion_mul(quaternion_mul(q, pure), quaternion_conj(q))
        return jnp.concatenate([phase, rotated[..., 1:]], axis=-1)

    def _mul_with_tangent(self, left: Array, left_tangent: Array, right: Array, right_tangent: Array) -> tuple[Array, Array]:
        return u2_mul(left, right), left_tangent + self._adjoint_algebra(left, right_tangent)

    def _conj_with_tangent(self, links: Array, tangent: Array) -> tuple[Array, Array]:
        inverse = u2_conj(links)
        return inverse, -self._adjoint_algebra(inverse, tangent)

    def _loop_feature_tangent(self, loops: Array, loop_tangent: Array) -> Array:
        loops = u2_normalize(loops)
        phase = loops[..., :1]
        q = loops[..., 1:]
        q0 = q[..., :1]
        qv = q[..., 1:]
        phase_tangent = loop_tangent[..., :1]
        pure_tangent = jnp.concatenate([jnp.zeros_like(loop_tangent[..., 1:2]), loop_tangent[..., 1:]], axis=-1)
        q_tangent = quaternion_mul(pure_tangent, q[..., jnp.newaxis, :])
        q0_tangent = q_tangent[..., :1]
        qv_tangent = q_tangent[..., 1:]

        sin_phase = jnp.sin(phase)[..., jnp.newaxis, :]
        cos_phase = jnp.cos(phase)[..., jnp.newaxis, :]
        q0 = q0[..., jnp.newaxis, :]
        qv = qv[..., jnp.newaxis, :]

        sin_like_phase = q0_tangent * sin_phase + q0 * cos_phase * phase_tangent
        sin_like_traceless = qv_tangent * cos_phase - qv * sin_phase * phase_tangent
        cos_like_phase = q0_tangent * cos_phase - q0 * sin_phase * phase_tangent
        cos_like_traceless = -qv_tangent * sin_phase - qv * cos_phase * phase_tangent
        return jnp.concatenate([sin_like_phase, sin_like_traceless, cos_like_phase, cos_like_traceless], axis=-1)

    def _loop_delta_jac(self, coeffs: Array, loops: Array, loop_tangent: Array, signs: Array) -> Array:
        batch_size, n_loops = loops.shape[:2]
        feature_tangent = self._loop_feature_tangent(loops, loop_tangent)
        coeffs = coeffs.reshape(batch_size, n_loops, 4, loops.shape[2], loops.shape[3])
        coeffs = jnp.moveaxis(coeffs, 2, -1)[..., jnp.newaxis, :]
        signs = signs.astype(loops.dtype).reshape(1, n_loops, 1, 1, 1, 1)

        phase_jac = coeffs[..., 0:1] * feature_tangent[..., 0:1] * signs
        phase_jac = phase_jac + coeffs[..., 2:3] * feature_tangent[..., 4:5]
        traceless_jac = coeffs[..., 1:2] * feature_tangent[..., 1:4] * signs
        traceless_jac = traceless_jac + coeffs[..., 3:4] * feature_tangent[..., 5:8]
        return jnp.concatenate([phase_jac, traceless_jac], axis=-1)

    def _plaq_delta_jac(self, coeffs: Array, plaq_loops: Array, plaq_tangent: Array) -> Array:
        signs = jnp.asarray([-1.0, 1.0, 1.0, -1.0], dtype=plaq_loops.dtype)
        per_loop = self._loop_delta_jac(coeffs, plaq_loops, plaq_tangent, signs)
        return jnp.stack([per_loop[:, 0:2].sum(axis=1), per_loop[:, 2:4].sum(axis=1)], axis=1)

    def _rect_delta_jac(self, coeffs: Array, rect_loops: Array, rect_tangent: Array) -> Array:
        signs = jnp.asarray([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0], dtype=rect_loops.dtype)
        per_loop = self._loop_delta_jac(coeffs, rect_loops, rect_tangent, signs)
        return jnp.stack([per_loop[:, 0:4].sum(axis=1), per_loop[:, 4:8].sum(axis=1)], axis=1)

    @staticmethod
    def _su2_exp_derivative(algebra: Array, algebra_tangent: Array) -> Array:
        r_sq = jnp.sum(algebra**2, axis=-1, keepdims=True)
        r = jnp.sqrt(jnp.clip(r_sq, min=1e-12))
        dot = jnp.sum(algebra[..., jnp.newaxis, :] * algebra_tangent, axis=-1, keepdims=True)

        scale = jnp.sin(r) / r
        scale_derivative = (r * jnp.cos(r) - jnp.sin(r)) / jnp.clip(r**3, min=1e-12)
        scale_small = 1 - r_sq / 6 + r_sq**2 / 120
        scale_derivative_small = -1 / 3 + r_sq / 30 - r_sq**2 / 840
        scale = jnp.where(r_sq < 1e-8, scale_small, scale)[..., jnp.newaxis, :]
        scale_derivative = jnp.where(r_sq < 1e-8, scale_derivative_small, scale_derivative)[..., jnp.newaxis, :]

        scalar_tangent = -scale * dot
        vector_tangent = scale * algebra_tangent + scale_derivative * dot * algebra[..., jnp.newaxis, :]
        return jnp.concatenate([scalar_tangent, vector_tangent], axis=-1)

    def _exp_tangent(self, algebra: Array, algebra_tangent: Array) -> Array:
        value = u2_exp(algebra)
        quaternion_tangent = self._su2_exp_derivative(algebra[..., 1:], algebra_tangent[..., 1:])
        left_tangent = quaternion_mul(quaternion_tangent, quaternion_conj(value[..., 1:])[..., jnp.newaxis, :])
        return jnp.concatenate([algebra_tangent[..., :1], left_tangent[..., 1:]], axis=-1)

    def _layer_jacobian_blocks(
        self,
        links: Array,
        index: int,
        plaq_coeffs: Array,
        rect_coeffs: Array,
        delta: Array,
    ) -> Array:
        batch_size = links.shape[0]
        mask = get_link_mask(index, batch_size, self.lattice_size)
        link_tangent = self._identity_tangent_like(links) * mask.astype(links.dtype)[..., jnp.newaxis]

        plaq_loops, plaq_tangent = self._plaq_loop_stack_with_tangent(links, link_tangent)
        rect_loops, rect_tangent = self._rect_loop_stack_with_tangent(links, link_tangent)
        delta_jac = self._plaq_delta_jac(plaq_coeffs, plaq_loops, plaq_tangent)
        delta_jac = delta_jac + self._rect_delta_jac(rect_coeffs, rect_loops, rect_tangent)
        delta_jac = delta_jac * mask.astype(delta_jac.dtype)[..., jnp.newaxis]

        exp_delta = u2_exp(delta)
        return self._exp_tangent(delta, delta_jac) + self._adjoint_algebra(exp_delta, link_tangent)

    def _apply_subset(self, params: Params, links: Array, index: int) -> Array:
        delta = self._compute_delta(params, links, index)
        return u2_normalize(u2_mul(u2_exp(delta), links))

    def forward_with_params(self, params: Params, links: Array) -> Array:
        links_curr = u2_normalize(links)
        for index in range(self.n_subsets):
            links_curr = self._apply_subset(params, links_curr, index)
        return u2_normalize(links_curr)

    def forward(self, links: Array) -> Array:
        links = jnp.asarray(links)
        if links.ndim == 4:
            return self.forward_with_params(self.params, links[jnp.newaxis, ...])[0]
        return self.forward_with_params(self.params, links)

    def field_transformation(self, links: Array) -> Array:
        return self.forward(jnp.asarray(links))

    def _inverse_settings(self, max_iter: int | None = None, tol: float | None = None) -> tuple[int, float]:
        resolved_max_iter = int(
            max_iter
            or self.hyperparams.get("inverse_max_iters", self.hyperparams.get("inverse_iters", 200))
        )
        resolved_tol = float(tol or self.hyperparams.get("inverse_tol", 1e-6))
        return resolved_max_iter, resolved_tol

    def inverse_with_params(
        self,
        params: Params,
        links: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> Array:
        return self.inverse_with_diagnostics(params, links, max_iter=max_iter, tol=tol)[0]

    def inverse_with_diagnostics(
        self,
        params: Params,
        links: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        max_iter, tol = self._inverse_settings(max_iter, tol)
        links_curr = u2_normalize(links)
        final_diffs = []
        n_not_converged = jnp.asarray(0, dtype=jnp.int32)

        for index in reversed(range(self.n_subsets)):
            links_iter = links_curr
            init_diff = jnp.asarray(jnp.inf, dtype=links.dtype)
            init_converged = jnp.asarray(False)

            def iteration_step(carry: tuple[Array, Array, Array], _: Array) -> tuple[tuple[Array, Array, Array], None]:
                links_iter, diff, converged = carry
                delta = self._compute_delta(params, links_iter, index)
                links_next = u2_normalize(u2_mul(u2_exp(-delta), links_curr))
                relative = u2_log(u2_mul(links_next, u2_conj(links_iter)))
                denominator = jnp.clip(jnp.linalg.norm(u2_log(links_iter)), min=1e-12)
                next_diff = jnp.linalg.norm(relative) / denominator
                should_update = jnp.logical_not(converged)
                links_iter = jnp.where(should_update, links_next, links_iter)
                diff = jnp.where(should_update, next_diff, diff)
                converged = jnp.logical_or(converged, next_diff < tol)
                return (links_iter, diff, converged), None

            (links_iter, final_diff, converged), _ = jax.lax.scan(
                iteration_step,
                (links_iter, init_diff, init_converged),
                xs=None,
                length=max_iter,
            )
            final_diffs.append(final_diff)
            n_not_converged = n_not_converged + jnp.asarray(jnp.logical_not(converged), dtype=jnp.int32)
            links_curr = links_iter

        out = u2_normalize(links_curr)
        final_diffs = jnp.stack(final_diffs)
        recon = self.forward_with_params(params, out)
        relative = u2_log(u2_mul(recon, u2_conj(links)))
        diagnostics = {
            "max_final_diff": jnp.max(final_diffs),
            "mean_final_diff": jnp.mean(final_diffs),
            "n_not_converged": n_not_converged,
            "round_trip_mean_abs_err": jnp.mean(jnp.abs(relative)),
        }
        return out, diagnostics

    def inverse(
        self,
        links: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> Array | tuple[Array, dict[str, Array]]:
        links = jnp.asarray(links)
        if links.ndim == 4:
            out, diagnostics = self.inverse_with_diagnostics(self.params, links[jnp.newaxis, ...], max_iter=max_iter, tol=tol)
            out = out[0]
        else:
            out, diagnostics = self.inverse_with_diagnostics(self.params, links, max_iter=max_iter, tol=tol)
        return (out, diagnostics) if return_diagnostics else out

    def inverse_field_transformation(self, links: Array) -> Array:
        return self.inverse(links)

    def _subset_coeffs_delta(self, params: Params, links: Array, index: int) -> tuple[Array, Array, Array]:
        batch_size = links.shape[0]
        plaq_features, rect_features = self._cnn_features(links, index)
        plaq_coeffs, rect_coeffs = apply_model(params["subsets"][index], self.model_tag, plaq_features, rect_features)
        plaq_coeffs = plaq_coeffs.reshape(batch_size, 4, 4, self.lattice_size, self.lattice_size)
        rect_coeffs = rect_coeffs.reshape(batch_size, 8, 4, self.lattice_size, self.lattice_size)
        delta = self._plaq_delta(plaq_coeffs, self._plaq_loop_stack(links))
        delta = delta + self._rect_delta(rect_coeffs, self._rect_loop_stack(links))
        delta = delta * get_link_mask(index, batch_size, self.lattice_size)
        return plaq_coeffs, rect_coeffs, delta

    def compute_jac_logdet_with_params(self, params: Params, links: Array) -> Array:
        links_curr = u2_normalize(links)
        logdet = jnp.zeros((links.shape[0],), dtype=links.dtype)
        for index in range(self.n_subsets):
            plaq_coeffs, rect_coeffs, delta = self._subset_coeffs_delta(params, links_curr, index)
            jacobian_blocks = self._layer_jacobian_blocks(links_curr, index, plaq_coeffs, rect_coeffs, delta)
            mask = get_link_mask(index, links.shape[0], self.lattice_size)[..., 0]
            blocks = jacobian_blocks.reshape(links.shape[0], -1, 4, 4)
            mask_flat = mask.reshape(links.shape[0], -1)
            identity = jnp.eye(4, dtype=links.dtype)
            blocks = jnp.where(mask_flat[..., jnp.newaxis, jnp.newaxis], blocks, identity)
            _, local_logdet = jnp.linalg.slogdet(blocks)
            logdet = logdet + jnp.sum(local_logdet, axis=1)
            links_curr = u2_normalize(u2_mul(u2_exp(delta), links_curr))
        return logdet

    def compute_jac_logdet_autodiff_with_params(self, params: Params, links: Array) -> Array:
        """Full-field autodiff Jacobian check for the first batch item only."""
        links_single = jnp.asarray(links)[0]
        base_output = self.forward_with_params(params, links_single[jnp.newaxis, ...])[0]
        algebra_shape = (*links_single.shape[:-1], 4)

        def flat_forward(flat_algebra: Array) -> Array:
            algebra = flat_algebra.reshape(algebra_shape)
            varied_input = u2_mul(u2_exp(algebra), links_single)
            varied_output = self.forward_with_params(params, varied_input[jnp.newaxis, ...])[0]
            relative_output = u2_mul(varied_output, u2_conj(base_output))
            return u2_log(relative_output).reshape(-1)

        jacobian = jax.jacfwd(flat_forward)(jnp.zeros(algebra_shape, dtype=links_single.dtype).reshape(-1))
        _, logabsdet = jnp.linalg.slogdet(jacobian)
        return logabsdet[jnp.newaxis]

    def _check_jacobian_if_requested(self, params: Params, links: Array) -> None:
        if not self.if_check_jac:
            return
        links = jnp.asarray(links)
        analytic = self.compute_jac_logdet_with_params(params, links[:1])
        autodiff = self.compute_jac_logdet_autodiff_with_params(params, links[:1])
        analytic_value = float(jax.block_until_ready(analytic[0]))
        autodiff_value = float(jax.block_until_ready(autodiff[0]))
        abs_diff = abs(autodiff_value - analytic_value)
        rel_diff = abs_diff / max(abs(analytic_value), 1e-12)
        if not np.isclose(autodiff_value, analytic_value, rtol=1e-4, atol=1e-6):
            print(
                "\nWarning: Jacobian log determinant difference "
                f"abs={abs_diff:.2e}, rel={rel_diff:.2e}"
            )
            print(">>> Jacobian is not correct!")
        else:
            print(f"\nJacobian log det (analytic): {analytic_value:.2e}, (autodiff): {autodiff_value:.2e}")
            print(">>> Jacobian is all good!")

    def compute_jac_logdet(self, links: Array) -> Array:
        return self.compute_jac_logdet_with_params(self.params, jnp.asarray(links))

    def new_action_with_params(self, params: Params, links_new: Array, beta: float) -> Array:
        links_ori = self.forward_with_params(params, links_new[jnp.newaxis, ...])[0]
        jac_logdet = self.compute_jac_logdet_with_params(params, links_new[jnp.newaxis, ...])[0]
        return action_from_field(links_ori, beta) - jac_logdet

    def compute_force(self, links: Array, beta: float, *, transformed: bool = False) -> Array:
        if transformed:
            self._check_jacobian_if_requested(self.params, links)

            def force_single(link_field: Array) -> Array:
                def varied_action(algebra: Array) -> Array:
                    return self.new_action_with_params(self.params, u2_mul(u2_exp(algebra), link_field), beta)

                return jax.grad(varied_action)(jnp.zeros((*link_field.shape[:-1], 4), dtype=link_field.dtype))

            return jax.vmap(force_single)(jnp.asarray(links))

        def original_force_single(link_field: Array) -> Array:
            def varied_action(algebra: Array) -> Array:
                return action_from_field(u2_mul(u2_exp(algebra), link_field), beta)

            return jax.grad(varied_action)(jnp.zeros((*link_field.shape[:-1], 4), dtype=link_field.dtype))

        return jax.vmap(original_force_single)(jnp.asarray(links))

    def loss_fn_with_params(self, params: Params, links_ori: Array, beta: float) -> Array:
        links_new = self.inverse_with_params(params, links_ori)

        def force_single(link_field: Array) -> Array:
            def varied_action(algebra: Array) -> Array:
                return self.new_action_with_params(params, u2_mul(u2_exp(algebra), link_field), beta)

            return jax.grad(varied_action)(jnp.zeros((*link_field.shape[:-1], 4), dtype=link_field.dtype))

        force = jax.vmap(force_single)(links_new)
        volume = self.lattice_size * self.lattice_size
        flat = force.reshape(force.shape[0], -1)
        return jnp.mean(
            jnp.linalg.norm(flat, ord=2, axis=1) / (volume**0.5)
            + jnp.linalg.norm(flat, ord=4, axis=1) / (volume**0.25)
            + jnp.linalg.norm(flat, ord=6, axis=1) / (volume ** (1 / 6))
            + jnp.linalg.norm(flat, ord=8, axis=1) / (volume ** (1 / 8))
        )

    def loss_fn(self, links: Array) -> Array:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        return self.loss_fn_with_params(self.params, jnp.asarray(links), self.train_beta)

    def _batches(self, data: np.ndarray, batch_size: int, rng: np.random.Generator, *, shuffle: bool) -> list[np.ndarray]:
        indices = np.arange(len(data))
        if shuffle:
            rng.shuffle(indices)
        return [data[indices[start : start + batch_size]] for start in range(0, len(indices), batch_size)]

    @staticmethod
    def _unreplicate(tree: Any) -> Any:
        return jax.tree_util.tree_map(lambda value: value[0], tree)

    @staticmethod
    def _shard_batch(batch: np.ndarray, n_devices: int) -> Array:
        if len(batch) % n_devices != 0:
            raise ValueError(f"Batch length {len(batch)} is not divisible by local_device_count={n_devices}")
        return jnp.asarray(batch.reshape(n_devices, len(batch) // n_devices, *batch.shape[1:]))

    def train(
        self,
        train_data: Array,
        test_data: Array,
        train_beta: float,
        *,
        n_epochs: int,
        batch_size: int,
        data_parallel: bool = False,
    ) -> None:
        self.train_beta = float(train_beta)
        train_np = np.asarray(train_data, dtype=np.float32)
        test_np = np.asarray(test_data, dtype=np.float32)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        devices = jax.local_devices()
        n_devices = len(devices)
        if data_parallel:
            if batch_size % n_devices != 0:
                raise ValueError(
                    f"--data_parallel requires batch_size ({batch_size}) to be divisible by "
                    f"jax.local_device_count() ({n_devices})"
                )
            print(f"data_parallel: enabled with local_device_count={n_devices}")

        @jax.jit
        def train_step(params: Params, opt_state: Any, batch: Array) -> tuple[Params, Any, Array]:
            loss, grads = jax.value_and_grad(self.loss_fn_with_params)(params, batch, self.train_beta)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        eval_step = jax.jit(lambda params, batch: self.loss_fn_with_params(params, batch, self.train_beta))
        train_step_parallel = jax.pmap(
            lambda params, opt_state, batch: self._parallel_train_step(params, opt_state, batch),
            axis_name="devices",
            devices=devices,
        )
        eval_step_parallel = jax.pmap(
            lambda params, batch: jax.lax.pmean(self.loss_fn_with_params(params, batch, self.train_beta), axis_name="devices"),
            axis_name="devices",
            devices=devices,
        )
        diag_step = jax.jit(
            lambda params, batch: self.inverse_with_diagnostics(
                params,
                batch,
                max_iter=int(self.hyperparams.get("inverse_max_iters", self.hyperparams.get("inverse_iters", 200))),
                tol=float(self.hyperparams.get("inverse_tol", 1e-6)),
            )[1]
        )
        rng = np.random.default_rng(0)
        train_losses: list[float] = []
        test_losses: list[float] = []
        best_loss = float("inf")
        bad_epochs = 0
        params_parallel = jax.device_put_replicated(self.params, devices) if data_parallel else None
        opt_state_parallel = jax.device_put_replicated(self.opt_state, devices) if data_parallel else None
        for epoch in tqdm(range(n_epochs), desc="Training epochs"):
            epoch_losses = []
            for batch in self._batches(train_np, batch_size, rng, shuffle=True):
                if data_parallel and len(batch) % n_devices == 0:
                    params_parallel, opt_state_parallel, loss = train_step_parallel(
                        params_parallel,
                        opt_state_parallel,
                        self._shard_batch(batch, n_devices),
                    )
                    loss = loss[0]
                else:
                    if data_parallel:
                        self.params = self._unreplicate(params_parallel)
                        self.opt_state = self._unreplicate(opt_state_parallel)
                    self.params, self.opt_state, loss = train_step(self.params, self.opt_state, jnp.asarray(batch))
                    if data_parallel:
                        params_parallel = jax.device_put_replicated(self.params, devices)
                        opt_state_parallel = jax.device_put_replicated(self.opt_state, devices)
                epoch_losses.append(float(loss))
            train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            eval_losses = []
            for batch in self._batches(test_np, batch_size, rng, shuffle=False):
                if data_parallel and len(batch) % n_devices == 0:
                    loss = eval_step_parallel(params_parallel, self._shard_batch(batch, n_devices))[0]
                else:
                    params = self._unreplicate(params_parallel) if data_parallel else self.params
                    loss = eval_step(params, jnp.asarray(batch))
                eval_losses.append(float(loss))
            test_loss = float(np.mean(eval_losses)) if eval_losses else train_loss
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.6f} test_loss={test_loss:.6f}")
            if data_parallel:
                self.params = self._unreplicate(params_parallel)
                self.opt_state = self._unreplicate(opt_state_parallel)
            probe_np = test_np[: min(8, len(test_np), batch_size)]
            diag = diag_step(self.params, jnp.asarray(probe_np))
            diag = {key: float(jax.block_until_ready(value)) for key, value in diag.items()}
            print(
                f"inverse_diag: "
                f"max_final_diff={diag['max_final_diff']:.2e} "
                f"mean_final_diff={diag['mean_final_diff']:.2e} "
                f"n_not_converged={diag['n_not_converged']:.0f} "
                f"round_trip_mean_abs_err={diag['round_trip_mean_abs_err']:.2e}"
            )
            if test_loss < best_loss:
                best_loss = test_loss
                bad_epochs = 0
                self.save_best_model(train_beta, epoch, test_loss)
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.hyperparams.get("early_stop_patience", 20)):
                    print("Early stopping")
                    break
        if data_parallel:
            self.params = self._unreplicate(params_parallel)
            self.opt_state = self._unreplicate(opt_state_parallel)
        tag = f"train_beta{format_beta(train_beta)}_{self.save_tag}"
        np.savetxt(self.dump_dir / f"train_loss_{tag}.csv", np.asarray(train_losses), fmt="%.8e")
        np.savetxt(self.dump_dir / f"test_loss_{tag}.csv", np.asarray(test_losses), fmt="%.8e")
        fig, ax = plt.subplots()
        ax.plot(train_losses, label="train")
        ax.plot(test_losses, label="test")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_dir / f"cnn_loss_{tag}.pdf", transparent=True)
        plt.close(fig)

    def _parallel_train_step(self, params: Params, opt_state: Any, batch: Array) -> tuple[Params, Any, Array]:
        loss, grads = jax.value_and_grad(self.loss_fn_with_params)(params, batch, self.train_beta)
        grads = jax.lax.pmean(grads, axis_name="devices")
        loss = jax.lax.pmean(loss, axis_name="devices")
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss
