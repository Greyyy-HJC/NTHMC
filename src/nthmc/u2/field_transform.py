"""JAX neural field transformation for 2D U(2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np

from nthmc.core.checkpoint import load_checkpoint
from nthmc.u2.models import choose_model, init_transform_params
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
LoopAxis = Union[int, tuple[int, int], None]
LoopToken = tuple[int, LoopAxis, LoopAxis, bool]


class FieldTransformation:
    """JAX U(2) field transformation for evaluation and HMC."""

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
        self.model = choose_model(model_tag)
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.hyperparams: dict[str, Any] = {
            "init_std": 0.001,
            "inverse_max_iters": 200,
            "inverse_tol": 1e-6,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)
        self.params = init_transform_params(
            jax.random.PRNGKey(0),
            self.model,
            n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )
        self._link_masks = jnp.stack(
            [get_link_mask(index, 1, self.lattice_size)[0] for index in range(self.n_subsets)]
        )
        self._plaq_masks = jnp.stack(
            [get_plaq_mask(index, 1, self.lattice_size)[0] for index in range(self.n_subsets)]
        )
        self._rect_masks = jnp.stack(
            [get_rect_mask(index, 1, self.lattice_size)[0] for index in range(self.n_subsets)]
        )

    def _checkpoint_template(self) -> Params:
        return init_transform_params(
            jax.random.PRNGKey(0),
            self.model,
            self.n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )

    def checkpoint_path(self, train_beta: float) -> Path:
        return self.model_dir / f"best_model_train_beta{format_beta(train_beta)}_{self.save_tag}.npz"

    def load_best_model(self, train_beta: float) -> None:
        self.params, metadata = load_checkpoint(self.checkpoint_path(train_beta), self._checkpoint_template())
        print(f"Loaded JAX U(2) checkpoint from epoch {metadata.get('epoch')} with loss {metadata.get('loss')}")

    @staticmethod
    def _features_from_loops(loops: Array) -> Array:
        loops = u2_normalize(loops)
        phase = loops[..., :1]
        q0 = loops[..., 1:2]
        cos_phase = jnp.cos(phase)
        sin_phase = jnp.sin(phase)
        trace_sq_factor = 2 * (2 * q0**2 - 1)
        features = jnp.concatenate(
            [
                q0 * cos_phase,
                q0 * sin_phase,
                cos_phase,
                sin_phase,
                trace_sq_factor * jnp.cos(2 * phase),
                trace_sq_factor * jnp.sin(2 * phase),
            ],
            axis=-1,
        )
        return jnp.moveaxis(features, -1, 1)

    @staticmethod
    def _masked_loops(loops: Array, mask: Array) -> Array:
        return jnp.where(mask, loops, identity_like(loops))

    @staticmethod
    def _stack_subset_params(params: Params) -> Params:
        return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *params["subsets"])

    def _cnn_features_with_masks(self, links: Array, plaq_mask: Array, rect_mask: Array) -> tuple[Array, Array]:
        batch_size = links.shape[0]
        plaquettes = plaquette_from_field_batch(links)
        rectangles = rectangle_from_field_batch(links)
        plaq_mask = jnp.broadcast_to(plaq_mask, (batch_size, *plaq_mask.shape))
        rect_mask = jnp.broadcast_to(rect_mask, (batch_size, *rect_mask.shape))
        plaq_features = self._features_from_loops(self._masked_loops(plaquettes, plaq_mask))
        rect_features = self._features_from_loops(
            self._masked_loops(rectangles, rect_mask)
        ).swapaxes(1, 2)
        rect_features = rect_features.reshape(batch_size, 12, self.lattice_size, self.lattice_size)
        return plaq_features, rect_features

    def _cnn_features(self, links: Array, index: int) -> tuple[Array, Array]:
        return self._cnn_features_with_masks(links, self._plaq_masks[index], self._rect_masks[index])

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
        return self._compute_delta_subset(
            params["subsets"][index],
            links,
            self._link_masks[index],
            self._plaq_masks[index],
            self._rect_masks[index],
        )

    def _compute_delta_subset(
        self,
        subset_params: Params,
        links: Array,
        link_mask: Array,
        plaq_mask: Array,
        rect_mask: Array,
    ) -> Array:
        batch_size = links.shape[0]
        plaq_features, rect_features = self._cnn_features_with_masks(links, plaq_mask, rect_mask)
        plaq_coeffs, rect_coeffs = self.model.apply(subset_params, plaq_features, rect_features)
        plaq_coeffs = plaq_coeffs.reshape(batch_size, 4, 4, self.lattice_size, self.lattice_size)
        rect_coeffs = rect_coeffs.reshape(batch_size, 8, 4, self.lattice_size, self.lattice_size)
        delta = self._plaq_delta(plaq_coeffs, self._plaq_loop_stack(links))
        delta = delta + self._rect_delta(rect_coeffs, self._rect_loop_stack(links))
        return delta * link_mask

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
        mask: Array,
        plaq_coeffs: Array,
        rect_coeffs: Array,
        delta: Array,
    ) -> Array:
        batch_size = links.shape[0]
        mask = jnp.broadcast_to(mask, (batch_size, *mask.shape))
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
        subset_params = self._stack_subset_params(params)

        def apply_subset(links_curr: Array, inputs: tuple[Params, Array, Array, Array]) -> tuple[Array, None]:
            current_params, link_mask, plaq_mask, rect_mask = inputs
            delta = self._compute_delta_subset(current_params, links_curr, link_mask, plaq_mask, rect_mask)
            links_curr = u2_normalize(u2_mul(u2_exp(delta), links_curr))
            return links_curr, None

        links_out, _ = jax.lax.scan(
            jax.checkpoint(apply_subset, prevent_cse=False),
            u2_normalize(links),
            (subset_params, self._link_masks, self._plaq_masks, self._rect_masks),
        )
        return u2_normalize(links_out)

    def forward(self, links: Array) -> Array:
        links = jnp.asarray(links)
        if links.ndim == 4:
            return self.forward_with_params(self.params, links[jnp.newaxis, ...])[0]
        return self.forward_with_params(self.params, links)

    def field_transformation(self, links: Array) -> Array:
        return self.forward(jnp.asarray(links))

    def _inverse_settings(self, max_iter: int | None = None, tol: Array | None = None) -> tuple[int, Array]:
        resolved_max_iter = int(
            max_iter
            or self.hyperparams.get("inverse_max_iters", self.hyperparams.get("inverse_iters", 200))
        )
        resolved_tol = self.hyperparams.get("inverse_tol", 1e-6) if tol is None else tol
        return resolved_max_iter, resolved_tol

    @staticmethod
    def _valid_sample_mask(batch: Array, sample_mask: Array | None) -> Array:
        if sample_mask is None:
            return jnp.ones((batch.shape[0],), dtype=bool)
        return jnp.asarray(sample_mask, dtype=bool)

    @staticmethod
    def _masked_batch_mean(values: Array, valid: Array) -> Array:
        weights = valid.astype(values.dtype)
        return jnp.sum(values * weights) / jnp.clip(jnp.sum(weights), min=1)

    def _inverse_update(
        self,
        target: Array,
        links_iter: Array,
        subset_inputs: tuple[Params, Array, Array, Array],
    ) -> tuple[Array, Array]:
        subset_params, link_mask, plaq_mask, rect_mask = subset_inputs
        delta = self._compute_delta_subset(subset_params, links_iter, link_mask, plaq_mask, rect_mask)
        links_next = u2_normalize(u2_mul(u2_exp(-delta), target))
        relative = u2_log(u2_mul(links_next, u2_conj(links_iter)))
        reduce_axes = tuple(range(1, links_iter.ndim))
        denominator = jnp.clip(jnp.sqrt(jnp.sum(u2_log(links_iter) ** 2, axis=reduce_axes)), min=1e-12)
        next_diff = jax.lax.stop_gradient(
            jnp.sqrt(jnp.sum(relative**2, axis=reduce_axes)) / denominator
        )
        return links_next, next_diff

    def _inverse_subset_dynamic(
        self,
        target: Array,
        subset_inputs: tuple[Params, Array, Array, Array],
        valid: Array,
        max_iter: int,
        tol: Array,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        init = (
            target,
            jnp.where(valid, jnp.inf, 0).astype(target.dtype),
            jnp.logical_not(valid),
            jnp.zeros(valid.shape, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def condition(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
            return jnp.logical_and(carry[4] < max_iter, jnp.any(jnp.logical_not(carry[2])))

        def update(carry: tuple[Array, Array, Array, Array, Array]):
            links_iter, diff, converged, iterations, step = carry
            active = jnp.logical_not(converged)
            links_next, next_diff = self._inverse_update(target, links_iter, subset_inputs)
            broadcast_active = active.reshape((active.shape[0],) + (1,) * (links_iter.ndim - 1))
            links_iter = jnp.where(broadcast_active, links_next, links_iter)
            diff = jnp.where(active, next_diff, diff)
            converged = jnp.logical_or(converged, next_diff < tol)
            iterations = iterations + active.astype(jnp.int32)
            return links_iter, diff, converged, iterations, step + 1

        result, final_diff, converged, iterations, _ = jax.lax.while_loop(condition, update, init)
        return result, (final_diff, converged, iterations)

    def _inverse_dynamic_scan(
        self, params: Params, links: Array, valid: Array, max_iter: int, tol: Array
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        subset_params = self._stack_subset_params(params)
        return jax.lax.scan(
            lambda current, inputs: self._inverse_subset_dynamic(current, inputs, valid, max_iter, tol),
            u2_normalize(links),
            (subset_params, self._link_masks, self._plaq_masks, self._rect_masks),
            reverse=True,
        )

    def inverse_with_diagnostics(
        self,
        params: Params,
        links: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
        sample_mask: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        max_iter, resolved_tol = self._inverse_settings(max_iter, tol)
        valid = self._valid_sample_mask(links, sample_mask)
        out, (final_diffs, converged, iterations) = self._inverse_dynamic_scan(
            params, links, valid, max_iter, jnp.asarray(resolved_tol, links.dtype)
        )
        out = u2_normalize(out)
        recon = self.forward_with_params(params, out)
        relative = u2_log(u2_mul(recon, u2_conj(links)))
        reduce_axes = tuple(range(1, relative.ndim))
        round_trip_per_sample = jnp.mean(jnp.abs(relative), axis=reduce_axes)
        valid_matrix = jnp.broadcast_to(valid, final_diffs.shape)
        valid_diffs = jnp.where(valid_matrix, final_diffs, 0)
        valid_iterations = jnp.where(valid_matrix, iterations, 0)
        valid_count = jnp.clip(jnp.sum(valid_matrix), min=1)
        diagnostics = {
            "max_final_diff": jnp.max(valid_diffs),
            "mean_final_diff": jnp.sum(valid_diffs) / valid_count,
            "mean_iterations": jnp.sum(valid_iterations) / valid_count,
            "max_iterations": jnp.max(valid_iterations),
            "n_not_converged": jnp.sum(jnp.logical_and(jnp.logical_not(converged), valid_matrix)),
            "round_trip_mean_abs_err": self._masked_batch_mean(round_trip_per_sample, valid),
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
        return self._subset_coeffs_delta_with_masks(
            params["subsets"][index],
            links,
            self._link_masks[index],
            self._plaq_masks[index],
            self._rect_masks[index],
        )

    def _subset_coeffs_delta_with_masks(
        self,
        subset_params: Params,
        links: Array,
        link_mask: Array,
        plaq_mask: Array,
        rect_mask: Array,
    ) -> tuple[Array, Array, Array]:
        batch_size = links.shape[0]
        plaq_features, rect_features = self._cnn_features_with_masks(links, plaq_mask, rect_mask)
        plaq_coeffs, rect_coeffs = self.model.apply(subset_params, plaq_features, rect_features)
        plaq_coeffs = plaq_coeffs.reshape(batch_size, 4, 4, self.lattice_size, self.lattice_size)
        rect_coeffs = rect_coeffs.reshape(batch_size, 8, 4, self.lattice_size, self.lattice_size)
        delta = self._plaq_delta(plaq_coeffs, self._plaq_loop_stack(links))
        delta = delta + self._rect_delta(rect_coeffs, self._rect_loop_stack(links))
        delta = delta * link_mask
        return plaq_coeffs, rect_coeffs, delta

    def compute_jac_logdet_with_params(self, params: Params, links: Array) -> Array:
        subset_params = self._stack_subset_params(params)

        def accumulate(carry: tuple[Array, Array], inputs: tuple[Params, Array, Array, Array]):
            links_curr, logdet = carry
            current_params, link_mask, plaq_mask, rect_mask = inputs
            plaq_coeffs, rect_coeffs, delta = self._subset_coeffs_delta_with_masks(
                current_params, links_curr, link_mask, plaq_mask, rect_mask
            )
            jacobian_blocks = self._layer_jacobian_blocks(
                links_curr, link_mask, plaq_coeffs, rect_coeffs, delta
            )
            mask = jnp.broadcast_to(link_mask, (links.shape[0], *link_mask.shape))[..., 0]
            blocks = jacobian_blocks.reshape(links.shape[0], -1, 4, 4)
            mask_flat = mask.reshape(links.shape[0], -1)
            identity = jnp.eye(4, dtype=links.dtype)
            blocks = jnp.where(mask_flat[..., jnp.newaxis, jnp.newaxis], blocks, identity)
            _, local_logdet = jnp.linalg.slogdet(blocks)
            logdet = logdet + jnp.sum(local_logdet, axis=1)
            links_curr = u2_normalize(u2_mul(u2_exp(delta), links_curr))
            return (links_curr, logdet), None

        (_, logdet), _ = jax.lax.scan(
            jax.checkpoint(accumulate, prevent_cse=False),
            (u2_normalize(links), jnp.zeros((links.shape[0],), dtype=links.dtype)),
            (subset_params, self._link_masks, self._plaq_masks, self._rect_masks),
        )
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
