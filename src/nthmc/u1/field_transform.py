"""Pure JAX neural field transformation for 2D U(1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nthmc.core.checkpoint import load_checkpoint
from nthmc.u1.models import choose_model, init_transform_params
from nthmc.u1.u1_observables import (
    action,
    format_beta,
    get_field_mask,
    get_plaq_mask,
    get_rect_mask,
    plaq_from_field_batch,
    rect_from_field_batch,
    regularize,
)

Array = Any
Params = dict[str, Any]


class FieldTransformation:
    """JAX U(1) field transformation for evaluation and HMC."""

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
        hyperparams: dict[str, float] | None = None,
    ) -> None:
        self.lattice_size = lattice_size
        self.device = device
        self.n_subsets = n_subsets
        self.if_check_jac = if_check_jac
        self.model_tag = model_tag
        self.model = choose_model(model_tag)
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.hyperparams: dict[str, float] = {
            "init_std": 0.001,
            "inverse_max_iters": 200,
            "inverse_tol": 1e-6,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)
        key = jax.random.PRNGKey(0)
        self.params = init_transform_params(
            key,
            self.model,
            n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )
        self._field_masks = jnp.stack(
            [get_field_mask(index, 1, self.lattice_size)[0] for index in range(self.n_subsets)]
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
        params, metadata = load_checkpoint(self.checkpoint_path(train_beta), self._checkpoint_template())
        if metadata.get("model_tag") != self.model_tag:
            raise ValueError(f"Checkpoint model_tag={metadata.get('model_tag')!r} does not match {self.model_tag!r}")
        self.params = params
        print(f"Loaded JAX checkpoint from epoch {metadata.get('epoch')} with loss {metadata.get('loss')}")

    @staticmethod
    def _plaq_angle_stack(plaq: Array) -> Array:
        return jnp.stack([plaq, jnp.roll(plaq, 1, 2), plaq, jnp.roll(plaq, 1, 1)], axis=1)

    @staticmethod
    def _rect_angle_stack(rect: Array) -> Array:
        rect0 = rect[:, 0]
        rect1 = rect[:, 1]
        return jnp.stack(
            [
                jnp.roll(rect0, 1, 1),
                jnp.roll(rect0, (1, 1), (1, 2)),
                rect0,
                jnp.roll(rect0, 1, 2),
                jnp.roll(rect1, 1, 2),
                jnp.roll(rect1, (1, 1), (1, 2)),
                rect1,
                jnp.roll(rect1, 1, 1),
            ],
            axis=1,
        )

    @staticmethod
    def _stack_subset_params(params: Params) -> Params:
        return jax.tree_util.tree_map(lambda *values: jnp.stack(values), *params["subsets"])

    def _compute_k0_k1(
        self,
        subset_params: Params,
        theta: Array,
        plaq: Array,
        rect: Array,
        plaq_mask: Array,
        rect_mask: Array,
    ) -> tuple[Array, Array]:
        batch_size = theta.shape[0]
        plaq_mask = jnp.broadcast_to(plaq_mask, (batch_size, *plaq_mask.shape))
        rect_mask = jnp.broadcast_to(rect_mask, (batch_size, *rect_mask.shape))
        plaq_features = jnp.stack([jnp.sin(plaq * plaq_mask), jnp.cos(plaq * plaq_mask)], axis=1)
        rect_masked = rect * rect_mask
        rect_features = jnp.concatenate([jnp.sin(rect_masked), jnp.cos(rect_masked)], axis=1)
        return self.model.apply(subset_params, plaq_features, rect_features)

    @staticmethod
    def _plaq_phase_shift(k0: Array, plaq_angles: Array, theta: Array) -> Array:
        sin_signs = jnp.asarray([-1, 1, 1, -1], dtype=theta.dtype)
        cos_signs = -sin_signs
        stack = jnp.concatenate(
            [jnp.sin(plaq_angles) * sin_signs.reshape(1, 4, 1, 1), jnp.cos(plaq_angles) * cos_signs.reshape(1, 4, 1, 1)],
            axis=1,
        )
        temp = k0 * stack
        return jnp.stack([temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5], temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7]], axis=1)

    @staticmethod
    def _plaq_jac_shift(k0: Array, plaq_angles: Array) -> Array:
        temp = k0 * jnp.concatenate([-jnp.cos(plaq_angles), -jnp.sin(plaq_angles)], axis=1)
        return jnp.stack([temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5], temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7]], axis=1)

    @staticmethod
    def _rect_phase_shift(k1: Array, rect_angles: Array, theta: Array) -> Array:
        signs = jnp.asarray([-1, 1, -1, 1, 1, -1, 1, -1], dtype=theta.dtype)
        stack = jnp.concatenate(
            [jnp.sin(rect_angles) * signs.reshape(1, 8, 1, 1), jnp.cos(rect_angles) * (-signs).reshape(1, 8, 1, 1)],
            axis=1,
        )
        temp = k1 * stack
        return jnp.stack([temp[:, 0:4].sum(1) + temp[:, 8:12].sum(1), temp[:, 4:8].sum(1) + temp[:, 12:16].sum(1)], axis=1)

    @staticmethod
    def _rect_jac_shift(k1: Array, rect_angles: Array) -> Array:
        temp = k1 * jnp.concatenate([-jnp.cos(rect_angles), -jnp.sin(rect_angles)], axis=1)
        return jnp.stack([temp[:, 0:4].sum(1) + temp[:, 8:12].sum(1), temp[:, 4:8].sum(1) + temp[:, 12:16].sum(1)], axis=1)

    def ft_phase_with_params(self, params: Params, theta: Array, index: int) -> Array:
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)
        k0, k1 = self._compute_k0_k1(
            params["subsets"][index], theta, plaq, rect, self._plaq_masks[index], self._rect_masks[index]
        )
        shift = self._plaq_phase_shift(k0, self._plaq_angle_stack(plaq), theta) + self._rect_phase_shift(k1, self._rect_angle_stack(rect), theta)
        return shift * self._field_masks[index]

    def _ft_phase_with_subset(
        self,
        subset_params: Params,
        theta: Array,
        field_mask: Array,
        plaq_mask: Array,
        rect_mask: Array,
    ) -> Array:
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)
        k0, k1 = self._compute_k0_k1(subset_params, theta, plaq, rect, plaq_mask, rect_mask)
        shift = self._plaq_phase_shift(k0, self._plaq_angle_stack(plaq), theta)
        shift = shift + self._rect_phase_shift(k1, self._rect_angle_stack(rect), theta)
        return shift * field_mask

    def forward_with_params(self, params: Params, theta: Array) -> Array:
        subset_params = self._stack_subset_params(params)

        def apply_subset(theta_curr: Array, inputs: tuple[Params, Array, Array, Array]) -> tuple[Array, None]:
            current_params, field_mask, plaq_mask, rect_mask = inputs
            theta_curr = theta_curr + self._ft_phase_with_subset(
                current_params, theta_curr, field_mask, plaq_mask, rect_mask
            )
            return theta_curr, None

        theta_out, _ = jax.lax.scan(
            jax.checkpoint(apply_subset, prevent_cse=False),
            theta,
            (subset_params, self._field_masks, self._plaq_masks, self._rect_masks),
        )
        return theta_out

    def forward(self, theta: Array) -> Array:
        return self.forward_with_params(self.params, theta)

    def field_transformation(self, theta: Array) -> Array:
        return self.forward(jnp.asarray(theta)[jnp.newaxis, ...])[0]

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

    def _inverse_subset_dynamic(
        self,
        target: Array,
        subset_inputs: tuple[Params, Array, Array, Array],
        valid: Array,
        max_iter: int,
        tol: Array,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        subset_params, field_mask, plaq_mask, rect_mask = subset_inputs
        init = (
            target,
            jnp.where(valid, jnp.inf, 0).astype(target.dtype),
            jnp.logical_not(valid),
            jnp.zeros(valid.shape, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

        def condition(carry: tuple[Array, Array, Array, Array, Array]) -> Array:
            _, _, converged, _, step = carry
            return jnp.logical_and(step < max_iter, jnp.any(jnp.logical_not(converged)))

        def update(carry: tuple[Array, Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array]:
            theta_iter, diff, converged, iterations, step = carry
            active = jnp.logical_not(converged)
            theta_next = target - self._ft_phase_with_subset(
                subset_params, theta_iter, field_mask, plaq_mask, rect_mask
            )
            reduce_axes = tuple(range(1, theta_iter.ndim))
            denominator = jnp.clip(jnp.sqrt(jnp.sum(theta_iter**2, axis=reduce_axes)), min=1e-12)
            next_diff = jax.lax.stop_gradient(
                jnp.sqrt(jnp.sum((theta_next - theta_iter) ** 2, axis=reduce_axes)) / denominator
            )
            broadcast_active = active.reshape((active.shape[0],) + (1,) * (theta_iter.ndim - 1))
            theta_iter = jnp.where(broadcast_active, theta_next, theta_iter)
            diff = jnp.where(active, next_diff, diff)
            converged = jnp.logical_or(converged, next_diff < tol)
            iterations = iterations + active.astype(jnp.int32)
            return theta_iter, diff, converged, iterations, step + 1

        result, final_diff, converged, iterations, _ = jax.lax.while_loop(condition, update, init)
        return result, (final_diff, converged, iterations)

    def _inverse_dynamic_scan(
        self, params: Params, theta: Array, valid: Array, max_iter: int, tol: Array
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        subset_params = self._stack_subset_params(params)

        def solve_subset(theta_curr: Array, inputs: tuple[Params, Array, Array, Array]):
            return self._inverse_subset_dynamic(theta_curr, inputs, valid, max_iter, tol)

        return jax.lax.scan(
            solve_subset,
            theta,
            (subset_params, self._field_masks, self._plaq_masks, self._rect_masks),
            reverse=True,
        )

    def inverse_with_diagnostics(
        self,
        params: Params,
        theta: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
        sample_mask: Array | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        max_iter, resolved_tol = self._inverse_settings(max_iter, tol)
        valid = self._valid_sample_mask(theta, sample_mask)
        theta_curr, (final_diffs, converged, iterations) = self._inverse_dynamic_scan(
            params, theta, valid, max_iter, jnp.asarray(resolved_tol, theta.dtype)
        )
        recon = self.forward_with_params(params, theta_curr)
        reduce_axes = tuple(range(1, theta.ndim))
        round_trip_per_sample = jnp.mean(jnp.abs(recon - theta), axis=reduce_axes)
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
        return theta_curr, diagnostics

    def inverse(
        self,
        theta: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
        return_diagnostics: bool = False,
        **_: Any,
    ) -> Array | tuple[Array, dict[str, Array]]:
        theta = jnp.asarray(theta)
        if theta.ndim == 3:
            out, diagnostics = self.inverse_with_diagnostics(self.params, theta[jnp.newaxis, ...], max_iter=max_iter, tol=tol)
            out = out[0]
        else:
            out, diagnostics = self.inverse_with_diagnostics(self.params, theta, max_iter=max_iter, tol=tol)
        return (out, diagnostics) if return_diagnostics else out

    def inverse_field_transformation(self, theta: Array) -> Array:
        return self.inverse(jnp.asarray(theta)[jnp.newaxis, ...])[0]

    def compute_jac_logdet_with_params(self, params: Params, theta: Array) -> Array:
        subset_params = self._stack_subset_params(params)

        def accumulate(carry: tuple[Array, Array], inputs: tuple[Params, Array, Array, Array]):
            theta_curr, log_det = carry
            current_params, field_mask, plaq_mask, rect_mask = inputs
            plaq = plaq_from_field_batch(theta_curr)
            rect = rect_from_field_batch(theta_curr)
            k0, k1 = self._compute_k0_k1(current_params, theta_curr, plaq, rect, plaq_mask, rect_mask)
            plaq_jac = self._plaq_jac_shift(k0, self._plaq_angle_stack(plaq)) * field_mask
            rect_jac = self._rect_jac_shift(k1, self._rect_angle_stack(rect)) * field_mask
            log_det = log_det + jnp.log(jnp.clip(1 + plaq_jac + rect_jac, min=1e-8)).sum(axis=(1, 2, 3))
            theta_curr = theta_curr + self._ft_phase_with_subset(
                current_params, theta_curr, field_mask, plaq_mask, rect_mask
            )
            return (theta_curr, log_det), None

        (_, log_det), _ = jax.lax.scan(
            jax.checkpoint(accumulate, prevent_cse=False),
            (theta, jnp.zeros(theta.shape[0], dtype=theta.dtype)),
            (subset_params, self._field_masks, self._plaq_masks, self._rect_masks),
        )
        return log_det

    def compute_jac_logdet_autodiff_with_params(self, params: Params, theta: Array) -> Array:
        """Full-field autodiff Jacobian check for the first batch item only."""
        theta_single = jnp.asarray(theta)[0]
        flat_shape = theta_single.shape

        def flat_forward(flat_theta: Array) -> Array:
            field = flat_theta.reshape(flat_shape)
            transformed = self.forward_with_params(params, field[jnp.newaxis, ...])[0]
            return transformed.reshape(-1)

        jacobian = jax.jacfwd(flat_forward)(theta_single.reshape(-1))
        _, logabsdet = jnp.linalg.slogdet(jacobian)
        return logabsdet[jnp.newaxis]

    def _check_jacobian_if_requested(self, params: Params, theta: Array) -> None:
        if not self.if_check_jac:
            return
        theta = jnp.asarray(theta)
        manual = self.compute_jac_logdet_with_params(params, theta[:1])
        autodiff = self.compute_jac_logdet_autodiff_with_params(params, theta[:1])
        manual_value = float(jax.block_until_ready(manual[0]))
        autodiff_value = float(jax.block_until_ready(autodiff[0]))
        abs_diff = abs(autodiff_value - manual_value)
        rel_diff = abs_diff / max(abs(manual_value), 1e-12)
        if not np.isclose(autodiff_value, manual_value, rtol=1e-4, atol=1e-6):
            print(
                "\nWarning: Jacobian log determinant difference "
                f"abs={abs_diff:.2e}, rel={rel_diff:.2e}"
            )
            print(">>> Jacobian is not correct!")
        else:
            print(f"\nJacobian log det (analytic): {manual_value:.2e}, (autodiff): {autodiff_value:.2e}")
            print(">>> Jacobian is all good!")

    def compute_jac_logdet(self, theta: Array) -> Array:
        return self.compute_jac_logdet_with_params(self.params, jnp.asarray(theta))

    def new_action_with_params(self, params: Params, theta_new: Array, beta: float) -> Array:
        theta_ori = self.forward_with_params(params, theta_new[jnp.newaxis, ...])[0]
        return action(theta_ori, beta) - self.compute_jac_logdet_with_params(params, theta_new[jnp.newaxis, ...])[0]

    def compute_force(self, theta: Array, beta: float, *, transformed: bool = False) -> Array:
        if transformed:
            self._check_jacobian_if_requested(self.params, theta)
            return jax.vmap(jax.grad(lambda x: self.new_action_with_params(self.params, x, beta)))(jnp.asarray(theta))
        return jax.vmap(jax.grad(lambda x: action(x, beta)))(jnp.asarray(theta))
