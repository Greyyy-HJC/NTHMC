"""Pure JAX neural field transformation for 2D U(1)."""

from __future__ import annotations

import json
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
from nthmc.u1.models import apply_model, init_transform_params
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
    """JAX U(1) field transformation with Optax training and npz checkpoints."""

    def __init__(
        self,
        lattice_size: int,
        *,
        device: str = "cpu",
        n_subsets: int = 8,
        if_check_jac: bool = False,
        model_tag: str = "addcos",
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
        self.save_tag = save_tag or "opt"
        self.model_dir = Path(model_dir)
        self.plot_dir = Path(plot_dir)
        self.dump_dir = Path(dump_dir)
        self.train_beta: float | None = None
        self.hyperparams: dict[str, float] = {
            "init_std": 0.001,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "factor": 0.5,
            "patience": 5,
            "early_stop_patience": 20,
            "max_grad_norm": 10.0,
            "inverse_max_iters": 200,
            "inverse_tol": 1e-6,
        }
        if hyperparams:
            self.hyperparams.update(hyperparams)
        key = jax.random.PRNGKey(0)
        self.params = init_transform_params(
            key,
            model_tag,
            n_subsets,
            init_std=float(self.hyperparams["init_std"]),
        )
        self.optimizer = self._make_optimizer(float(self.hyperparams["lr"]))
        self.opt_state = self.optimizer.init(self.params)

    def _make_optimizer(self, lr: float) -> optax.GradientTransformation:
        chain = []
        max_norm = float(self.hyperparams.get("max_grad_norm", 0.0))
        if max_norm > 0:
            chain.append(optax.clip_by_global_norm(max_norm))
        chain.append(optax.adamw(lr, weight_decay=float(self.hyperparams.get("weight_decay", 0.0))))
        return optax.chain(*chain)

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
            "system": "2du1",
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
        params, metadata = load_checkpoint(self.checkpoint_path(train_beta), self._checkpoint_template())
        if metadata.get("model_tag") != self.model_tag:
            raise ValueError(f"Checkpoint model_tag={metadata.get('model_tag')!r} does not match {self.model_tag!r}")
        self.params = params
        self.opt_state = self.optimizer.init(self.params)
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

    def _compute_k0_k1(self, params: Params, theta: Array, index: int, plaq: Array, rect: Array) -> tuple[Array, Array]:
        batch_size = theta.shape[0]
        plaq_mask = get_plaq_mask(index, batch_size, self.lattice_size)
        rect_mask = get_rect_mask(index, batch_size, self.lattice_size)
        plaq_features = jnp.stack([jnp.sin(plaq * plaq_mask), jnp.cos(plaq * plaq_mask)], axis=1)
        rect_masked = rect * rect_mask
        rect_features = jnp.concatenate([jnp.sin(rect_masked), jnp.cos(rect_masked)], axis=1)
        return apply_model(params["subsets"][index], self.model_tag, plaq_features, rect_features)

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
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)
        k0, k1 = self._compute_k0_k1(params, theta, index, plaq, rect)
        shift = self._plaq_phase_shift(k0, self._plaq_angle_stack(plaq), theta) + self._rect_phase_shift(k1, self._rect_angle_stack(rect), theta)
        return shift * get_field_mask(index, batch_size, self.lattice_size)

    def forward_with_params(self, params: Params, theta: Array) -> Array:
        theta_curr = theta
        for index in range(self.n_subsets):
            theta_curr = theta_curr + self.ft_phase_with_params(params, theta_curr, index)
        return theta_curr

    def forward(self, theta: Array) -> Array:
        return self.forward_with_params(self.params, theta)

    def field_transformation(self, theta: Array) -> Array:
        return self.forward(jnp.asarray(theta)[jnp.newaxis, ...])[0]

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
        theta: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> Array:
        return self.inverse_with_diagnostics(params, theta, max_iter=max_iter, tol=tol)[0]

    def inverse_with_diagnostics(
        self,
        params: Params,
        theta: Array,
        *,
        max_iter: int | None = None,
        tol: float | None = None,
    ) -> tuple[Array, dict[str, Array]]:
        max_iter, tol = self._inverse_settings(max_iter, tol)
        theta_curr = theta
        final_diffs = []
        n_not_converged = jnp.asarray(0, dtype=jnp.int32)

        for index in reversed(range(self.n_subsets)):
            theta_iter = theta_curr
            init_diff = jnp.asarray(jnp.inf, dtype=theta.dtype)
            init_converged = jnp.asarray(False)

            def iteration_step(carry: tuple[Array, Array, Array], _: Array) -> tuple[tuple[Array, Array, Array], None]:
                theta_iter, diff, converged = carry
                theta_next = theta_curr - self.ft_phase_with_params(params, theta_iter, index)
                denom = jnp.clip(jnp.linalg.norm(theta_iter), min=1e-12)
                next_diff = jnp.linalg.norm(theta_next - theta_iter) / denom
                should_update = jnp.logical_not(converged)
                theta_iter = jnp.where(should_update, theta_next, theta_iter)
                diff = jnp.where(should_update, next_diff, diff)
                converged = jnp.logical_or(converged, next_diff < tol)
                return (theta_iter, diff, converged), None

            (theta_iter, final_diff, converged), _ = jax.lax.scan(
                iteration_step,
                (theta_iter, init_diff, init_converged),
                xs=None,
                length=max_iter,
            )
            final_diffs.append(final_diff)
            n_not_converged = n_not_converged + jnp.asarray(jnp.logical_not(converged), dtype=jnp.int32)
            theta_curr = theta_iter

        final_diffs = jnp.stack(final_diffs)
        recon = self.forward_with_params(params, theta_curr)
        round_trip_err = jnp.mean(jnp.abs(recon - theta))

        diagnostics = {
            "max_final_diff": jnp.max(final_diffs),
            "mean_final_diff": jnp.mean(final_diffs),
            "n_not_converged": n_not_converged,
            "round_trip_mean_abs_err": round_trip_err,
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
        batch_size = theta.shape[0]
        log_det = jnp.zeros(batch_size, dtype=theta.dtype)
        theta_curr = theta
        for index in range(self.n_subsets):
            field_mask = get_field_mask(index, batch_size, self.lattice_size)
            plaq = plaq_from_field_batch(theta_curr)
            rect = rect_from_field_batch(theta_curr)
            k0, k1 = self._compute_k0_k1(params, theta_curr, index, plaq, rect)
            plaq_jac = self._plaq_jac_shift(k0, self._plaq_angle_stack(plaq)) * field_mask
            rect_jac = self._rect_jac_shift(k1, self._rect_angle_stack(rect)) * field_mask
            log_det = log_det + jnp.log(jnp.clip(1 + plaq_jac + rect_jac, min=1e-8)).sum(axis=(1, 2, 3))
            theta_curr = theta_curr + self.ft_phase_with_params(params, theta_curr, index)
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

    def loss_fn_with_params(self, params: Params, theta_ori: Array, beta: float) -> Array:
        theta_new = self.inverse_with_params(params, theta_ori)
        force_new = jax.vmap(jax.grad(lambda x: self.new_action_with_params(params, x, beta)))(theta_new)
        volume = self.lattice_size * self.lattice_size
        force_flat = force_new.reshape(force_new.shape[0], -1)
        return jnp.mean(
            jnp.linalg.norm(force_flat, ord=2, axis=1) / (volume**0.5)
            + jnp.linalg.norm(force_flat, ord=4, axis=1) / (volume**0.25)
            + jnp.linalg.norm(force_flat, ord=6, axis=1) / (volume ** (1 / 6))
            + jnp.linalg.norm(force_flat, ord=8, axis=1) / (volume ** (1 / 8))
        )

    def loss_fn(self, theta_ori: Array) -> Array:
        if self.train_beta is None:
            raise RuntimeError("train_beta is not set")
        return self.loss_fn_with_params(self.params, jnp.asarray(theta_ori), self.train_beta)

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
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

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
