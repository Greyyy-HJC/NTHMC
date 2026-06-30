"""JAX backend for U(1) FT-HMC evaluation.

This module intentionally mirrors the existing PyTorch U(1) implementation while
keeping the JAX path isolated.  Training and checkpoint format stay PyTorch-based;
JAX receives frozen CNN weights converted to plain arrays for evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
    raise ImportError("The JAX U(1) backend requires jax. Install the project with JAX support first.") from exc


Array = Any
Params = dict[str, Any]


def regularize(theta: Array) -> Array:
    """Wrap angles to [-pi, pi)."""
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - jnp.floor(theta_wrapped) - 0.5)


def plaq_from_field(theta: Array) -> Array:
    """Calculate plaquettes for a single U(1) field with shape [2, L, L]."""
    theta0, theta1 = theta[0], theta[1]
    return theta0 - theta1 - jnp.roll(theta0, shift=-1, axis=1) + jnp.roll(theta1, shift=-1, axis=0)


def plaq_from_field_batch(theta: Array) -> Array:
    """Calculate plaquettes for a batch with shape [batch, 2, L, L]."""
    theta0, theta1 = theta[:, 0], theta[:, 1]
    return theta0 - theta1 - jnp.roll(theta0, shift=-1, axis=2) + jnp.roll(theta1, shift=-1, axis=1)


def rect_from_field_batch(theta: Array) -> Array:
    """Calculate 1x2 and 2x1 rectangle loops for a batch of U(1) fields."""
    theta0, theta1 = theta[:, 0], theta[:, 1]
    rect0 = (
        theta0
        + jnp.roll(theta0, shift=-1, axis=1)
        + jnp.roll(theta1, shift=-2, axis=1)
        - jnp.roll(theta0, shift=(-1, -1), axis=(1, 2))
        - jnp.roll(theta0, shift=-1, axis=2)
        - theta1
    )
    rect1 = (
        theta0
        + jnp.roll(theta1, shift=-1, axis=1)
        + jnp.roll(theta1, shift=(-1, -1), axis=(1, 2))
        - jnp.roll(theta0, shift=-2, axis=2)
        - jnp.roll(theta1, shift=-1, axis=2)
        - theta1
    )
    return jnp.stack([rect0, rect1], axis=1)


def plaq_mean_from_field(theta: Array) -> Array:
    """Calculate the mean plaquette for a single field."""
    return jnp.mean(jnp.cos(regularize(plaq_from_field(theta))))


def topo_from_field(theta: Array) -> Array:
    """Calculate the integer-valued U(1) topological charge."""
    theta_p = regularize(plaq_from_field(theta))
    return jnp.floor(0.1 + jnp.sum(theta_p) / (2 * math.pi))


def action(theta: Array, beta: float) -> Array:
    """Wilson action for one U(1) field."""
    theta_p = regularize(plaq_from_field(theta))
    return -beta * jnp.sum(jnp.cos(theta_p))


def force(theta: Array, beta: float) -> Array:
    """Autodiff force for one U(1) field."""
    return jax.grad(lambda field: action(field, beta))(theta)


def _as_jax_array(tensor: torch.Tensor) -> Array:
    return jnp.asarray(tensor.detach().cpu().numpy())


def _clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _state_dict_to_params(state_dict: dict[str, torch.Tensor]) -> Params:
    state_dict = _clean_state_dict(state_dict)
    return {
        "conv_input": {
            "weight": _as_jax_array(state_dict["conv_input.weight"]),
            "bias": _as_jax_array(state_dict["conv_input.bias"]),
        },
        "conv_output": {
            "weight": _as_jax_array(state_dict["conv_output.weight"]),
            "bias": _as_jax_array(state_dict["conv_output.bias"]),
        },
    }


def torch_field_transform_to_jax_params(field_transform: Any) -> Params:
    """Convert an in-memory PyTorch U(1) field transformation to JAX params."""
    return {
        "model_tag": field_transform.model_tag,
        "subsets": [_state_dict_to_params(model.state_dict()) for model in field_transform.models],
    }


def load_checkpoint_params(
    checkpoint_path: str | Path,
    *,
    model_tag: str,
    n_subsets: int = 8,
    map_location: str = "cpu",
) -> Params:
    """Load a PyTorch U(1) checkpoint and return JAX-ready frozen weights."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    return {
        "model_tag": model_tag,
        "subsets": [
            _state_dict_to_params(checkpoint[f"model_state_dict_{index}"])
            for index in range(n_subsets)
        ],
    }


def _circular_conv2d_nchw(x: Array, layer: Params) -> Array:
    weight = layer["weight"]
    bias = layer["bias"]
    pad_h = weight.shape[2] // 2
    pad_w = weight.shape[3] // 2
    x = jnp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="wrap")
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + bias.reshape(1, -1, 1, 1)


def _gelu_exact(x: Array) -> Array:
    return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


def _apply_model(model_params: Params, model_tag: str, plaq_features: Array, rect_features: Array) -> tuple[Array, Array]:
    x = jnp.concatenate([plaq_features, rect_features], axis=1)
    x = _gelu_exact(_circular_conv2d_nchw(x, model_params["conv_input"]))
    x = _circular_conv2d_nchw(x, model_params["conv_output"])

    if model_tag == "base":
        x = jnp.arctan(x) / math.pi / 3
        plaq_sin_coeffs = x[:, :4]
        rect_sin_coeffs = x[:, 4:]
        return (
            jnp.concatenate([plaq_sin_coeffs, jnp.zeros_like(plaq_sin_coeffs)], axis=1),
            jnp.concatenate([rect_sin_coeffs, jnp.zeros_like(rect_sin_coeffs)], axis=1),
        )
    if model_tag == "addcos":
        return jnp.tanh(x[:, :8]) / 5, jnp.tanh(x[:, 8:]) / 40
    raise ValueError(f"Unsupported U(1) JAX model_tag: {model_tag!r}")


def _field_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, 2, lattice_size, lattice_size), dtype=bool)
    direction = 0 if index < 4 else 1
    parity = index % 4
    row_slice = slice(0, None, 2) if parity < 2 else slice(1, None, 2)
    col_slice = slice(0, None, 2) if parity in (0, 2) else slice(1, None, 2)
    mask[:, direction, row_slice, col_slice] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, 2, lattice_size, lattice_size))


def _plaq_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, lattice_size, lattice_size), dtype=bool)
    if index in (0, 1):
        mask[:, 1::2, :] = True
    elif index in (2, 3):
        mask[:, 0::2, :] = True
    elif index in (4, 6):
        mask[:, :, 1::2] = True
    elif index in (5, 7):
        mask[:, :, 0::2] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, lattice_size, lattice_size))


def _rect_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, 2, lattice_size, lattice_size), dtype=bool)
    if index == 0:
        mask[:, 1, 1::2, :] = True
        mask[:, 1, 0::2, 1::2] = True
    elif index == 1:
        mask[:, 1, 1::2, :] = True
        mask[:, 1, 0::2, 0::2] = True
    elif index == 2:
        mask[:, 1, 0::2, :] = True
        mask[:, 1, 1::2, 1::2] = True
    elif index == 3:
        mask[:, 1, 0::2, :] = True
        mask[:, 1, 1::2, 0::2] = True
    elif index == 4:
        mask[:, 0, :, 1::2] = True
        mask[:, 0, 1::2, 0::2] = True
    elif index == 5:
        mask[:, 0, :, 0::2] = True
        mask[:, 0, 1::2, 1::2] = True
    elif index == 6:
        mask[:, 0, :, 1::2] = True
        mask[:, 0, 0::2, 0::2] = True
    elif index == 7:
        mask[:, 0, :, 0::2] = True
        mask[:, 0, 0::2, 1::2] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, 2, lattice_size, lattice_size))


@dataclass(frozen=True)
class JaxU1FieldTransformation:
    """Frozen U(1) field transformation implemented in JAX."""

    params: Params
    lattice_size: int
    n_subsets: int = 8

    @property
    def model_tag(self) -> str:
        return str(self.params["model_tag"])

    def _compute_k0_k1(self, theta: Array, index: int, plaq: Array, rect: Array) -> tuple[Array, Array]:
        batch_size = theta.shape[0]
        plaq_mask = _plaq_mask(index, batch_size, self.lattice_size)
        rect_mask = _rect_mask(index, batch_size, self.lattice_size)
        plaq_masked = plaq * plaq_mask
        rect_masked = rect * rect_mask
        plaq_features = jnp.stack([jnp.sin(plaq_masked), jnp.cos(plaq_masked)], axis=1)
        rect_features = jnp.concatenate([jnp.sin(rect_masked), jnp.cos(rect_masked)], axis=1)
        return _apply_model(self.params["subsets"][index], self.model_tag, plaq_features, rect_features)

    @staticmethod
    def _plaq_angle_stack(plaq: Array) -> Array:
        return jnp.stack(
            [
                plaq,
                jnp.roll(plaq, shift=1, axis=2),
                plaq,
                jnp.roll(plaq, shift=1, axis=1),
            ],
            axis=1,
        )

    @staticmethod
    def _rect_angle_stack(rect: Array) -> Array:
        rect0 = rect[:, 0]
        rect1 = rect[:, 1]
        return jnp.stack(
            [
                jnp.roll(rect0, shift=1, axis=1),
                jnp.roll(rect0, shift=(1, 1), axis=(1, 2)),
                rect0,
                jnp.roll(rect0, shift=1, axis=2),
                jnp.roll(rect1, shift=1, axis=2),
                jnp.roll(rect1, shift=(1, 1), axis=(1, 2)),
                rect1,
                jnp.roll(rect1, shift=1, axis=1),
            ],
            axis=1,
        )

    def _plaq_phase_shift(self, k0: Array, plaq_angles: Array, theta: Array) -> Array:
        sin_signs = jnp.asarray([-1, 1, 1, -1], dtype=theta.dtype)
        cos_signs = -sin_signs
        plaq_stack = jnp.concatenate(
            [
                jnp.sin(plaq_angles) * sin_signs.reshape(1, 4, 1, 1),
                jnp.cos(plaq_angles) * cos_signs.reshape(1, 4, 1, 1),
            ],
            axis=1,
        )
        temp = k0 * plaq_stack
        return jnp.stack(
            [
                temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5],
                temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7],
            ],
            axis=1,
        )

    @staticmethod
    def _plaq_jac_shift(k0: Array, plaq_angles: Array) -> Array:
        temp = k0 * jnp.concatenate([-jnp.cos(plaq_angles), -jnp.sin(plaq_angles)], axis=1)
        return jnp.stack(
            [
                temp[:, 0] + temp[:, 1] + temp[:, 4] + temp[:, 5],
                temp[:, 2] + temp[:, 3] + temp[:, 6] + temp[:, 7],
            ],
            axis=1,
        )

    def _rect_phase_shift(self, k1: Array, rect_angles: Array, theta: Array) -> Array:
        signs = jnp.asarray([-1, 1, -1, 1, 1, -1, 1, -1], dtype=theta.dtype)
        rect_stack = jnp.concatenate(
            [
                jnp.sin(rect_angles) * signs.reshape(1, 8, 1, 1),
                jnp.cos(rect_angles) * (-signs).reshape(1, 8, 1, 1),
            ],
            axis=1,
        )
        temp = k1 * rect_stack
        return jnp.stack(
            [
                temp[:, 0:4].sum(axis=1) + temp[:, 8:12].sum(axis=1),
                temp[:, 4:8].sum(axis=1) + temp[:, 12:16].sum(axis=1),
            ],
            axis=1,
        )

    @staticmethod
    def _rect_jac_shift(k1: Array, rect_angles: Array) -> Array:
        temp = k1 * jnp.concatenate([-jnp.cos(rect_angles), -jnp.sin(rect_angles)], axis=1)
        return jnp.stack(
            [
                temp[:, 0:4].sum(axis=1) + temp[:, 8:12].sum(axis=1),
                temp[:, 4:8].sum(axis=1) + temp[:, 12:16].sum(axis=1),
            ],
            axis=1,
        )

    def ft_phase(self, theta: Array, index: int) -> Array:
        batch_size = theta.shape[0]
        plaq = plaq_from_field_batch(theta)
        rect = rect_from_field_batch(theta)
        plaq_angles = self._plaq_angle_stack(plaq)
        rect_angles = self._rect_angle_stack(rect)
        k0, k1 = self._compute_k0_k1(theta, index, plaq, rect)
        shift = self._plaq_phase_shift(k0, plaq_angles, theta) + self._rect_phase_shift(k1, rect_angles, theta)
        return shift * _field_mask(index, batch_size, self.lattice_size)

    def forward_batch(self, theta: Array) -> Array:
        theta_curr = theta
        for index in range(self.n_subsets):
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)
        return theta_curr

    def field_transformation(self, theta: Array) -> Array:
        return self.forward_batch(theta[jnp.newaxis, ...])[0]

    def compute_jac_logdet(self, theta: Array) -> Array:
        batch_size = theta.shape[0]
        log_det = jnp.zeros(batch_size, dtype=theta.dtype)
        theta_curr = theta

        for index in range(self.n_subsets):
            field_mask = _field_mask(index, batch_size, self.lattice_size)
            plaq = plaq_from_field_batch(theta_curr)
            rect = rect_from_field_batch(theta_curr)
            plaq_angles = self._plaq_angle_stack(plaq)
            rect_angles = self._rect_angle_stack(rect)
            k0, k1 = self._compute_k0_k1(theta_curr, index, plaq, rect)
            plaq_jac_shift = self._plaq_jac_shift(k0, plaq_angles) * field_mask
            rect_jac_shift = self._rect_jac_shift(k1, rect_angles) * field_mask
            log_det = log_det + jnp.log(1 + plaq_jac_shift + rect_jac_shift).sum(axis=(1, 2, 3))
            theta_curr = theta_curr + self.ft_phase(theta_curr, index)

        return log_det

    def original_action(self, theta: Array, beta: float) -> Array:
        return action(theta, beta)

    def new_action(self, theta_new: Array, beta: float) -> Array:
        theta_ori = self.field_transformation(theta_new)
        return self.original_action(theta_ori, beta) - self.compute_jac_logdet(theta_new[jnp.newaxis, ...])[0]

    def new_force(self, theta_new: Array, beta: float) -> Array:
        return jax.grad(lambda field: self.new_action(field, beta))(theta_new)


class JaxFTHMCResult(NamedTuple):
    theta: Array
    therm_plaq: Array
    therm_acceptance_rate: Array
    plaq: Array
    acceptance_rate: Array
    topo: Array
    hamiltonians: Array


def build_fthmc_chain(
    transform: JaxU1FieldTransformation,
    *,
    beta: float,
    n_thermalization: int,
    n_configs: int,
    n_steps: int,
    step_size: float,
) -> Any:
    """Return a jittable full FT-HMC chain function keyed by a JAX PRNG key."""
    lattice_size = transform.lattice_size
    lam = 0.1931833

    def new_action(theta: Array) -> Array:
        return transform.new_action(theta, beta)

    def new_force(theta: Array) -> Array:
        return jax.grad(new_action)(theta)

    def omelyan(theta: Array, pi: Array) -> tuple[Array, Array]:
        pi_next = pi - lam * step_size * new_force(theta)

        def body(step_index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            theta_next, body_pi = carry
            theta_next = theta_next + 0.5 * step_size * body_pi
            body_pi = body_pi - (1 - 2 * lam) * step_size * new_force(theta_next)
            theta_next = theta_next + 0.5 * step_size * body_pi
            body_pi = jax.lax.cond(
                step_index != n_steps - 1,
                lambda value: value - 2 * lam * step_size * new_force(theta_next),
                lambda value: value,
                body_pi,
            )
            return theta_next, body_pi

        theta_next, pi_next = jax.lax.fori_loop(0, n_steps, body, (theta, pi_next))
        pi_next = pi_next - lam * step_size * new_force(theta_next)
        return regularize(theta_next), pi_next

    def metropolis_step(theta: Array, key: Array) -> tuple[Array, Array, Array, Array]:
        key_pi, key_accept, key_next = jax.random.split(key, 3)
        pi = jax.random.normal(key_pi, theta.shape, dtype=theta.dtype)
        h_old = new_action(theta) + 0.5 * jnp.sum(pi**2)
        new_theta, new_pi = omelyan(theta, pi)
        h_new = new_action(new_theta) + 0.5 * jnp.sum(new_pi**2)
        accept_prob = jnp.exp(-(h_new - h_old))
        accepted = jax.random.uniform(key_accept, (), dtype=theta.dtype) < accept_prob
        theta_out = jnp.where(accepted, new_theta, theta)
        hamiltonian = jnp.where(accepted, h_new, h_old)
        return theta_out, key_next, accepted, hamiltonian

    def observable_plaq(theta: Array) -> Array:
        theta_ori = regularize(transform.field_transformation(theta))
        return plaq_mean_from_field(theta_ori)

    def observable_topo(theta: Array) -> Array:
        theta_ori = regularize(transform.field_transformation(theta))
        return topo_from_field(theta_ori)

    def chain(key: Array) -> JaxFTHMCResult:
        theta0 = jnp.zeros((2, lattice_size, lattice_size), dtype=jnp.float32)

        def therm_body(carry: tuple[Array, Array], _: Array) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            theta, carry_key = carry
            theta = regularize(theta)
            plaq_value = observable_plaq(theta)
            theta, carry_key, accepted, _ = metropolis_step(theta, carry_key)
            return (theta, carry_key), (plaq_value, accepted)

        (theta_thermalized, key), (therm_plaq, therm_accepted) = jax.lax.scan(
            therm_body,
            (theta0, key),
            xs=None,
            length=n_thermalization,
        )

        def run_body(carry: tuple[Array, Array], _: Array) -> tuple[tuple[Array, Array], tuple[Array, Array, Array, Array]]:
            theta, carry_key = carry
            theta, carry_key, accepted, hamiltonian = metropolis_step(theta, carry_key)
            theta = regularize(theta)
            return (theta, carry_key), (observable_plaq(theta), accepted, observable_topo(theta), hamiltonian)

        (theta_final, _), (plaq, accepted, topo, hamiltonians) = jax.lax.scan(
            run_body,
            (theta_thermalized, key),
            xs=None,
            length=n_configs,
        )
        return JaxFTHMCResult(
            theta=theta_final,
            therm_plaq=therm_plaq,
            therm_acceptance_rate=jnp.mean(therm_accepted.astype(jnp.float32)),
            plaq=plaq,
            acceptance_rate=jnp.mean(accepted.astype(jnp.float32)),
            topo=topo,
            hamiltonians=hamiltonians,
        )

    return chain
