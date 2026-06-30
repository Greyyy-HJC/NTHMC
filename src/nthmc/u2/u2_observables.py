"""JAX U(2) observables with split U(1) phase and SU(2) quaternion links."""

from __future__ import annotations

import math
import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import i0, i1, iv

Array = Any


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def format_beta(beta: float) -> str:
    value = float(beta)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def regularize_phase(theta: Array) -> Array:
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - jnp.floor(theta_wrapped) - 0.5)


def quaternion_normalize(q: Array, *, eps: float = 1e-12) -> Array:
    return q / jnp.clip(jnp.linalg.norm(q, axis=-1, keepdims=True), min=eps)


def quaternion_conj(q: Array) -> Array:
    return jnp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quaternion_mul(q: Array, r: Array) -> Array:
    q0, qv = q[..., :1], q[..., 1:]
    r0, rv = r[..., :1], r[..., 1:]
    scalar = q0 * r0 - jnp.sum(qv * rv, axis=-1, keepdims=True)
    vector = q0 * rv + r0 * qv - jnp.cross(qv, rv, axis=-1)
    return jnp.concatenate([scalar, vector], axis=-1)


def su2_exp(algebra: Array) -> Array:
    r_sq = jnp.sum(algebra**2, axis=-1, keepdims=True)
    small = r_sq < 1e-12
    r = jnp.sqrt(jnp.clip(r_sq, min=1e-12))
    scalar = jnp.where(small, 1 - 0.5 * r_sq + r_sq**2 / 24, jnp.cos(r))
    scale = jnp.where(small, 1 - r_sq / 6 + r_sq**2 / 120, jnp.sin(r) / r)
    return quaternion_normalize(jnp.concatenate([scalar, scale * algebra], axis=-1))


def u2_normalize(links: Array) -> Array:
    return jnp.concatenate([regularize_phase(links[..., :1]), quaternion_normalize(links[..., 1:])], axis=-1)


def u2_mul(left: Array, right: Array) -> Array:
    phase = regularize_phase(left[..., :1] + right[..., :1])
    quaternion = quaternion_mul(left[..., 1:], right[..., 1:])
    return u2_normalize(jnp.concatenate([phase, quaternion], axis=-1))


def u2_conj(links: Array) -> Array:
    return jnp.concatenate([regularize_phase(-links[..., :1]), quaternion_conj(links[..., 1:])], axis=-1)


def u2_exp(algebra: Array) -> Array:
    if algebra.shape[-1] != 4:
        raise ValueError("U(2) algebra tensors must have four coefficients in the last dimension")
    return jnp.concatenate([regularize_phase(algebra[..., :1]), su2_exp(algebra[..., 1:])], axis=-1)


def u2_log(links: Array) -> Array:
    links = u2_normalize(links)
    phase = links[..., :1]
    quaternion = links[..., 1:]
    q0 = quaternion[..., :1]
    qv = quaternion[..., 1:]
    qv_norm = jnp.linalg.norm(qv, axis=-1, keepdims=True)
    angle = jnp.atan2(qv_norm, q0)
    scale = jnp.where(qv_norm > 1e-12, angle / jnp.clip(qv_norm, min=1e-12), jnp.ones_like(qv_norm))
    return jnp.concatenate([phase, scale * qv], axis=-1)


def identity_field(lattice_size: int, *, dtype: Any = jnp.float32, **_: Any) -> Array:
    field = jnp.zeros((2, lattice_size, lattice_size, 5), dtype=dtype)
    return field.at[..., 1].set(1.0)


def identity_like(links: Array) -> Array:
    result = jnp.zeros_like(links)
    return result.at[..., 1].set(1.0)


def plaquette_from_field(links: Array) -> Array:
    links = u2_normalize(links)
    link0, link1 = links[0], links[1]
    return u2_mul(u2_mul(u2_mul(link0, jnp.roll(link1, -1, 0)), u2_conj(jnp.roll(link0, -1, 1))), u2_conj(link1))


def plaquette_from_field_batch(links: Array) -> Array:
    links = u2_normalize(links)
    link0, link1 = links[:, 0], links[:, 1]
    return u2_mul(u2_mul(u2_mul(link0, jnp.roll(link1, -1, 1)), u2_conj(jnp.roll(link0, -1, 2))), u2_conj(link1))


def rectangle_from_field_batch(links: Array) -> Array:
    links = u2_normalize(links)
    link0, link1 = links[:, 0], links[:, 1]
    rect0 = u2_mul(
        u2_mul(
            u2_mul(u2_mul(u2_mul(link0, jnp.roll(link0, -1, 1)), jnp.roll(link1, -2, 1)), u2_conj(jnp.roll(link0, (-1, -1), (1, 2)))),
            u2_conj(jnp.roll(link0, -1, 2)),
        ),
        u2_conj(link1),
    )
    rect1 = u2_mul(
        u2_mul(
            u2_mul(u2_mul(u2_mul(link0, jnp.roll(link1, -1, 1)), jnp.roll(link1, (-1, -1), (1, 2))), u2_conj(jnp.roll(link0, -2, 2))),
            u2_conj(jnp.roll(link1, -1, 2)),
        ),
        u2_conj(link1),
    )
    return jnp.stack([rect0, rect1], axis=1)


def loop_sin_cos_features(loops: Array) -> Array:
    loops = u2_normalize(loops)
    phase = loops[..., :1]
    q0 = loops[..., 1:2]
    qv = loops[..., 2:]
    sin_like = jnp.concatenate([q0 * jnp.sin(phase), qv * jnp.cos(phase)], axis=-1)
    cos_like = jnp.concatenate([q0 * jnp.cos(phase), -qv * jnp.sin(phase)], axis=-1)
    return jnp.concatenate([sin_like, cos_like], axis=-1)


def plaquette_mean_from_field(links: Array) -> Array:
    plaquettes = plaquette_from_field(links)
    return jnp.mean(jnp.cos(plaquettes[..., 0]) * plaquettes[..., 1])


def plaquette_mean_from_field_batch(links: Array) -> Array:
    plaquettes = plaquette_from_field_batch(links)
    return jnp.mean(jnp.cos(plaquettes[..., 0]) * plaquettes[..., 1], axis=(1, 2))


def action_from_field_batch(links: Array, beta: float) -> Array:
    volume = links.shape[2] * links.shape[3]
    return beta * volume * (1 - plaquette_mean_from_field_batch(links))


def action_from_field(links: Array, beta: float) -> Array:
    return action_from_field_batch(links[jnp.newaxis, ...], beta)[0]


def force_from_field(links: Array, beta: float) -> Array:
    def varied_action(algebra: Array) -> Array:
        varied = u2_mul(u2_exp(algebra), links)
        return action_from_field(varied, beta)

    algebra = jnp.zeros((*links.shape[:-1], 4), dtype=links.dtype)
    return jax.grad(varied_action)(algebra)


def topology_from_field(links: Array) -> Array:
    plaquettes = plaquette_from_field(links)
    determinant_phase = regularize_phase(2 * plaquettes[..., 0])
    return jnp.floor(0.1 + jnp.sum(determinant_phase) / (2 * math.pi))


def plaq_mean_theory(beta: float) -> float:
    x = beta / 2
    partition = i0(x) ** 2 - i1(x) ** 2
    return float(0.5 * i1(x) * (i0(x) - iv(2, x)) / partition)


def u2_to_matrix(links: Array) -> Array:
    links = u2_normalize(links)
    phase = links[..., 0]
    q = links[..., 1:]
    a0, a1, a2, a3 = jnp.moveaxis(q, -1, 0)
    phase_factor = jnp.exp(1j * phase)
    matrix = jnp.stack(
        [
            jnp.stack([a0 + 1j * a3, a2 + 1j * a1], axis=-1),
            jnp.stack([-a2 + 1j * a1, a0 - 1j * a3], axis=-1),
        ],
        axis=-2,
    )
    return phase_factor[..., None, None] * matrix


def matrix_to_u2(matrix: Array) -> Array:
    matrix = jnp.asarray(matrix)
    determinant = jnp.linalg.det(matrix)
    phase = 0.5 * jnp.angle(determinant)
    phase_factor = jnp.exp(1j * phase)
    su2_matrix = matrix / phase_factor[..., None, None]
    m00 = su2_matrix[..., 0, 0]
    m01 = su2_matrix[..., 0, 1]
    m10 = su2_matrix[..., 1, 0]
    m11 = su2_matrix[..., 1, 1]
    q0 = 0.5 * (m00.real + m11.real)
    q1 = 0.5 * (m01.imag + m10.imag)
    q2 = 0.5 * (m01.real - m10.real)
    q3 = 0.5 * (m00.imag - m11.imag)
    quaternion = quaternion_normalize(jnp.stack([q0, q1, q2, q3], axis=-1))
    return u2_normalize(jnp.concatenate([phase[..., None].astype(quaternion.dtype), quaternion], axis=-1))


def get_link_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, 2, lattice_size, lattice_size, 1), dtype=bool)
    direction = 0 if index < 4 else 1
    parity = index % 4
    row_slice = slice(0, None, 2) if parity < 2 else slice(1, None, 2)
    col_slice = slice(0, None, 2) if parity in (0, 2) else slice(1, None, 2)
    mask[:, direction, row_slice, col_slice, :] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, 2, lattice_size, lattice_size, 1))


def get_plaq_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, lattice_size, lattice_size, 1), dtype=bool)
    if index in (0, 1):
        mask[:, 1::2, :, :] = True
    elif index in (2, 3):
        mask[:, 0::2, :, :] = True
    elif index in (4, 6):
        mask[:, :, 1::2, :] = True
    elif index in (5, 7):
        mask[:, :, 0::2, :] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, lattice_size, lattice_size, 1))


def get_rect_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, 2, lattice_size, lattice_size, 1), dtype=bool)
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, 2, lattice_size, lattice_size, 1))


def real_dtype_from_links(links: Array) -> Any:
    return links.dtype


def autocorrelation(topo: np.ndarray, max_lag: int, beta: float, volume: int) -> np.ndarray:
    topo = np.round(topo).astype(int)
    topo = topo - np.mean(topo)
    autocorrelations = np.zeros(max_lag + 1)
    for delta in range(max_lag + 1):
        if delta == 0:
            autocorrelations[delta] = 1.0
        elif delta >= len(topo):
            autocorrelations[delta] = np.nan
        else:
            topo_diff_squared = np.mean((topo[:-delta] - topo[delta:]) ** 2)
            autocorrelations[delta] = 1 - topo_diff_squared / (2 * volume)
    return autocorrelations
