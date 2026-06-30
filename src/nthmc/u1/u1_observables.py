"""JAX U(1) lattice observables and utility functions."""

from __future__ import annotations

import math
import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad
from scipy.special import i0, i1

Array = Any


def set_seed(seed: int) -> None:
    """Set host-side seeds; JAX device randomness is handled by explicit keys."""
    random.seed(seed)
    np.random.seed(seed)


def format_beta(beta: float) -> str:
    value = float(beta)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def regularize(theta: Array) -> Array:
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - jnp.floor(theta_wrapped) - 0.5)


def plaq_from_field(theta: Array) -> Array:
    theta0, theta1 = theta[0], theta[1]
    return theta0 - theta1 - jnp.roll(theta0, shift=-1, axis=1) + jnp.roll(theta1, shift=-1, axis=0)


def plaq_from_field_batch(theta: Array) -> Array:
    theta0, theta1 = theta[:, 0], theta[:, 1]
    return theta0 - theta1 - jnp.roll(theta0, shift=-1, axis=2) + jnp.roll(theta1, shift=-1, axis=1)


def rect_from_field_batch(theta: Array) -> Array:
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
    return jnp.mean(jnp.cos(regularize(plaq_from_field(theta))))


def topo_from_field(theta: Array) -> Array:
    theta_p = regularize(plaq_from_field(theta))
    return jnp.floor(0.1 + jnp.sum(theta_p) / (2 * math.pi))


def action(theta: Array, beta: float) -> Array:
    return -beta * jnp.sum(jnp.cos(regularize(plaq_from_field(theta))))


def force(theta: Array, beta: float) -> Array:
    return jax.grad(lambda field: action(field, beta))(theta)


def plaq_mean_theory(beta: float) -> float:
    return float(i1(beta) / i0(beta))


def chi_infinity(beta: float) -> float:
    def numerator_integrand(phi: float) -> float:
        return (phi / (2 * math.pi)) ** 2 * math.exp(beta * math.cos(phi))

    def denominator_integrand(phi: float) -> float:
        return math.exp(beta * math.cos(phi))

    numerator, _ = quad(numerator_integrand, -math.pi, math.pi)
    denominator, _ = quad(denominator_integrand, -math.pi, math.pi)
    return numerator / denominator


def autocorrelation_from_chi(topo: np.ndarray, max_lag: int, beta: float, volume: int) -> np.ndarray:
    topo = np.round(topo).astype(int)
    topo = topo - np.mean(topo)
    chi_t_inf = chi_infinity(beta)
    autocorrelations = np.zeros(max_lag + 1)
    for delta in range(max_lag + 1):
        if delta == 0:
            autocorrelations[delta] = 1.0
        elif delta >= len(topo):
            autocorrelations[delta] = np.nan
        else:
            topo_diff_squared = np.mean((topo[:-delta] - topo[delta:]) ** 2)
            autocorrelations[delta] = 1 - topo_diff_squared / (2 * volume * chi_t_inf)
    return autocorrelations


def get_field_mask(index: int, batch_size: int, lattice_size: int) -> Array:
    mask = np.zeros((1, 2, lattice_size, lattice_size), dtype=bool)
    direction = 0 if index < 4 else 1
    parity = index % 4
    row_slice = slice(0, None, 2) if parity < 2 else slice(1, None, 2)
    col_slice = slice(0, None, 2) if parity in (0, 2) else slice(1, None, 2)
    mask[:, direction, row_slice, col_slice] = True
    return jnp.broadcast_to(jnp.asarray(mask), (batch_size, 2, lattice_size, lattice_size))


def get_plaq_mask(index: int, batch_size: int, lattice_size: int) -> Array:
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


def get_rect_mask(index: int, batch_size: int, lattice_size: int) -> Array:
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
