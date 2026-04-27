"""U(1) lattice observables and utility functions."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import i0, i1


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible single-process runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_beta(beta: float) -> str:
    """Return a compact beta tag that preserves the old integer-as-decimal style."""
    value = float(beta)
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def regularize(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - torch.floor(theta_wrapped) - 0.5)


def plaq_from_field(theta: torch.Tensor) -> torch.Tensor:
    """Calculate plaquettes for a single U(1) field with shape [2, L, L]."""
    theta0, theta1 = theta[0], theta[1]
    return theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=1) + torch.roll(theta1, shifts=-1, dims=0)


def plaq_from_field_batch(theta: torch.Tensor) -> torch.Tensor:
    """Calculate plaquettes for a batch with shape [batch, 2, L, L]."""
    theta0, theta1 = theta[:, 0], theta[:, 1]
    return theta0 - theta1 - torch.roll(theta0, shifts=-1, dims=2) + torch.roll(theta1, shifts=-1, dims=1)


def rect_from_field_batch(theta: torch.Tensor) -> torch.Tensor:
    """Calculate 1x2 and 2x1 rectangle loops for a batch of U(1) fields."""
    theta0, theta1 = theta[:, 0], theta[:, 1]
    rect0 = (
        theta0
        + torch.roll(theta0, shifts=-1, dims=1)
        + torch.roll(theta1, shifts=-2, dims=1)
        - torch.roll(theta0, shifts=(-1, -1), dims=(1, 2))
        - torch.roll(theta0, shifts=-1, dims=2)
        - theta1
    )
    rect1 = (
        theta0
        + torch.roll(theta1, shifts=-1, dims=1)
        + torch.roll(theta1, shifts=(-1, -1), dims=(1, 2))
        - torch.roll(theta0, shifts=-2, dims=2)
        - torch.roll(theta1, shifts=-1, dims=2)
        - theta1
    )
    return torch.stack([rect0, rect1], dim=1)


def plaq_mean_from_field(theta: torch.Tensor) -> torch.Tensor:
    """Calculate the mean plaquette for a single field."""
    return torch.mean(torch.cos(regularize(plaq_from_field(theta))))


def topo_from_field(theta: torch.Tensor) -> torch.Tensor:
    """Calculate the integer-valued U(1) topological charge."""
    theta_p = regularize(plaq_from_field(theta))
    return torch.floor(0.1 + torch.sum(theta_p) / (2 * math.pi))


def plaq_mean_theory(beta: float) -> float:
    """Infinite-volume theoretical plaquette for 2D U(1)."""
    return float(i1(beta) / i0(beta))


def chi_infinity(beta: float) -> float:
    """Infinite-volume topological susceptibility for 2D U(1)."""

    def numerator_integrand(phi: float) -> float:
        return (phi / (2 * math.pi)) ** 2 * math.exp(beta * math.cos(phi))

    def denominator_integrand(phi: float) -> float:
        return math.exp(beta * math.cos(phi))

    numerator, _ = quad(numerator_integrand, -math.pi, math.pi)
    denominator, _ = quad(denominator_integrand, -math.pi, math.pi)
    return numerator / denominator


def autocorrelation_from_chi(topo: np.ndarray, max_lag: int, beta: float, volume: int) -> np.ndarray:
    """Compute the normalized topological-charge autocorrelation estimate."""
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


def get_field_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask the link subset updated by one coupling layer."""
    mask = torch.zeros((batch_size, 2, lattice_size, lattice_size), dtype=torch.bool, device=device)
    direction = 0 if index < 4 else 1
    parity = index % 4
    row_slice = slice(0, None, 2) if parity < 2 else slice(1, None, 2)
    col_slice = slice(0, None, 2) if parity in (0, 2) else slice(1, None, 2)
    mask[:, direction, row_slice, col_slice] = True
    return mask


def get_plaq_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask plaquettes that depend on the active link subset."""
    mask = torch.zeros((batch_size, lattice_size, lattice_size), dtype=torch.bool, device=device)
    if index in (0, 1):
        mask[:, 1::2, :] = True
    elif index in (2, 3):
        mask[:, 0::2, :] = True
    elif index in (4, 6):
        mask[:, :, 1::2] = True
    elif index in (5, 7):
        mask[:, :, 0::2] = True
    return mask


def get_rect_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask rectangles that depend on the active link subset."""
    mask = torch.zeros((batch_size, 2, lattice_size, lattice_size), dtype=torch.bool, device=device)

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

    return mask
