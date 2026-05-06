"""U(2) lattice observables with split U(1) phase and SU(2) links."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from scipy.special import i0, i1, iv


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


def regularize_phase(theta: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    theta_wrapped = (theta - math.pi) / (2 * math.pi)
    return 2 * math.pi * (theta_wrapped - torch.floor(theta_wrapped) - 0.5)


def quaternion_normalize(q: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Project quaternions to unit norm along the last dimension."""
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)


def quaternion_conj(q: torch.Tensor) -> torch.Tensor:
    """Return the SU(2) inverse for unit quaternions."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Multiply SU(2) quaternions using the i sigma convention."""
    q0, qv = q[..., :1], q[..., 1:]
    r0, rv = r[..., :1], r[..., 1:]
    scalar = q0 * r0 - torch.sum(qv * rv, dim=-1, keepdim=True)
    vector = q0 * rv + r0 * qv - torch.linalg.cross(qv, rv, dim=-1)
    return torch.cat([scalar, vector], dim=-1)


def su2_exp(algebra: torch.Tensor) -> torch.Tensor:
    """Map three real algebra coefficients to SU(2) unit quaternions."""
    r_sq = torch.sum(algebra**2, dim=-1, keepdim=True)
    small = r_sq < 1e-12
    r = torch.sqrt(torch.clamp(r_sq, min=1e-12))
    scalar = torch.where(small, 1 - 0.5 * r_sq + r_sq**2 / 24, torch.cos(r))
    scale = torch.where(small, 1 - r_sq / 6 + r_sq**2 / 120, torch.sin(r) / r)
    return quaternion_normalize(torch.cat([scalar, scale * algebra], dim=-1))


def u2_normalize(links: torch.Tensor) -> torch.Tensor:
    """Normalize split U(2) links with shape [..., 5]."""
    phase = regularize_phase(links[..., :1])
    quaternion = quaternion_normalize(links[..., 1:])
    return torch.cat([phase, quaternion], dim=-1)


def u2_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply split U(2) links represented as phase plus SU(2) quaternion."""
    phase = regularize_phase(left[..., :1] + right[..., :1])
    quaternion = quaternion_mul(left[..., 1:], right[..., 1:])
    return u2_normalize(torch.cat([phase, quaternion], dim=-1))


def u2_conj(links: torch.Tensor) -> torch.Tensor:
    """Return the group inverse of split U(2) links."""
    return torch.cat([regularize_phase(-links[..., :1]), quaternion_conj(links[..., 1:])], dim=-1)


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if real_dtype == torch.float64 else torch.complex64


def u2_exp(algebra: torch.Tensor) -> torch.Tensor:
    """Map real U(2) algebra coefficients to split phase and SU(2) links."""
    if algebra.shape[-1] != 4:
        raise ValueError("U(2) algebra tensors must have four coefficients in the last dimension")

    phase = regularize_phase(algebra[..., :1])
    quaternion = su2_exp(algebra[..., 1:]) # real numbers (q0, q1, q2, q3)
    return torch.cat([phase, quaternion], dim=-1)


def u2_log(links: torch.Tensor) -> torch.Tensor:
    """Map split U(2) links to real U(2) algebra coefficients."""
    links = u2_normalize(links)
    phase = links[..., :1]
    quaternion = links[..., 1:]
    q0 = quaternion[..., :1]
    qv = quaternion[..., 1:]
    qv_norm = torch.linalg.norm(qv, dim=-1, keepdim=True)
    angle = torch.atan2(qv_norm, q0)
    scale = torch.where(qv_norm > 1e-12, angle / qv_norm.clamp_min(1e-12), torch.ones_like(qv_norm))
    return torch.cat([phase, scale * qv], dim=-1)


def identity_field(
    lattice_size: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return an identity U(2) gauge field with shape [2, L, L, 5]."""
    real_dtype = torch.get_default_dtype() if dtype is None else dtype
    field = torch.zeros((2, lattice_size, lattice_size, 5), device=device, dtype=real_dtype)
    field[..., 1] = 1.0
    return field


def plaquette_from_field(links: torch.Tensor) -> torch.Tensor:
    """Calculate plaquettes for a single split U(2) field with shape [2, L, L, 5]."""
    links = u2_normalize(links)
    link0, link1 = links[0], links[1]
    return u2_mul(
        u2_mul(
            u2_mul(link0, u2_conj(link1)),
            u2_conj(torch.roll(link0, shifts=-1, dims=1)),
        ),
        torch.roll(link1, shifts=-1, dims=0),
    )


def plaquette_from_field_batch(links: torch.Tensor) -> torch.Tensor:
    """Calculate plaquettes for split U(2) fields with shape [batch, 2, L, L, 5]."""
    links = u2_normalize(links)
    link0, link1 = links[:, 0], links[:, 1]
    return u2_mul(
        u2_mul(
            u2_mul(link0, u2_conj(link1)),
            u2_conj(torch.roll(link0, shifts=-1, dims=2)),
        ),
        torch.roll(link1, shifts=-1, dims=1),
    )


def rectangle_from_field_batch(links: torch.Tensor) -> torch.Tensor:
    """Calculate 1x2 and 2x1 rectangle loops for split U(2) fields."""
    links = u2_normalize(links)
    link0, link1 = links[:, 0], links[:, 1]

    rect0 = u2_mul(
        u2_mul(
            u2_mul(
                u2_mul(
                    u2_mul(link0, u2_conj(link1)),
                    u2_conj(torch.roll(link0, shifts=-1, dims=2)),
                ),
                u2_conj(torch.roll(link0, shifts=(-1, -1), dims=(1, 2))),
            ),
            torch.roll(link1, shifts=-2, dims=1),
        ),
        torch.roll(link0, shifts=-1, dims=1),
    )
    rect1 = u2_mul(
        u2_mul(
            u2_mul(
                u2_mul(
                    u2_mul(link0, u2_conj(link1)),
                    u2_conj(torch.roll(link1, shifts=-1, dims=2)),
                ),
                u2_conj(torch.roll(link0, shifts=-2, dims=2)),
            ),
            torch.roll(link1, shifts=(-1, -1), dims=(1, 2)),
        ),
        torch.roll(link1, shifts=-1, dims=1),
    )
    return torch.stack([rect0, rect1], dim=1)


def loop_sin_cos_features(loops: torch.Tensor) -> torch.Tensor:
    """Return sin-like and cos-like trace/traceless algebra coefficients for U(2) loops."""
    loops = u2_normalize(loops)
    phase = loops[..., :1]
    q0 = loops[..., 1:2] # real number
    qv = loops[..., 2:] # real numbers
    sin_like = torch.cat([q0 * torch.sin(phase), qv * torch.cos(phase)], dim=-1)
    cos_like = torch.cat([q0 * torch.cos(phase), -qv * torch.sin(phase)], dim=-1)
    return torch.cat([sin_like, cos_like], dim=-1)


def plaquette_mean_from_field(links: torch.Tensor) -> torch.Tensor:
    """Calculate the mean normalized plaquette, 0.5 * ReTr(U_p)."""
    plaquettes = plaquette_from_field(links)
    return torch.mean(torch.cos(plaquettes[..., 0]) * plaquettes[..., 1])


def plaquette_mean_from_field_batch(links: torch.Tensor) -> torch.Tensor:
    """Calculate per-configuration normalized plaquette means for a batch."""
    plaquettes = plaquette_from_field_batch(links)
    return torch.mean(torch.cos(plaquettes[..., 0]) * plaquettes[..., 1], dim=(1, 2))


def action_from_field_batch(links: torch.Tensor, beta: float) -> torch.Tensor:
    """Calculate the Wilson action for each split U(2) field in a batch."""
    volume = links.shape[2] * links.shape[3]
    return beta * volume * (1 - plaquette_mean_from_field_batch(links))


def topology_from_field(links: torch.Tensor) -> torch.Tensor:
    """Calculate integer-valued Q = 1/(2*pi) sum_x arg(det P_x01)."""
    plaquettes = plaquette_from_field(links)
    determinant_phase = regularize_phase(2 * plaquettes[..., 0])
    topo = torch.sum(determinant_phase) / (2 * math.pi)
    return torch.floor(0.1 + topo)


def plaq_mean_theory(beta: float) -> float:
    """Infinite-volume theoretical plaquette for 2D U(2) Wilson gauge theory."""
    x = beta / 2
    partition = i0(x) ** 2 - i1(x) ** 2
    return float(0.5 * i1(x) * (i0(x) - iv(2, x)) / partition)


def real_dtype_from_links(links: torch.Tensor) -> torch.dtype:
    """Return the real dtype associated with a split link field."""
    return links.dtype


def u2_to_matrix(links: torch.Tensor) -> torch.Tensor:
    """Convert split U(2) links to complex 2x2 matrices."""
    links = u2_normalize(links)
    phase = links[..., 0]
    q = links[..., 1:]
    a0, a1, a2, a3 = torch.unbind(q, dim=-1)
    complex_dtype = _complex_dtype(links.dtype)
    phase_factor = torch.exp(1j * phase.to(complex_dtype))
    m00 = a0.to(complex_dtype) + 1j * a3.to(complex_dtype)
    m01 = a2.to(complex_dtype) + 1j * a1.to(complex_dtype)
    m10 = -a2.to(complex_dtype) + 1j * a1.to(complex_dtype)
    m11 = a0.to(complex_dtype) - 1j * a3.to(complex_dtype)
    matrix = torch.stack(
        [
            torch.stack([m00, m01], dim=-1),
            torch.stack([m10, m11], dim=-1),
        ],
        dim=-2,
    )
    return phase_factor[..., None, None] * matrix


def matrix_to_u2(matrix: torch.Tensor) -> torch.Tensor:
    """Convert complex U(2) matrices to split phase plus SU(2) quaternions."""
    if matrix.shape[-2:] != (2, 2):
        raise ValueError("U(2) matrix tensors must end with shape [2, 2]")

    determinant = torch.linalg.det(matrix)
    phase = 0.5 * torch.angle(determinant)
    complex_dtype = matrix.dtype
    phase_factor = torch.exp(1j * phase.to(complex_dtype))
    su2_matrix = matrix / phase_factor[..., None, None]

    m00 = su2_matrix[..., 0, 0]
    m01 = su2_matrix[..., 0, 1]
    m10 = su2_matrix[..., 1, 0]
    m11 = su2_matrix[..., 1, 1]
    q0 = 0.5 * (m00.real + m11.real)
    q1 = 0.5 * (m01.imag + m10.imag)
    q2 = 0.5 * (m01.real - m10.real)
    q3 = 0.5 * (m00.imag - m11.imag)
    quaternion = quaternion_normalize(torch.stack([q0, q1, q2, q3], dim=-1))
    return u2_normalize(torch.cat([phase[..., None].to(quaternion.dtype), quaternion], dim=-1))


def get_link_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask the link subset updated by one U(2) coupling layer."""
    mask = torch.zeros((batch_size, 2, lattice_size, lattice_size, 1), dtype=torch.bool, device=device)
    direction = 0 if index < 4 else 1
    parity = index % 4
    row_slice = slice(0, None, 2) if parity < 2 else slice(1, None, 2)
    col_slice = slice(0, None, 2) if parity in (0, 2) else slice(1, None, 2)
    mask[:, direction, row_slice, col_slice, :] = True
    return mask


def get_plaq_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask plaquettes that depend on the active link subset."""
    mask = torch.zeros((batch_size, lattice_size, lattice_size, 1), dtype=torch.bool, device=device)
    if index in (0, 1):
        mask[:, 1::2, :, :] = True
    elif index in (2, 3):
        mask[:, 0::2, :, :] = True
    elif index in (4, 6):
        mask[:, :, 1::2, :] = True
    elif index in (5, 7):
        mask[:, :, 0::2, :] = True
    return mask


def get_rect_mask(index: int, batch_size: int, lattice_size: int, device: torch.device | str) -> torch.Tensor:
    """Mask rectangles that depend on the active link subset."""
    mask = torch.zeros((batch_size, 2, lattice_size, lattice_size, 1), dtype=torch.bool, device=device)

    if index == 0:
        mask[:, 1, 1::2, :, :] = True
        mask[:, 1, 0::2, 1::2, :] = True
    elif index == 1:
        mask[:, 1, 1::2, :, :] = True
        mask[:, 1, 0::2, 0::2, :] = True
    elif index == 2:
        mask[:, 1, 0::2, :, :] = True
        mask[:, 1, 1::2, 1::2, :] = True
    elif index == 3:
        mask[:, 1, 0::2, :, :] = True
        mask[:, 1, 1::2, 0::2, :] = True
    elif index == 4:
        mask[:, 0, :, 1::2, :] = True
        mask[:, 0, 1::2, 0::2, :] = True
    elif index == 5:
        mask[:, 0, :, 0::2, :] = True
        mask[:, 0, 1::2, 1::2, :] = True
    elif index == 6:
        mask[:, 0, :, 1::2, :] = True
        mask[:, 0, 0::2, 0::2, :] = True
    elif index == 7:
        mask[:, 0, :, 0::2, :] = True
        mask[:, 0, 0::2, 1::2, :] = True

    return mask


def identity_like(links: torch.Tensor) -> torch.Tensor:
    """Return identity split U(2) links with the same leading link shape."""
    result = torch.zeros_like(links)
    result[..., 1] = 1.0
    return result


def autocorrelation(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute a normalized autocorrelation estimate for a one-dimensional series."""
    values = np.asarray(values, dtype=float)
    centered = values - np.mean(values)
    variance = np.mean(centered**2)
    result = np.full(max_lag + 1, np.nan)
    result[0] = 1.0
    if variance == 0:
        return result
    for lag in range(1, max_lag + 1):
        if lag < len(centered):
            result[lag] = np.mean(centered[:-lag] * centered[lag:]) / variance
    return result
