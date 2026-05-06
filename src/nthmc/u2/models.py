"""Neural-network models for U(2) field transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NetConfig:
    plaq_input_channels: int = 6
    rect_input_channels: int = 12
    plaq_loop_count: int = 4
    rect_loop_count: int = 8
    coeff_slots_per_loop: int = 4
    phase_coeff_slots: tuple[int, int] = (0, 2)
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)

    @property
    def phase_coeff_slots_per_loop(self) -> int:
        return len(self.phase_coeff_slots)

    @property
    def plaq_phase_channels(self) -> int:
        return self.plaq_loop_count * self.phase_coeff_slots_per_loop

    @property
    def rect_phase_channels(self) -> int:
        return self.rect_loop_count * self.phase_coeff_slots_per_loop


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(2) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig()
        input_channels = self.config.plaq_input_channels + self.config.rect_input_channels
        output_channels = self.config.plaq_phase_channels + self.config.rect_phase_channels
        self.conv_input = nn.Conv2d(
            input_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.conv_output = nn.Conv2d(
            self.config.hidden_channels,
            output_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )

    def _expand_phase_coefficients(self, phase_coeffs: torch.Tensor, loop_count: int) -> torch.Tensor:
        """Place learned phase coefficients into the full per-loop (k0,k1,k2,k3) layout."""
        batch_size, _, height, width = phase_coeffs.shape
        coeffs_by_loop = torch.zeros(
            batch_size,
            loop_count,
            self.config.coeff_slots_per_loop,
            height,
            width,
            device=phase_coeffs.device,
            dtype=phase_coeffs.dtype,
        )
        phase_by_loop = phase_coeffs.reshape(
            batch_size,
            loop_count,
            self.config.phase_coeff_slots_per_loop,
            height,
            width,
        )
        for phase_index, coeff_slot in enumerate(self.config.phase_coeff_slots):
            coeffs_by_loop[:, :, coeff_slot] = phase_by_loop[:, :, phase_index]
        return coeffs_by_loop.reshape(
            batch_size,
            loop_count * self.config.coeff_slots_per_loop,
            height,
            width,
        )

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.conv_output(x)
        plaq_phase_coeffs = torch.tanh(x[:, : self.config.plaq_phase_channels]) / 5
        rect_phase_coeffs = torch.tanh(x[:, self.config.plaq_phase_channels :]) / 40

        plaq_coeffs = self._expand_phase_coefficients(
            plaq_phase_coeffs,
            self.config.plaq_loop_count,
        )
        rect_coeffs = self._expand_phase_coefficients(
            rect_phase_coeffs,
            self.config.rect_loop_count,
        )
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
