"""Neural-network models for U(2) field transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NetConfig:
    plaq_input_channels: int = 2
    rect_input_channels: int = 4
    plaq_output_channels: int = 8
    rect_output_channels: int = 16
    full_plaq_output_channels: int = 16
    full_rect_output_channels: int = 32
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(2) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig()
        input_channels = self.config.plaq_input_channels + self.config.rect_input_channels
        output_channels = self.config.plaq_output_channels + self.config.rect_output_channels
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

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.conv_output(x)
        plaq_phase_coeffs = torch.tanh(x[:, : self.config.plaq_output_channels]) / 5
        rect_phase_coeffs = torch.tanh(x[:, self.config.plaq_output_channels :]) / 40

        plaq_coeffs = torch.zeros(
            x.shape[0],
            self.config.full_plaq_output_channels,
            *x.shape[2:],
            device=x.device,
            dtype=x.dtype,
        )
        rect_coeffs = torch.zeros(
            x.shape[0],
            self.config.full_rect_output_channels,
            *x.shape[2:],
            device=x.device,
            dtype=x.dtype,
        )
        plaq_coeffs.reshape(x.shape[0], 4, 4, *x.shape[2:])[:, :, [0, 2]] = plaq_phase_coeffs.reshape(
            x.shape[0],
            4,
            2,
            *x.shape[2:],
        )
        rect_coeffs.reshape(x.shape[0], 8, 4, *x.shape[2:])[:, :, [0, 2]] = rect_phase_coeffs.reshape(
            x.shape[0],
            8,
            2,
            *x.shape[2:],
        )
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
