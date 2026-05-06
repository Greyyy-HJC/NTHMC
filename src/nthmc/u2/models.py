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
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)

    @property
    def plaq_output_channels(self) -> int:
        return self.plaq_loop_count * self.coeff_slots_per_loop

    @property
    def rect_output_channels(self) -> int:
        return self.rect_loop_count * self.coeff_slots_per_loop


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
        plaq_coeffs = torch.tanh(x[:, : self.config.plaq_output_channels]) / 5
        rect_coeffs = torch.tanh(x[:, self.config.plaq_output_channels :]) / 40
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
