"""Neural-network models for U(2) field transformations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NetConfig:
    input_channels: int = 6
    output_channels: int = 4
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(2) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig()
        self.conv_input = nn.Conv2d(
            self.config.input_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.conv_output = nn.Conv2d(
            self.config.hidden_channels,
            self.config.output_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.conv_input(features))
        x = torch.tanh(self.conv_output(x)) / 4
        return x.permute(0, 2, 3, 1).contiguous()


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
