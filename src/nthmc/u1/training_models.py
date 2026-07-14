"""PyTorch neural-network models for U(1) field-transformation training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class NetConfig:
    plaq_input_channels: int = 2
    rect_input_channels: int = 4
    plaq_output_channels: int = 4
    rect_output_channels: int = 8
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)

    @property
    def input_channels(self) -> int:
        return self.plaq_input_channels + self.rect_input_channels

    @property
    def output_channels(self) -> int:
        return self.plaq_output_channels + self.rect_output_channels


class _LayerScale(nn.Module):
    """Zero-initialized per-channel gate matching the JAX runtime model."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(1) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        config = NetConfig()
        self.config = config
        
        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            config.input_channels,
            config.hidden_channels,
            config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.output_channels,
            config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.out_scale = _LayerScale(config.output_channels)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = torch.arctan(self.out_scale(self.conv_output(x))) / math.pi / 3
        plaq_sin_coeffs = x[:, : self.config.plaq_output_channels]
        rect_sin_coeffs = x[:, self.config.plaq_output_channels :]
        plaq_coeffs = torch.cat([plaq_sin_coeffs, torch.zeros_like(plaq_sin_coeffs)], dim=1)
        rect_coeffs = torch.cat([rect_sin_coeffs, torch.zeros_like(rect_sin_coeffs)], dim=1)
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a model class by tag. NTHMC currently implements only the base model."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid U(1) model tag: {model_tag!r}")
