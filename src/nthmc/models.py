"""Neural-network models for field transformations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BaseNetConfig:
    plaq_input_channels: int = 2
    rect_input_channels: int = 4
    plaq_output_channels: int = 4
    rect_output_channels: int = 8
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)


class BaseLocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(1) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        config = BaseNetConfig()
        input_channels = config.plaq_input_channels + config.rect_input_channels
        output_channels = config.plaq_output_channels + config.rect_output_channels

        self.config = config
        self.conv_input = nn.Conv2d(
            input_channels,
            config.hidden_channels,
            config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            output_channels,
            config.kernel_size,
            padding="same",
            padding_mode="circular",
        )

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = torch.arctan(self.conv_output(x)) / math.pi / 3
        plaq_coeffs = x[:, : self.config.plaq_output_channels]
        rect_coeffs = x[:, self.config.plaq_output_channels :]
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a model class by tag. NTHMC currently implements only the base model."""
    if model_tag not in {"base", "simple"}:
        raise ValueError(f"Unsupported model_tag={model_tag!r}; only 'base' is implemented.")
    return BaseLocalNet

