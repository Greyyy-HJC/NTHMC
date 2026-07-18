"""PyTorch neural-network models for U(2) field-transformation training."""

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
    def input_channels(self) -> int:
        return self.plaq_input_channels + self.rect_input_channels

    @property
    def plaq_output_channels(self) -> int:
        return self.plaq_loop_count * self.coeff_slots_per_loop

    @property
    def rect_output_channels(self) -> int:
        return self.rect_loop_count * self.coeff_slots_per_loop

    @property
    def output_channels(self) -> int:
        return self.plaq_output_channels + self.rect_output_channels


def _scale_coefficients(x: torch.Tensor, plaq_output_channels: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Split and squash raw conv outputs into bounded plaquette/rectangle coefficients."""
    plaq_coeffs = torch.tanh(x[:, :plaq_output_channels]) / 5
    rect_coeffs = torch.tanh(x[:, plaq_output_channels:]) / 40
    return plaq_coeffs, rect_coeffs


class _LayerScale(nn.Module):
    """Zero-initialized per-channel gate applied to the pre-tanh logits."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(2) field transformation.

    A zero-initialized per-channel output gate (``_LayerScale``) makes the model start as the
    identity transform without crippling the convolution weights, so the convolutions keep a
    healthy default initialization and the network ramps away from identity during training.
    """

    def __init__(self, config: NetConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetConfig()
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
        self.out_scale = _LayerScale(self.config.output_channels)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.out_scale(self.conv_output(x))
        return _scale_coefficients(x, self.config.plaq_output_channels)


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid U(2) model tag: {model_tag!r}")
