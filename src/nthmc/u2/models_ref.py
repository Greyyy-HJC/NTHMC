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


def _scale_split_coefficients(
    plaq_logits: torch.Tensor,
    rect_logits: torch.Tensor,
    *,
    plaq_cap: float | torch.Tensor = 1 / 5,
    rect_cap: float | torch.Tensor = 1 / 40,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squash split plaquette/rectangle logits with explicit caps."""
    return torch.tanh(plaq_logits) * plaq_cap, torch.tanh(rect_logits) * rect_cap


class _LayerScale(nn.Module):
    """Per-channel ReZero gate applied to the pre-tanh logits.

    The learnable per-channel scale is initialized to zero, so every model starts as the
    exact identity transform (the output coefficients are zero) while the convolutions keep
    their healthy default initialization. Training ramps the gate up from zero. ``gain`` is a
    fixed (non-learnable) multiplier that sets how quickly a given variant's gate effectively
    grows; it defaults to 1. Because the gate scales the logits before the tanh heads, the
    strict tanh caps (and thus reversibility) are untouched.
    """

    def __init__(self, channels: int, gain: float = 1.0) -> None:
        super().__init__()
        self.gain = gain
        self.scale = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gain * self.scale * x


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


class WideLocalNet(LocalNet):
    """Base CNN with only the hidden layer widened."""

    def __init__(self) -> None:
        super().__init__(NetConfig(hidden_channels=32))
        self.out_scale = _LayerScale(self.config.output_channels, gain=0.9)


class CapLocalNet(LocalNet):
    """Base CNN with only the coefficient caps changed to 90% of a 50/50 split."""

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.out_scale(self.conv_output(x))
        plaq_logits = x[:, : self.config.plaq_output_channels]
        rect_logits = x[:, self.config.plaq_output_channels :]
        return _scale_split_coefficients(plaq_logits, rect_logits, plaq_cap=0.1125, rect_cap=0.05625)


class MultiScaleCapLocalNet(nn.Module):
    """Multi-scale wide split-head model with a fixed 90% 50/50 cap split."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig(hidden_channels=32)
        self.conv_input = nn.Conv2d(
            self.config.input_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.local_branch = nn.Conv2d(
            self.config.hidden_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.dilated_branch = nn.Conv2d(
            self.config.hidden_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
            dilation=2,
        )
        self.point_branch = nn.Conv2d(self.config.hidden_channels, self.config.hidden_channels, kernel_size=1)
        self.mix = nn.Conv2d(3 * self.config.hidden_channels, self.config.hidden_channels, kernel_size=1)
        self.activation = nn.GELU()
        self.plaq_head = nn.Conv2d(self.config.hidden_channels, self.config.plaq_output_channels, kernel_size=1)
        self.rect_head = nn.Conv2d(self.config.hidden_channels, self.config.rect_output_channels, kernel_size=1)
        self.plaq_scale = _LayerScale(self.config.plaq_output_channels, gain=0.8)
        self.rect_scale = _LayerScale(self.config.rect_output_channels, gain=0.8)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        local = self.activation(self.local_branch(x))
        dilated = self.activation(self.dilated_branch(x))
        point = self.activation(self.point_branch(x))
        x = self.activation(self.mix(torch.cat([local, dilated, point], dim=1)))
        plaq_logits = self.plaq_scale(self.plaq_head(x))
        rect_logits = self.rect_scale(self.rect_head(x))
        return _scale_split_coefficients(plaq_logits, rect_logits, plaq_cap=0.1125, rect_cap=0.05625)


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag in {"base"}:
        return LocalNet
    if model_tag in {"wide"}:
        return WideLocalNet
    if model_tag in {"cap"}:
        return CapLocalNet
    if model_tag in {"mscap"}:
        return MultiScaleCapLocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
