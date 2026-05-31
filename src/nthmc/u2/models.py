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
    """Baseline structure (RF 5x5) with a wider hidden layer."""

    def __init__(self) -> None:
        super().__init__(NetConfig(hidden_channels=32))
        self.out_scale = _LayerScale(self.config.output_channels, gain=0.9)


class WideSplitLocalNet(nn.Module):
    """Wide shared trunk plus split plaquette/rectangle heads."""

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
        self.trunk = nn.Conv2d(
            self.config.hidden_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.plaq_head = nn.Conv2d(self.config.hidden_channels, self.config.plaq_output_channels, kernel_size=1)
        self.rect_head = nn.Conv2d(self.config.hidden_channels, self.config.rect_output_channels, kernel_size=1)
        self.plaq_scale = _LayerScale(self.config.plaq_output_channels, gain=0.9)
        self.rect_scale = _LayerScale(self.config.rect_output_channels, gain=0.9)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.activation(self.trunk(x))
        plaq_logits = self.plaq_scale(self.plaq_head(x))
        rect_logits = self.rect_scale(self.rect_head(x))
        return _scale_split_coefficients(plaq_logits, rect_logits)


class WideFlexCapLocalNet(WideSplitLocalNet):
    """Wide split-head model with a learnable reversibility-constrained cap split."""

    def __init__(self) -> None:
        super().__init__()
        self.cap_logit = nn.Parameter(torch.zeros(1))

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.activation(self.trunk(x))
        plaq_logits = self.plaq_scale(self.plaq_head(x))
        rect_logits = self.rect_scale(self.rect_head(x))
        plaq_budget = torch.sigmoid(self.cap_logit)
        c_plaq = plaq_budget / 4
        c_rect = (1 - plaq_budget) / 8
        return _scale_split_coefficients(plaq_logits, rect_logits, plaq_cap=c_plaq, rect_cap=c_rect)


class _ResidualBlock(nn.Module):
    """3x3 -> GELU -> 3x3 residual block with a LayerScale-gated branch.

    The gated branch starts near zero, so the block begins as an identity map and the
    network behaves like the shallow base path initially, adding depth gradually.
    """

    def __init__(self, channels: int, kernel_size: tuple[int, int], gain: float = 0.6) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding="same", padding_mode="circular")
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding="same", padding_mode="circular")
        self.layer_scale = _LayerScale(channels, gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.activation(self.conv1(x)))
        return x + self.layer_scale(residual)


class ResidualLocalNet(nn.Module):
    """Deeper CNN with 3x3 residual blocks reaching an intermediate receptive field (~13x13)."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig(hidden_channels=24)
        n_blocks = 1
        self.conv_input = nn.Conv2d(
            self.config.input_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.blocks = nn.ModuleList(
            _ResidualBlock(self.config.hidden_channels, self.config.kernel_size)
            for _ in range(n_blocks)
        )
        self.conv_output = nn.Conv2d(
            self.config.hidden_channels,
            self.config.output_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.out_scale = _LayerScale(self.config.output_channels, gain=0.8)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        for block in self.blocks:
            x = block(x)
        x = self.out_scale(self.conv_output(x))
        return _scale_coefficients(x, self.config.plaq_output_channels)


class DilatedLocalNet(nn.Module):
    """Dilated CNN reaching an intermediate receptive field (~15x15) with few parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig(hidden_channels=16)
        dilations = (1, 2, 1)
        self.activation = nn.GELU()
        convs = []
        in_channels = self.config.input_channels
        for dilation in dilations:
            convs.append(
                nn.Conv2d(
                    in_channels,
                    self.config.hidden_channels,
                    self.config.kernel_size,
                    padding="same",
                    padding_mode="circular",
                    dilation=dilation,
                )
            )
            in_channels = self.config.hidden_channels
        self.convs = nn.ModuleList(convs)
        self.conv_output = nn.Conv2d(self.config.hidden_channels, self.config.output_channels, kernel_size=1)
        self.out_scale = _LayerScale(self.config.output_channels, gain=0.5)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        for conv in self.convs:
            x = self.activation(conv(x))
        x = self.out_scale(self.conv_output(x))
        return _scale_coefficients(x, self.config.plaq_output_channels)


class MlpLocalNet(nn.Module):
    """UV-scale control: a 3x3 conv followed by per-site 1x1 layers (RF 3x3)."""

    def __init__(self) -> None:
        super().__init__()
        self.config = NetConfig(hidden_channels=16)
        self.conv_input = nn.Conv2d(
            self.config.input_channels,
            self.config.hidden_channels,
            self.config.kernel_size,
            padding="same",
            padding_mode="circular",
        )
        self.activation = nn.GELU()
        self.pointwise_hidden = nn.Conv2d(self.config.hidden_channels, 2 * self.config.hidden_channels, kernel_size=1)
        self.conv_output = nn.Conv2d(2 * self.config.hidden_channels, self.config.output_channels, kernel_size=1)
        self.out_scale = _LayerScale(self.config.output_channels, gain=0.3)

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.activation(self.pointwise_hidden(x))
        x = self.out_scale(self.conv_output(x))
        return _scale_coefficients(x, self.config.plaq_output_channels)


class FlexCapLocalNet(LocalNet):
    """Base backbone with a learnable, reversibility-constrained coefficient-cap split.

    Reversibility only requires 4*c_plaq + 8*c_rect <= 1 (presentation/Field_transform.md),
    not specifically c_plaq=1/5, c_rect=1/40 (which is the arbitrary 0.8/0.2 budget split).
    This variant keeps the exact base 2-layer backbone but lets training learn how to split
    the budget between the plaquette and rectangle caps, enforcing 4*c_plaq + 8*c_rect = 1
    exactly through a sigmoid so the active-link Jacobian stays non-singular for any value.
    """

    def __init__(self) -> None:
        super().__init__()
        # sigmoid(cap_logit) is the fraction of the unit budget assigned to the plaquette caps;
        # logit 0 starts at an even 0.5/0.5 split (c_plaq=1/8, c_rect=1/16).
        self.cap_logit = nn.Parameter(torch.zeros(1))

    def forward(self, plaq_features: torch.Tensor, rect_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([plaq_features, rect_features], dim=1)
        x = self.activation(self.conv_input(x))
        x = self.out_scale(self.conv_output(x))
        plaq_budget = torch.sigmoid(self.cap_logit)
        c_plaq = plaq_budget / 4
        c_rect = (1 - plaq_budget) / 8
        plaq_coeffs = torch.tanh(x[:, : self.config.plaq_output_channels]) * c_plaq
        rect_coeffs = torch.tanh(x[:, self.config.plaq_output_channels :]) * c_rect
        return plaq_coeffs, rect_coeffs


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a U(2) model class by tag."""
    if model_tag in {"base"}:
        return LocalNet
    if model_tag in {"wide"}:
        return WideLocalNet
    if model_tag in {"widesplit"}:
        return WideSplitLocalNet
    if model_tag in {"wideflex"}:
        return WideFlexCapLocalNet
    if model_tag in {"residual"}:
        return ResidualLocalNet
    if model_tag in {"dilated"}:
        return DilatedLocalNet
    if model_tag in {"mlp"}:
        return MlpLocalNet
    if model_tag in {"flex"}:
        return FlexCapLocalNet
    raise ValueError(f"Invalid model tag: {model_tag}")
