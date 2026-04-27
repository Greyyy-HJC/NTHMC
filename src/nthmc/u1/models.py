"""Neural-network models for field transformations."""

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


class LocalNet(nn.Module):
    """Small two-layer CNN used as the baseline U(1) field transformation."""

    def __init__(self) -> None:
        super().__init__()
        config = NetConfig()
        input_channels = config.plaq_input_channels + config.rect_input_channels
        output_channels = config.plaq_output_channels + config.rect_output_channels

        self.config = config
        
        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            input_channels,
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


class LocalNetAddCosNoRect(nn.Module):
    """
    Totally remove the rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * config.plaq_output_channels #! add cos terms

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            combined_input_channels,
            config.hidden_channels,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation = nn.GELU()  # 0 parameters
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            combined_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv_input(x)
        x = self.activation(x)  # 0 parameters
        
        # Second conv layer (1,308 parameters used)
        x = self.conv_output(x)
        
        # Output scaling
        plaq_coeffs = torch.tanh(x[:, :, :, :]) / 4  # [batch_size, 8, L, L]
        
        return plaq_coeffs 


def choose_model(model_tag: str) -> type[nn.Module]:
    """Return a model class by tag. NTHMC currently implements only the base model."""
    if model_tag == 'base':
        return LocalNet
    elif model_tag == 'addcos':
        return LocalNetAddCosNoRect
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")

