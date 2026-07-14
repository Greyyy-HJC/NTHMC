"""Pure JAX CNN models for U(1) field transformations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

Array = Any
Params = dict[str, Any]


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


def _scale_coefficients(x: Array, plaq_output_channels: int) -> tuple[Array, Array]:
    """Squash logits and zero-pad cos halves so field_transform gets sin/cos-shaped coeffs."""
    x = jnp.arctan(x) / math.pi / 3
    plaq_sin = x[:, :plaq_output_channels]
    rect_sin = x[:, plaq_output_channels:]
    return (
        jnp.concatenate([plaq_sin, jnp.zeros_like(plaq_sin)], axis=1),
        jnp.concatenate([rect_sin, jnp.zeros_like(rect_sin)], axis=1),
    )


def _conv_init(key: Array, in_channels: int, out_channels: int, kernel_size: tuple[int, int], init_std: float) -> Params:
    weight_key, bias_key = jax.random.split(key)
    weight = init_std * jax.random.normal(weight_key, (out_channels, in_channels, *kernel_size), dtype=jnp.float32)
    bias = init_std * jax.random.normal(bias_key, (out_channels,), dtype=jnp.float32)
    return {"weight": weight, "bias": bias}


def circular_conv2d_nchw(x: Array, layer: Params) -> Array:
    weight = layer["weight"]
    bias = layer["bias"]
    pad_h = weight.shape[2] // 2
    pad_w = weight.shape[3] // 2
    x = jnp.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="wrap")
    y = jax.lax.conv_general_dilated(
        x,
        weight,
        window_strides=(1, 1),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return y + bias.reshape(1, -1, 1, 1)


def gelu(x: Array) -> Array:
    return 0.5 * x * (1.0 + jax.lax.erf(x / math.sqrt(2.0)))


class LocalNet:
    """Small two-layer CNN used as the baseline U(1) field transformation.

    Functional JAX counterpart of the old Torch ``nn.Module``: ``init`` builds params,
    ``apply`` runs the forward pass. Exact identity comes from a zero ``out_scale`` gate.
    """

    @staticmethod
    def init(key: Array, *, init_std: float = 0.001, config: NetConfig | None = None) -> Params:
        config = config or NetConfig()
        key_input, key_output = jax.random.split(key)
        return {
            "conv_input": _conv_init(key_input, config.input_channels, config.hidden_channels, config.kernel_size, init_std),
            "conv_output": _conv_init(key_output, config.hidden_channels, config.output_channels, config.kernel_size, init_std),
            "out_scale": jnp.zeros((config.output_channels, 1, 1), dtype=jnp.float32),
        }

    @staticmethod
    def apply(params: Params, plaq_features: Array, rect_features: Array, *, config: NetConfig | None = None) -> tuple[Array, Array]:
        config = config or NetConfig()
        x = jnp.concatenate([plaq_features, rect_features], axis=1)
        x = gelu(circular_conv2d_nchw(x, params["conv_input"]))
        x = circular_conv2d_nchw(x, params["conv_output"]) * params["out_scale"][jnp.newaxis, ...]
        return _scale_coefficients(x, config.plaq_output_channels)


def init_transform_params(key: Array, model: type[LocalNet], n_subsets: int, *, init_std: float = 0.001) -> Params:
    keys = jax.random.split(key, n_subsets)
    return {"subsets": [model.init(k, init_std=init_std) for k in keys]}


def choose_model(model_tag: str) -> type[LocalNet]:
    """Return a U(1) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid U(1) model tag: {model_tag!r}")
