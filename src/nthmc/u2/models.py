"""Pure JAX CNN models for U(2) field transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from nthmc.u1.models import circular_conv2d_nchw, gelu

Array = Any
Params = dict[str, Any]


@dataclass(frozen=True)
class NetConfig:
    plaq_input_channels: int = 6
    rect_input_channels: int = 12
    plaq_output_channels: int = 16
    rect_output_channels: int = 32
    hidden_channels: int = 12
    kernel_size: tuple[int, int] = (3, 3)

    @property
    def input_channels(self) -> int:
        return self.plaq_input_channels + self.rect_input_channels

    @property
    def output_channels(self) -> int:
        return self.plaq_output_channels + self.rect_output_channels


def _scale_coefficients(x: Array, plaq_output_channels: int) -> tuple[Array, Array]:
    """Split and squash raw conv outputs into bounded plaquette/rectangle coefficients."""
    plaq_coeffs = jnp.tanh(x[:, :plaq_output_channels]) / 5
    rect_coeffs = jnp.tanh(x[:, plaq_output_channels:]) / 40
    return plaq_coeffs, rect_coeffs


def _conv_init(key: Array, in_channels: int, out_channels: int, kernel_size: tuple[int, int], init_std: float) -> Params:
    key_w, key_b = jax.random.split(key)
    return {
        "weight": init_std * jax.random.normal(key_w, (out_channels, in_channels, *kernel_size), dtype=jnp.float32),
        "bias": init_std * jax.random.normal(key_b, (out_channels,), dtype=jnp.float32),
    }


class LocalNet:
    """Small two-layer CNN used as the baseline U(2) field transformation.

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
    """Return a U(2) model class by tag."""
    if model_tag == "base":
        return LocalNet
    raise ValueError(f"Invalid U(2) model tag: {model_tag!r}")
