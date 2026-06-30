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
    plaq_input_channels: int = 8
    rect_input_channels: int = 16
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


def _conv_init(key: Array, in_channels: int, out_channels: int, kernel_size: tuple[int, int], init_std: float) -> Params:
    key_w, key_b = jax.random.split(key)
    return {
        "weight": init_std * jax.random.normal(key_w, (out_channels, in_channels, *kernel_size), dtype=jnp.float32),
        "bias": init_std * jax.random.normal(key_b, (out_channels,), dtype=jnp.float32),
    }


def _config_for_tag(model_tag: str) -> NetConfig:
    if model_tag in {"wide", "mscap"}:
        return NetConfig(hidden_channels=32)
    if model_tag in {"base", "cap"}:
        return NetConfig()
    raise ValueError(f"Invalid U(2) model tag: {model_tag!r}")


def init_model_params(key: Array, model_tag: str, *, init_std: float = 0.001) -> Params:
    config = _config_for_tag(model_tag)
    key_input, key_output, key_scale = jax.random.split(key, 3)
    return {
        "conv_input": _conv_init(key_input, config.input_channels, config.hidden_channels, config.kernel_size, init_std),
        "conv_output": _conv_init(key_output, config.hidden_channels, config.output_channels, config.kernel_size, init_std),
        "out_scale": jnp.zeros((config.output_channels, 1, 1), dtype=jnp.float32),
    }


def init_transform_params(key: Array, model_tag: str, n_subsets: int, *, init_std: float = 0.001) -> Params:
    keys = jax.random.split(key, n_subsets)
    return {"subsets": [init_model_params(k, model_tag, init_std=init_std) for k in keys]}


def apply_model(model_params: Params, model_tag: str, plaq_features: Array, rect_features: Array) -> tuple[Array, Array]:
    config = _config_for_tag(model_tag)
    x = jnp.concatenate([plaq_features, rect_features], axis=1)
    x = gelu(circular_conv2d_nchw(x, model_params["conv_input"]))
    x = circular_conv2d_nchw(x, model_params["conv_output"]) * model_params["out_scale"][jnp.newaxis, ...]
    plaq_logits = x[:, : config.plaq_output_channels]
    rect_logits = x[:, config.plaq_output_channels :]
    if model_tag in {"cap", "mscap"}:
        return jnp.tanh(plaq_logits) * 0.1125, jnp.tanh(rect_logits) * 0.05625
    return jnp.tanh(plaq_logits) / 5, jnp.tanh(rect_logits) / 40


def choose_model(model_tag: str) -> str:
    _config_for_tag(model_tag)
    return model_tag
