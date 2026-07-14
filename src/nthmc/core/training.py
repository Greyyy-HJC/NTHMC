"""Shared batching and checkpoint helpers for PyTorch model training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def fixed_batches(
    data: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return fixed-shape batches and a mask excluding copied tail samples."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(data) == 0:
        return []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(len(data), generator=generator) if shuffle else torch.arange(len(data))
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(indices), batch_size):
        batch = data[indices[start : start + batch_size]]
        valid_count = len(batch)
        if valid_count < batch_size:
            batch = torch.cat([batch, batch[-1:].expand(batch_size - valid_count, *batch.shape[1:])], dim=0)
        mask = torch.arange(batch_size) < valid_count
        batches.append((batch, mask))
    return batches


def local_batch(
    batch: torch.Tensor,
    mask: torch.Tensor,
    *,
    rank: int,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(batch) % world_size:
        raise ValueError(f"global batch size {len(batch)} must be divisible by world_size={world_size}")
    local_size = len(batch) // world_size
    start = rank * local_size
    return batch[start : start + local_size], mask[start : start + local_size]


def unwrap_model(model: torch.nn.Module, fabric: Any | None) -> torch.nn.Module:
    del fabric
    return model.module if hasattr(model, "module") else model


def models_to_jax_params(models: list[torch.nn.Module], fabric: Any | None = None) -> dict[str, Any]:
    """Build the existing JAX parameter pytree using OIHW Torch weights."""
    subsets = []
    for wrapped in models:
        model = unwrap_model(wrapped, fabric)
        subsets.append(
            {
                "conv_input": {
                    "weight": model.conv_input.weight.detach().cpu().numpy(),
                    "bias": model.conv_input.bias.detach().cpu().numpy(),
                },
                "conv_output": {
                    "weight": model.conv_output.weight.detach().cpu().numpy(),
                    "bias": model.conv_output.bias.detach().cpu().numpy(),
                },
                "out_scale": model.out_scale.scale.detach().cpu().numpy(),
            }
        )
    return {"subsets": subsets}


def save_jax_npz(path: str | Path, models: list[torch.nn.Module], metadata: dict[str, Any], fabric: Any | None = None) -> None:
    """Write the exact flattened NPZ layout consumed by core.checkpoint."""
    params = models_to_jax_params(models, fabric)
    arrays: dict[str, np.ndarray] = {}
    index = 0
    for subset in params["subsets"]:
        # JAX tree_flatten sorts dictionary keys recursively.
        for value in (
            subset["conv_input"]["bias"],
            subset["conv_input"]["weight"],
            subset["conv_output"]["bias"],
            subset["conv_output"]["weight"],
            subset["out_scale"],
        ):
            arrays[f"param_{index}"] = np.asarray(value)
            index += 1
    payload = {
        **arrays,
        "metadata_json": np.asarray(json.dumps({**metadata, "param_count": index, "opt_count": 0})),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_jax_npz(path: str | Path, models: list[torch.nn.Module], fabric: Any | None = None) -> dict[str, Any]:
    """Load model weights from the canonical JAX NPZ without initializing JAX."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        index = 0
        for wrapped in models:
            model = unwrap_model(wrapped, fabric)
            values = [data[f"param_{index + offset}"] for offset in range(5)]
            index += 5
            targets = (
                model.conv_input.bias,
                model.conv_input.weight,
                model.conv_output.bias,
                model.conv_output.weight,
                model.out_scale.scale,
            )
            for target, value in zip(targets, values):
                if tuple(target.shape) != tuple(value.shape):
                    raise ValueError(f"checkpoint shape {value.shape} does not match model shape {tuple(target.shape)}")
                target.data.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))
    return metadata
