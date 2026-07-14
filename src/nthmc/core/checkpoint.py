"""Small npz checkpoint helpers for JAX pytrees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def load_checkpoint(path: str | Path, template_params: Any) -> tuple[Any, dict[str, Any]]:
    """Load params from an npz checkpoint using a template pytree structure."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        leaves, treedef = jax.tree_util.tree_flatten(template_params)
        loaded = [data[f"param_{index}"] for index in range(len(leaves))]
        for index, (value, template) in enumerate(zip(loaded, leaves)):
            if value.shape != np.asarray(template).shape:
                raise ValueError(
                    f"Checkpoint parameter {index} has shape {value.shape}, "
                    f"expected {np.asarray(template).shape}"
                )
    return jax.tree_util.tree_unflatten(treedef, loaded), metadata
