"""Small npz checkpoint helpers for JAX pytrees."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _tree_to_named_arrays(prefix: str, tree: Any) -> tuple[dict[str, np.ndarray], list[Any]]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    arrays = {f"{prefix}_{index}": np.asarray(leaf) for index, leaf in enumerate(leaves)}
    specs = [{"name": f"{prefix}_{index}", "shape": list(np.asarray(leaf).shape)} for index, leaf in enumerate(leaves)]
    return arrays, [treedef, specs]


def save_checkpoint(
    path: str | Path,
    *,
    params: Any,
    opt_state: Any | None,
    metadata: dict[str, Any],
) -> None:
    """Save params, optimizer leaves, and metadata to an npz checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, param_info = _tree_to_named_arrays("param", params)
    opt_arrays: dict[str, np.ndarray] = {}
    opt_specs: list[dict[str, Any]] = []
    if opt_state is not None:
        opt_leaves, _ = jax.tree_util.tree_flatten(opt_state)
        opt_arrays = {f"opt_{index}": np.asarray(leaf) for index, leaf in enumerate(opt_leaves)}
        opt_specs = [{"name": f"opt_{index}", "shape": list(np.asarray(leaf).shape)} for index, leaf in enumerate(opt_leaves)]
    payload = {
        **arrays,
        **opt_arrays,
        "metadata_json": np.asarray(json.dumps({**metadata, "param_count": len(param_info[1]), "opt_count": len(opt_specs)})),
    }
    np.savez(path, **payload)


def load_checkpoint(path: str | Path, template_params: Any) -> tuple[Any, dict[str, Any]]:
    """Load params from an npz checkpoint using a template pytree structure."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"]))
        leaves, treedef = jax.tree_util.tree_flatten(template_params)
        loaded = [data[f"param_{index}"] for index in range(len(leaves))]
    return jax.tree_util.tree_unflatten(treedef, loaded), metadata
