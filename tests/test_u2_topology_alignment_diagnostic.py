#!/usr/bin/env python3
"""Diagnose whether the U(2) force aligns with the soft topology gradient."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_observables import format_beta, matrix_to_u2, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice_size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=2)
    parser.add_argument("--n_subsets", type=int, default=8)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=1029)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=REPO_ROOT / "2du2" / "artifacts" / "models")
    parser.add_argument("--alignment_eps", type=float, default=1e-6)
    parser.add_argument("--alignment_margin", type=float, default=0.1)
    parser.add_argument("--no_shuffle", action="store_true")
    return parser.parse_args()


def load_links(args: argparse.Namespace) -> torch.Tensor:
    beta_tag = format_beta(args.beta)
    data_path = args.data_path or REPO_ROOT / "2du2" / "configs" / f"links_L{args.lattice_size}_beta{beta_tag}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing U(2) config file: {data_path}")
    matrices = torch.from_numpy(np.load(data_path))
    links = matrix_to_u2(matrices).float()
    sample_count = min(len(links), args.batch_size * args.n_batches)
    if sample_count <= 0:
        raise ValueError(f"No configurations loaded from {data_path}")
    if args.no_shuffle:
        return links[:sample_count]
    generator = torch.Generator().manual_seed(args.rand_seed)
    return links[torch.randperm(len(links), generator=generator)[:sample_count]]


def cosine_by_config(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    numerator = torch.sum(left_flat * right_flat, dim=1)
    denominator = torch.linalg.vector_norm(left_flat, dim=1) * torch.linalg.vector_norm(right_flat, dim=1)
    return numerator / denominator.clamp_min(torch.finfo(left.dtype).eps)


def norm_by_config(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value.reshape(value.shape[0], -1), dim=1)


def summarize(name: str, values: list[torch.Tensor]) -> None:
    tensor = torch.cat([value.detach().cpu().reshape(-1) for value in values])
    summarize_tensor(name, tensor)


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: mean={tensor.mean().item():.6e} "
        f"std={tensor.std(unbiased=False).item():.6e} "
        f"min={tensor.min().item():.6e} max={tensor.max().item():.6e}"
    )


def summarize_alignment_objectives(
    prefix: str,
    values: list[torch.Tensor],
    *,
    eps: float,
    margin: float,
) -> None:
    cos = torch.cat([value.detach().cpu().reshape(-1) for value in values])
    abs_cos = torch.abs(cos)
    cos_sq = cos**2
    smooth_abs = torch.sqrt(cos_sq + eps)
    hinge = torch.clamp(margin - abs_cos, min=0.0)

    summarize_tensor(f"{prefix}_current_neg_cos_sq_loss", -cos_sq)
    summarize_tensor(f"{prefix}_smooth_neg_abs_cos_loss", -smooth_abs)
    summarize_tensor(f"{prefix}_hinge_orthogonal_penalty", hinge)
    summarize_tensor(f"{prefix}_current_neg_cos_sq_grad_scale", 2 * abs_cos)
    summarize_tensor(f"{prefix}_smooth_neg_abs_cos_grad_scale", abs_cos / smooth_abs)
    summarize_tensor(f"{prefix}_hinge_active_frac", (abs_cos < margin).to(cos.dtype))


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    set_seed(args.rand_seed)
    device = torch.device(args.device)

    links = load_links(args)
    field_transform = FieldTransformation(
        args.lattice_size,
        device=str(device),
        n_subsets=args.n_subsets,
        model_tag=args.model_tag,
        save_tag=args.save_tag,
        model_dir=args.model_dir,
    )
    field_transform.train_beta = args.beta
    if args.save_tag is not None:
        field_transform.load_best_model(args.beta)
    field_transform._set_models_mode(False)

    print(
        "# U(2) topology-alignment diagnostic "
        f"L={args.lattice_size} beta={args.beta} samples={len(links)} "
        f"batch_size={args.batch_size} n_batches={args.n_batches} "
        f"device={device} save_tag={args.save_tag}"
    )
    print(f"# model_dir={args.model_dir}")
    print(f"# alignment_eps={args.alignment_eps} alignment_margin={args.alignment_margin}")

    force_losses: list[torch.Tensor] = []
    full_cos: list[torch.Tensor] = []
    phase_cos: list[torch.Tensor] = []
    force_norms: list[torch.Tensor] = []
    topo_grad_norms: list[torch.Tensor] = []
    phase_force_norms: list[torch.Tensor] = []
    phase_topo_grad_norms: list[torch.Tensor] = []

    for start in range(0, len(links), args.batch_size):
        batch = links[start : start + args.batch_size].to(device)
        with torch.no_grad():
            transformed = field_transform.inverse(batch)
        with torch.enable_grad():
            force, _, _, _, topo_grad = field_transform.compute_transformed_force_terms(
                transformed.detach(),
                args.beta,
                create_graph=False,
                include_topo_grad=True,
            )
        force_components = field_transform._force_loss_components(force)
        force_losses.append(torch.tensor([field_transform._weighted_force_loss(force_components)]))
        full_cos.append(cosine_by_config(force, topo_grad))
        phase_cos.append(cosine_by_config(force[..., :1], topo_grad[..., :1]))
        force_norms.append(norm_by_config(force))
        topo_grad_norms.append(norm_by_config(topo_grad))
        phase_force_norms.append(norm_by_config(force[..., :1]))
        phase_topo_grad_norms.append(norm_by_config(topo_grad[..., :1]))

    summarize("force_weighted_loss", force_losses)
    summarize("force_topo_cos", full_cos)
    summarize("force_topo_abs_cos", [torch.abs(value) for value in full_cos])
    summarize("force_topo_cos_sq", [value**2 for value in full_cos])
    summarize_alignment_objectives(
        "force_topo",
        full_cos,
        eps=args.alignment_eps,
        margin=args.alignment_margin,
    )
    summarize("phase_force_topo_cos", phase_cos)
    summarize("phase_force_topo_abs_cos", [torch.abs(value) for value in phase_cos])
    summarize("phase_force_topo_cos_sq", [value**2 for value in phase_cos])
    summarize_alignment_objectives(
        "phase_force_topo",
        phase_cos,
        eps=args.alignment_eps,
        margin=args.alignment_margin,
    )
    summarize("force_norm", force_norms)
    summarize("topo_grad_norm", topo_grad_norms)
    summarize("phase_force_norm", phase_force_norms)
    summarize("phase_topo_grad_norm", phase_topo_grad_norms)


if __name__ == "__main__":
    main()
