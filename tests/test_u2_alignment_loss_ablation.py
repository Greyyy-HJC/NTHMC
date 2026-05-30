#!/usr/bin/env python3
"""Short U(2) ablation for alternative topology-alignment losses."""

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
    parser.add_argument("--n_subsets", type=int, default=8)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=1029)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--model_dir", type=Path, default=REPO_ROOT / "2du2" / "artifacts" / "models")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--n_train_steps", type=int, default=3)
    parser.add_argument("--n_eval_batches", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--alignment_losses", nargs="+", default=["force_only", "cos_sq", "smooth_abs", "hinge"])
    parser.add_argument("--alignment_weights", type=float, nargs="+", default=[1.0, 10.0])
    parser.add_argument("--alignment_channel", choices=["full", "phase"], default="phase")
    parser.add_argument("--alignment_eps", type=float, default=1e-6)
    parser.add_argument("--alignment_margin", type=float, default=0.1)
    return parser.parse_args()


def load_links(args: argparse.Namespace) -> torch.Tensor:
    beta_tag = format_beta(args.beta)
    data_path = args.data_path or REPO_ROOT / "2du2" / "configs" / f"links_L{args.lattice_size}_beta{beta_tag}.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing U(2) config file: {data_path}")
    matrices = torch.from_numpy(np.load(data_path))
    links = matrix_to_u2(matrices).float()
    needed = args.train_batch_size * args.n_train_steps + args.eval_batch_size * args.n_eval_batches
    if len(links) < needed:
        raise ValueError(f"Need at least {needed} configurations, got {len(links)} from {data_path}")
    generator = torch.Generator().manual_seed(args.rand_seed)
    return links[torch.randperm(len(links), generator=generator)]


def make_transform(args: argparse.Namespace, device: torch.device) -> FieldTransformation:
    transform = FieldTransformation(
        args.lattice_size,
        device=str(device),
        n_subsets=args.n_subsets,
        model_tag=args.model_tag,
        save_tag=args.save_tag,
        model_dir=args.model_dir,
        hyperparams={
            "lr": args.lr,
            "max_grad_norm": args.max_grad_norm,
        },
    )
    transform.train_beta = args.beta
    if args.save_tag is not None:
        transform.load_best_model(args.beta)
    return transform


def cosine_by_config(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_flat = left.reshape(left.shape[0], -1)
    right_flat = right.reshape(right.shape[0], -1)
    numerator = torch.sum(left_flat * right_flat, dim=1)
    denominator = torch.linalg.vector_norm(left_flat, dim=1) * torch.linalg.vector_norm(right_flat, dim=1)
    return numerator / denominator.clamp_min(torch.finfo(left.dtype).eps)


def alignment_cosine(force: torch.Tensor, topo_grad: torch.Tensor, channel: str) -> torch.Tensor:
    if channel == "phase":
        return cosine_by_config(force[..., :1], topo_grad[..., :1])
    return cosine_by_config(force, topo_grad)


def alignment_loss_from_cos(
    cos: torch.Tensor,
    loss_type: str,
    *,
    eps: float,
    margin: float,
) -> torch.Tensor:
    if loss_type == "force_only":
        return torch.zeros((), device=cos.device, dtype=cos.dtype)
    if loss_type == "cos_sq":
        return -torch.mean(cos**2)
    if loss_type == "smooth_abs":
        return -torch.mean(torch.sqrt(cos**2 + eps))
    if loss_type == "hinge":
        return torch.mean(torch.clamp(margin - torch.abs(cos), min=0.0))
    raise ValueError(f"Unknown alignment loss type: {loss_type}")


def train_step(
    transform: FieldTransformation,
    batch: torch.Tensor,
    args: argparse.Namespace,
    *,
    loss_type: str,
    weight: float,
) -> dict[str, float]:
    transform._set_models_mode(True)
    links_new = transform.inverse(batch)
    force, topo_grad = transform.compute_transformed_force_and_topology_grad(
        links_new,
        args.beta,
        create_graph=True,
    )
    force_loss = transform._weighted_force_loss_tensor(force)
    cos = alignment_cosine(force, topo_grad, args.alignment_channel)
    align_loss = alignment_loss_from_cos(
        cos,
        loss_type,
        eps=args.alignment_eps,
        margin=args.alignment_margin,
    )
    total_loss = force_loss + weight * align_loss

    for optimizer in transform.optimizers:
        optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(total_loss)
    grad_norm = transform._gradient_norm()
    transform._clip_gradients()
    for optimizer in transform.optimizers:
        optimizer.step()

    return {
        "total_loss": float(total_loss.detach().cpu()),
        "force_loss": float(force_loss.detach().cpu()),
        "align_loss": float(align_loss.detach().cpu()),
        "abs_cos": float(torch.mean(torch.abs(cos)).detach().cpu()),
        "cos_sq": float(torch.mean(cos**2).detach().cpu()),
        "grad_norm": grad_norm,
    }


def tensor_summary(values: list[torch.Tensor]) -> tuple[float, float, float, float]:
    tensor = torch.cat([value.detach().cpu().reshape(-1) for value in values])
    return (
        float(tensor.mean()),
        float(tensor.std(unbiased=False)),
        float(tensor.min()),
        float(tensor.max()),
    )


def print_summary(prefix: str, values: list[torch.Tensor]) -> None:
    mean, std, min_value, max_value = tensor_summary(values)
    print(f"{prefix}: mean={mean:.6e} std={std:.6e} min={min_value:.6e} max={max_value:.6e}")


def record_summary(
    metrics: dict[str, float],
    name: str,
    values: list[torch.Tensor],
) -> None:
    mean, std, min_value, max_value = tensor_summary(values)
    metrics[name] = mean
    print(f"{name}: mean={mean:.6e} std={std:.6e} min={min_value:.6e} max={max_value:.6e}")


def evaluate(
    transform: FieldTransformation,
    links: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    transform._set_models_mode(False)
    metrics: dict[str, float] = {}
    force_losses = []
    full_cos = []
    phase_cos = []
    topo_grad_norms = []
    jac_logdets = []

    eval_offset = args.train_batch_size * args.n_train_steps
    eval_stop = eval_offset + args.eval_batch_size * args.n_eval_batches
    for start in range(eval_offset, eval_stop, args.eval_batch_size):
        batch = links[start : start + args.eval_batch_size].to(transform.device)
        with torch.no_grad():
            transformed = transform.inverse(batch)
        with torch.enable_grad():
            force, _, _, jac_logdet, topo_grad = transform.compute_transformed_force_terms(
                transformed.detach(),
                args.beta,
                create_graph=False,
                include_topo_grad=True,
            )
        force_components = transform._force_loss_components(force)
        force_losses.append(torch.tensor([transform._weighted_force_loss(force_components)]))
        full_cos.append(cosine_by_config(force, topo_grad))
        phase_cos.append(cosine_by_config(force[..., :1], topo_grad[..., :1]))
        topo_grad_norms.append(torch.linalg.vector_norm(topo_grad.reshape(topo_grad.shape[0], -1), dim=1))
        jac_logdets.append(jac_logdet)

    record_summary(metrics, "eval_force_weighted_loss", force_losses)
    record_summary(metrics, "eval_force_topo_abs_cos", [torch.abs(value) for value in full_cos])
    record_summary(metrics, "eval_force_topo_cos_sq", [value**2 for value in full_cos])
    record_summary(metrics, "eval_phase_force_topo_abs_cos", [torch.abs(value) for value in phase_cos])
    record_summary(metrics, "eval_phase_force_topo_cos_sq", [value**2 for value in phase_cos])
    record_summary(metrics, "eval_topo_grad_norm", topo_grad_norms)
    record_summary(metrics, "eval_jac_logdet", jac_logdets)
    return metrics


def print_delta_summary(before: dict[str, float], after: dict[str, float]) -> None:
    for name, before_value in before.items():
        after_value = after[name]
        delta = after_value - before_value
        relative = delta / before_value if before_value != 0 else float("nan")
        print(f"delta_{name}: abs={delta:.6e} rel={relative:.6e}")


def variant_specs(args: argparse.Namespace) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for loss_type in args.alignment_losses:
        if loss_type == "force_only":
            specs.append((loss_type, 0.0))
        else:
            specs.extend((loss_type, weight) for weight in args.alignment_weights)
    return specs


def main() -> None:
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    set_seed(args.rand_seed)
    device = torch.device(args.device)
    links = load_links(args)

    print(
        "# U(2) alignment-loss ablation "
        f"L={args.lattice_size} beta={args.beta} save_tag={args.save_tag} "
        f"channel={args.alignment_channel} train_steps={args.n_train_steps} "
        f"train_batch_size={args.train_batch_size} eval_batch_size={args.eval_batch_size} "
        f"n_eval_batches={args.n_eval_batches} lr={args.lr}"
    )
    print(f"# model_dir={args.model_dir}")
    print(f"# alignment_eps={args.alignment_eps} alignment_margin={args.alignment_margin}")

    for loss_type, weight in variant_specs(args):
        print(f"\n=== variant loss_type={loss_type} weight={weight:g} ===")
        set_seed(args.rand_seed)
        transform = make_transform(args, device)

        print("--- before ---")
        before_metrics = evaluate(transform, links, args)

        for step in range(args.n_train_steps):
            start = step * args.train_batch_size
            batch = links[start : start + args.train_batch_size].to(device)
            metrics = train_step(transform, batch, args, loss_type=loss_type, weight=weight)
            print(
                f"step={step + 1} total_loss={metrics['total_loss']:.6e} "
                f"force_loss={metrics['force_loss']:.6e} align_loss={metrics['align_loss']:.6e} "
                f"abs_cos={metrics['abs_cos']:.6e} cos_sq={metrics['cos_sq']:.6e} "
                f"grad_norm={metrics['grad_norm']:.6e}"
            )

        print("--- after ---")
        after_metrics = evaluate(transform, links, args)
        print("--- delta ---")
        print_delta_summary(before_metrics, after_metrics)
        del transform
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
