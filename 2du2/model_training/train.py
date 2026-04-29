"""Train the base neural field transformation for 2D U(2)."""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lightning.fabric import Fabric

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_observables import format_beta, matrix_to_u2, set_seed


def beta_values(min_beta: float, max_beta: float, beta_gap: float) -> np.ndarray:
    return np.arange(min_beta, max_beta + 0.5 * beta_gap, beta_gap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the base U(2) field transformation")
    parser.add_argument("--lattice_size", type=int, default=8)
    parser.add_argument("--min_beta", type=float, required=True)
    parser.add_argument("--max_beta", type=float, required=True)
    parser.add_argument("--beta_gap", type=float, required=True)
    parser.add_argument("--continue_beta", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_subsets", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=0)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=1331)
    parser.add_argument("--if_identity_init", action="store_true")
    parser.add_argument("--if_check_jac", action="store_true")
    parser.add_argument("--if_compile", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--init_std", type=float, default=None)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--devices", default="auto")
    return parser.parse_args()


def load_split_links(data_path: Path) -> torch.Tensor:
    matrices = torch.from_numpy(np.load(data_path))
    return matrix_to_u2(matrices).float()


def main() -> None:
    args = parse_args()
    start_time = time.time()
    save_tag = args.save_tag or f"base_train_b{format_beta(args.min_beta)}_L{args.lattice_size}_{args.rand_seed}"
    torch.set_default_dtype(torch.float32)
    set_seed(args.rand_seed)
    fabric = Fabric(accelerator=args.accelerator, strategy=args.strategy, devices=args.devices)
    fabric.launch()
    device = str(fabric.device)

    script_dir = Path(__file__).resolve().parent
    domain_root = script_dir.parent
    gauge_dir = domain_root / "configs"
    model_dir = domain_root / "artifacts" / "models"
    plot_dir = script_dir / "plots"
    dump_dir = script_dir / "dumps"
    for directory in (model_dir, plot_dir, dump_dir):
        directory.mkdir(parents=True, exist_ok=True)

    hyperparams = {}
    if args.lr is not None:
        hyperparams["lr"] = args.lr
    if args.weight_decay is not None:
        hyperparams["weight_decay"] = args.weight_decay
    if args.init_std is not None:
        hyperparams["init_std"] = args.init_std

    fabric.print("=" * 60)
    fabric.print(">>> U(2) base field-transformation training")
    for key, value in vars(args).items():
        fabric.print(f"{key}: {value}")
    fabric.print(f"save_tag: {save_tag}")
    fabric.print(f"resolved_device: {device}")
    fabric.print(f"torch_cuda_device_count: {torch.cuda.device_count()}")
    fabric.print(f"hyperparams: {hyperparams}")
    fabric.print("=" * 60)

    field_transform = FieldTransformation(
        args.lattice_size,
        device=device,
        n_subsets=args.n_subsets,
        if_check_jac=args.if_check_jac,
        num_workers=args.n_workers,
        identity_init=args.if_identity_init,
        model_tag=args.model_tag,
        save_tag=save_tag,
        model_dir=model_dir,
        plot_dir=plot_dir,
        dump_dir=dump_dir,
        hyperparams=hyperparams,
        fabric=fabric,
        compile_enabled=args.if_compile,
    )

    if args.continue_beta is not None:
        field_transform.load_best_model(args.continue_beta)
        fabric.print(f">>> Continuing from beta={args.continue_beta}")
    else:
        fabric.print(">>> Training from scratch")

    for train_beta in beta_values(args.min_beta, args.max_beta, args.beta_gap):
        beta_start = time.time()
        beta_tag = format_beta(train_beta)
        data_path = gauge_dir / f"links_L{args.lattice_size}_beta{beta_tag}.npy"
        if not data_path.exists():
            raise FileNotFoundError(f"Missing training data: {data_path}")

        data = load_split_links(data_path)
        train_size = int(0.8 * len(data))
        train_data = data[:train_size]
        test_data = data[train_size:]
        fabric.print(f">>> Loaded {data_path}")
        fabric.print(f"Training data shape: {tuple(train_data.shape)}")
        fabric.print(f"Testing data shape: {tuple(test_data.shape)}")

        field_transform.train(train_data, test_data, float(train_beta), n_epochs=args.n_epochs, batch_size=args.batch_size)
        fabric.print(f">>> Completed beta={beta_tag} in {datetime.timedelta(seconds=int(time.time() - beta_start))}")
        fabric.print(f">>> Total elapsed: {datetime.timedelta(seconds=int(time.time() - start_time))}")


if __name__ == "__main__":
    main()
