"""Tune one FT-HMC step size and write it for later fixed-step runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nthmc.core.jax_env import bootstrap_cuda_wheel_paths, preconfigure_platform_from_argv, set_platform


bootstrap_cuda_wheel_paths()
preconfigure_platform_from_argv()

from nthmc.u2.field_transform import FieldTransformation
from nthmc.u2.u2_fthmc import HMCU2FT
from nthmc.u2.u2_observables import format_beta, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a 2D U(2) FT-HMC step size")
    parser.add_argument("--lattice_size", type=int, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--train_beta", type=float, required=True)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--ft_step_size", type=float, default=0.1)
    parser.add_argument("--n_tune_steps", type=int, default=200)
    parser.add_argument("--rand_seed", type=int, default=1029)
    parser.add_argument("--model_tag", type=str, default="base")
    parser.add_argument("--save_tag", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "gpu", "cuda"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "gpu" if args.device == "cuda" else args.device
    if device != "auto":
        set_platform(device)
    set_seed(args.rand_seed)

    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir.parents[1] / "artifacts" / "models"
    field_transform = FieldTransformation(
        args.lattice_size,
        device=device,
        n_subsets=8,
        if_check_jac=False,
        model_tag=args.model_tag,
        save_tag=args.save_tag,
        model_dir=model_dir,
        plot_dir=script_dir / "plots",
        dump_dir=script_dir / "dumps",
    )
    field_transform.load_best_model(args.train_beta)

    hmc = HMCU2FT(
        args.lattice_size,
        args.beta,
        0,
        args.n_steps,
        args.ft_step_size,
        field_transformation=field_transform.field_transformation,
        observable_field_transformation=field_transform.field_transformation,
        forward_and_jac_logdet=field_transform.forward_and_jac_logdet,
        device=device,
        tune_step_size=True,
        seed=args.rand_seed,
    )
    print(
        f">>> Tuning L={args.lattice_size} beta={format_beta(args.beta)} "
        f"seed={args.rand_seed} n_tune_steps={args.n_tune_steps}"
    )
    hmc.tune_step_size(n_tune_steps=args.n_tune_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f"{hmc.dt:.6e}\n")
    print(f">>> Wrote FT step size {hmc.dt:.6e} to {args.output}")


if __name__ == "__main__":
    main()
